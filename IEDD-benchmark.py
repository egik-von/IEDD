import os
import sys
import json
import cv2
import base64
import re
import time
import numpy as np
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

try:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    HAS_ROUGE = True
except ImportError:
    print("[Warning] rouge-score not found. Install via 'pip install rouge-score' for academic metrics.")
    HAS_ROUGE = False



OPENROUTER_API_KEY = "XXX"
BASE_URL = "https://openrouter.ai/api/v1"

INPUT_JSON_PATH = "XXX.json"


TARGET_MODELS = [
    # "moonshotai/kimi-k2.5"
    # "bytedance-seed/seed-1.6-flash"
    # "openai/gpt-4o",
    # "google/gemini-2.5-flash-lite"
    # "x-ai/grok-4.1-fast"
    # "anthropic/claude-3-haiku"
    # "z-ai/glm-4.6v"
    # "meta-llama/llama-4-maverick"
    "amazon/nova-2-lite-v1"
]

#  (LLM-as-a-Judge)
JUDGE_MODEL = "z-ai/glm-4.7-flash"


NUM_FRAMES = 6
IMAGE_SIZE = (512, 512)
VLM_TIMEOUT = 1200

RESULT_DIR = "./benchmark_results"



def now_ts_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_model_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def jsonable(obj):

    if obj is None:
        return None
 
    for attr in ("model_dump", "dict"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass

    for attr in ("json", "to_json"):
        if hasattr(obj, attr):
            try:
                return json.loads(getattr(obj, attr)())
            except Exception:
                pass

    try:
        return str(obj)
    except Exception:
        return None


def write_jsonl(fp, record: dict):
    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    fp.flush()


def encode_image_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def process_video(video_path, num_frames=6):
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames_base64 = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.resize(frame, IMAGE_SIZE)
            frames_base64.append(encode_image_base64(frame))
    cap.release()
    return frames_base64


def extract_number(text):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        try:
            return float(matches[0])
        except Exception:
            return None
    return None


def calculate_iou(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def calculate_rouge_l(pred, gt):
    if not HAS_ROUGE:
        return 0.0
    scores = scorer.score(gt, pred)
    return scores['rougeL'].fmeasure


class VLM_Benchmark:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=api_key,
        )

    def call_vlm(self, model_name, messages):
        start_time = time.time()
        last_err = None
        while True:
            elapsed = time.time() - start_time
            if elapsed > VLM_TIMEOUT:
                print(f"\n[FATAL] Timeout for {model_name}. Last error: {last_err}")
                sys.exit(1)
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                if content and len(content.strip()) > 0:
                    return content, jsonable(response)
                else:
                    last_err = "empty_response"
                    print(f"[Warn] Empty response, retrying...")
                    time.sleep(2)
            except Exception as e:
                last_err = str(e)
                if "429" in last_err:
                    time.sleep(10)
                else:
                    time.sleep(5)

    def call_judge(self, question, gt_answer, pred_answer, type_desc):
        prompt = f"""
Role: Academic Evaluator for Autonomous Driving.
Task: Compare Model Prediction against Ground Truth.
Context: {type_desc}

Question: {question}
GT: {gt_answer}
Pred: {pred_answer}

Criteria:
- Accuracy of driving physics/logic.
- Correctness of actions (stop, go, yield).
- Hallucination check.

Output: ONLY a float score 0.0 to 10.0.
""".strip()

        last_err = None
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                raw_text = (response.choices[0].message.content or "").strip()
                score = extract_number(raw_text)
                if score is None:
                    score = 0.0
                return float(score), raw_text, jsonable(response), prompt
            except Exception as e:
                last_err = str(e)
                time.sleep(2)


        return 0.0, f"[ERROR]{last_err}", None, prompt

    def evaluate_sample(self, sample, model_name, writers):
        """
        writers: dict with open file handles:
          - vlm_fp
          - judge_fp
          - sample_score_fp
        """
        video_path = sample.get('video', '')
        frames = process_video(video_path, NUM_FRAMES)
        if not frames:
            return None, []

        messages = []
        content_payload = [{"type": "text", "text": sample.get('system', '')}]
        for b64_img in frames:
            content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

        dialogue = sample.get('conversations', [])

        metrics = {
            "id_iou": 0.0,
            "type_acc": 0.0,
            "action_judge": 0.0,
            "action_rouge": 0.0,
            "q_mae": None,
            "q_score": 0.0,
            "eff_acc": 0.0,
            "reason_judge": 0.0,
            "reason_rouge": 0.0
        }

        logs = []
        interaction_type = "unknown"

        for i in range(0, len(dialogue), 2):
            user_q = dialogue[i]['value'].replace("<video>\n", "")
            gt_a = dialogue[i + 1]['value']
            question_idx = i // 2 + 1

            if i == 0:
                content_payload.append({"type": "text", "text": user_q})
                messages.append({"role": "user", "content": content_payload})
            else:
                messages.append({"role": "user", "content": user_q})

            # === call VLM ===
            pred_a, vlm_raw_resp = self.call_vlm(model_name, messages)
            messages.append({"role": "assistant", "content": pred_a})


            write_jsonl(writers["vlm_fp"], {
                "ts": time.time(),
                "sample_id": sample.get("id"),
                "video": video_path,
                "model": model_name,
                "question_idx": question_idx,
                "question": user_q,
                "gt": gt_a,
                "pred": pred_a,
                "vlm_raw_response": vlm_raw_resp,
            })

            if question_idx == 1:
                gt_ids = set(re.findall(r'\d+', gt_a))
                pred_ids = set(re.findall(r'\d+', pred_a))
                metrics['id_iou'] = calculate_iou(gt_ids, pred_ids)

            elif question_idx == 2:
                types = ["car follow", "merging", "crossing", "head on"]
                pred_clean = pred_a.lower()
                matched_type = next((t for t in types if t in pred_clean), None)
                gt_clean = gt_a.lower()
                gt_type = next((t for t in types if t in gt_clean), None)
                metrics['type_acc'] = 1.0 if (matched_type == gt_type and gt_type) else 0.0
                interaction_type = gt_type or "unknown"

            elif question_idx == 3:
                score, judge_text, judge_raw_resp, judge_prompt = self.call_judge(
                    user_q, gt_a, pred_a, "Action Description"
                )
                metrics['action_judge'] = score
                metrics['action_rouge'] = calculate_rouge_l(pred_a, gt_a)


                write_jsonl(writers["judge_fp"], {
                    "ts": time.time(),
                    "sample_id": sample.get("id"),
                    "video": video_path,
                    "target_model": model_name,
                    "judge_model": JUDGE_MODEL,
                    "question_idx": question_idx,
                    "type_desc": "Action Description",
                    "judge_prompt": judge_prompt,
                    "judge_raw_text": judge_text,
                    "judge_score": score,
                    "judge_raw_response": judge_raw_resp,
                })

            elif question_idx == 4:
                if interaction_type == "car follow":
                    metrics['q_mae'] = -1
                    metrics['q_score'] = -1
                else:
                    v_gt = extract_number(gt_a)
                    v_pred = extract_number(pred_a)
                    if v_gt is not None and v_pred is not None:
                        abs_err = abs(v_gt - v_pred)
                        metrics['q_mae'] = abs_err
                        metrics['q_score'] = max(0, 1 - abs_err / 0.5)
                    else:
                        metrics['q_mae'] = 10.0
                        metrics['q_score'] = 0.0

            elif question_idx == 5:
                nums_gt = re.findall(r"[-+]?\d*\.\d+|\d+", gt_a)
                nums_pred = re.findall(r"[-+]?\d*\.\d+|\d+", pred_a)
                logic_correct = False
                if len(nums_gt) >= 2 and len(nums_pred) >= 2:
                    gt_rel = float(nums_gt[0]) < float(nums_gt[1])
                    pred_rel = float(nums_pred[0]) < float(nums_pred[1])
                    if gt_rel == pred_rel:
                        logic_correct = True
                metrics['eff_acc'] = 1.0 if logic_correct else 0.0

            elif question_idx == 6:
                score, judge_text, judge_raw_resp, judge_prompt = self.call_judge(
                    user_q, gt_a, pred_a, "Counterfactual Reasoning"
                )
                metrics['reason_judge'] = score
                metrics['reason_rouge'] = calculate_rouge_l(pred_a, gt_a)


                write_jsonl(writers["judge_fp"], {
                    "ts": time.time(),
                    "sample_id": sample.get("id"),
                    "video": video_path,
                    "target_model": model_name,
                    "judge_model": JUDGE_MODEL,
                    "question_idx": question_idx,
                    "type_desc": "Counterfactual Reasoning",
                    "judge_prompt": judge_prompt,
                    "judge_raw_text": judge_text,
                    "judge_score": score,
                    "judge_raw_response": judge_raw_resp,
                })

            logs.append(f"Q{question_idx} | GT: {gt_a[:40]}... | Pred: {pred_a[:40]}...")


        write_jsonl(writers["sample_score_fp"], {
            "ts": time.time(),
            "sample_id": sample.get("id"),
            "video": video_path,
            "model": model_name,
            "metrics": metrics,
        })

        return metrics, logs



def main():
    if not OPENROUTER_API_KEY:
        print("Please set OPENROUTER_API_KEY env var, e.g. `export OPENROUTER_API_KEY=...`")
        return

    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)
    runner = VLM_Benchmark(OPENROUTER_API_KEY)

    for model in TARGET_MODELS:
        print(f"\n======== Academic Eval: {model} ========")

        safe_name = safe_model_name(model)
        run_dir = os.path.join(RESULT_DIR, f"{safe_name}_{now_ts_str()}")
        os.makedirs(run_dir, exist_ok=True)

        vlm_raw_path = os.path.join(run_dir, "vlm_raw.jsonl")
        judge_raw_path = os.path.join(run_dir, "judge_raw.jsonl")
        sample_score_path = os.path.join(run_dir, "sample_scores.jsonl")
        summary_json_path = os.path.join(run_dir, "summary.json")
        summary_txt_path = os.path.join(run_dir, "summary.txt")


        agg_metrics = {k: [] for k in [
            "id_iou", "type_acc",
            "action_judge", "action_rouge",
            "q_mae", "q_score",
            "eff_acc",
            "reason_judge", "reason_rouge"
        ]}

        detailed_logs = []

        with open(vlm_raw_path, "w", encoding="utf-8") as vlm_fp, \
             open(judge_raw_path, "w", encoding="utf-8") as judge_fp, \
             open(sample_score_path, "w", encoding="utf-8") as sample_score_fp:

            writers = {
                "vlm_fp": vlm_fp,
                "judge_fp": judge_fp,
                "sample_score_fp": sample_score_fp,
            }

            for sample in tqdm(data):
                metrics, logs = runner.evaluate_sample(sample, model, writers)
                if metrics is None:
                    continue

                for k, v in metrics.items():
                    if v is not None and v != -1:
                        agg_metrics[k].append(v)

                log_str = f"ID: {sample.get('id')}\n" + "\n".join(logs) + "\n" + "-" * 30
                detailed_logs.append(log_str)


        results = {}
        for k, v_list in agg_metrics.items():
            if v_list:
                results[k] = float(np.mean(v_list))
            else:
                results[k] = 0.0


        L1 = (results['id_iou'] + results['type_acc']) / 2
        L2 = results['action_judge'] / 10.0
        L3 = (results['q_score'] + results['eff_acc']) / 2
        L4 = results['reason_judge'] / 10.0
        WIS = 0.2 * L1 + 0.2 * L2 + 0.2 * L3 + 0.4 * L4


        summary = {
            "model": model,
            "judge_model": JUDGE_MODEL,
            "run_dir": run_dir,
            "wis": WIS,
            "results_mean": results,
            "wis_components": {
                "L1_perception": L1,
                "L2_description": L2,
                "L3_quant": L3,
                "L4_reasoning": L4
            },
            "paths": {
                "vlm_raw_jsonl": vlm_raw_path,
                "judge_raw_jsonl": judge_raw_path,
                "sample_scores_jsonl": sample_score_path
            }
        }

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        report = f"""
Academic Benchmark Report for Model: {model}
===============================================================
Overall WIS Score (Weighted): {WIS:.4f}

[Level 1: Perception]
- Object ID IoU (Intersection over Union): {results['id_iou']:.4f}
- Interaction Classification Accuracy:     {results['type_acc']:.4f}

[Level 2: Description]
- Action Semantics (LLM-Judge /10):        {results['action_judge']:.2f}
- Text Overlap (ROUGE-L):                  {results['action_rouge']:.4f} *Auxiliary

[Level 3: Quantitative]
- Numerical MAE (Mean Absolute Error):     {results['q_mae']:.4f} (Lower is better)
- Logic Consistency Accuracy:              {results['eff_acc']:.4f}

[Level 4: Reasoning]
- Reasoning Score (LLM-Judge /10):         {results['reason_judge']:.2f}
- Text Overlap (ROUGE-L):                  {results['reason_rouge']:.4f} *Auxiliary
===============================================================

[Saved Files]
- VLM raw outputs : {vlm_raw_path}
- Judge raw outputs: {judge_raw_path}
- Sample scores    : {sample_score_path}
- Summary (json)    : {summary_json_path}
- Summary (txt)     : {summary_txt_path}
""".strip()

        print(report)
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
