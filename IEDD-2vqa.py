import argparse
import ast
import glob
import json
import math
import os
import re
from typing import Dict, Tuple, Optional, List, Any

import pandas as pd


# ------------------------ Utils ------------------------ #

def read_csv_smart(path: str) -> pd.DataFrame:
    tried = []
    for enc in ["utf-8", "gbk", "gb2312", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            tried.append(enc)
    raise RuntimeError(f"Failed to read CSV {path} with encodings: {tried}")


def safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return str(value)


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(x)
    if isinstance(x, (int,)):
        return float(x)
    s = safe_str(x).strip()
    if not s:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def first_present(row_dict: dict, keys: List[str]) -> Any:
    """Return the first non-empty field value among keys."""
    for k in keys:
        if k in row_dict:
            v = row_dict.get(k)
            if v is None:
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            s = safe_str(v).strip()
            if s != "":
                return v
    return None


def parse_time_interval(interval_str: str) -> Tuple[Optional[float], Optional[float], str]:
    """Parse '1.7-6.7' style Time Interval. Return (t_start, t_end, raw_clean)."""
    if not isinstance(interval_str, str):
        return None, None, ""
    raw = interval_str.strip()
    parts = raw.split("-")
    if len(parts) != 2:
        return None, None, raw
    try:
        return float(parts[0]), float(parts[1]), raw
    except ValueError:
        return None, None, raw


def _format_time_1dp(t: float) -> str:
    """Match your filename style: 0.0, 3.9, 2.4 ..."""
    return f"{t:.1f}"


def parse_vehicle_pair(pair_str: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(pair_str, str):
        return None, None
    parts = [p.strip() for p in pair_str.split(";") if p.strip()]
    if len(parts) == 0:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[1]


def parse_e_veh(e_str: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not isinstance(e_str, str):
        return result
    items = [x.strip() for x in e_str.split(",") if x.strip()]
    for it in items:
        parts = it.split(":")
        if len(parts) != 2:
            continue
        vid = parts[0].strip()
        try:
            eff = float(parts[1])
        except ValueError:
            continue
        result[vid] = eff
    return result


def normalize_interaction_type(s: str) -> str:
    if not isinstance(s, str):
        return "crossing"
    x = " ".join(s.strip().lower().replace("_", " ").replace("-", " ").split())
    if "follow" in x:
        return "car follow"
    if "merge" in x:
        return "merging"
    if "cross" in x:
        return "crossing"
    if "head" in x and "on" in x:
        return "head on"
    return x if x in {"car follow", "merging", "crossing", "head on"} else "crossing"


# ------------------------ Robust Peak-Q parsing ------------------------ #

_TRIPLET_RE = re.compile(
    r"(?P<veh>[A-Za-z0-9_\-]+)\s*[:\|,]\s*(?P<t>-?\d+(?:\.\d+)?)\s*[:\|,]\s*(?P<q>-?\d+(?:\.\d+)?)"
)

_KV_RE = re.compile(
    r"(?:veh|vehicle)\s*=\s*(?P<veh>[A-Za-z0-9_\-]+).*?"
    r"(?:t|time)\s*=\s*(?P<t>-?\d+(?:\.\d+)?).*?"
    r"(?:q)\s*=\s*(?P<q>-?\d+(?:\.\d+)?)",
    re.IGNORECASE
)

def parse_peak_from_q_text(q_text: Any) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Try to parse peak (veh, t, q) from a text field that may contain:
      - single triplet: "A:1.2:0.9"
      - multiple triplets: "A:1.2:0.6;B:1.3:0.9"
      - brackets/quotes: "['A:1.2:0.9']"
      - key-value: "veh=A t=1.2 q=0.9"
      - JSON / python-literal containers
    Strategy: extract candidates and take max q.
    """
    if q_text is None:
        return None, None, None

    if isinstance(q_text, (int, float)) and not (isinstance(q_text, float) and (math.isnan(q_text) or math.isinf(q_text))):
        return None, None, float(q_text)

    s = safe_str(q_text).strip()
    if not s:
        return None, None, None

    obj = None
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                obj = None

    candidates: List[Tuple[Optional[str], Optional[float], Optional[float]]] = []

    def add_candidate(veh, t, q):
        tt = safe_float(t)
        qq = safe_float(q)
        vv = safe_str(veh).strip() if veh is not None else None
        if qq is None:
            return
        candidates.append((vv if vv else None, tt, qq))

    def walk(o):
        if isinstance(o, dict):
            if any(k in o for k in ["veh", "vehicle", "id"]) and any(k in o for k in ["t", "time"]) and "q" in o:
                add_candidate(o.get("veh") or o.get("vehicle") or o.get("id"), o.get("t") or o.get("time"), o.get("q"))
                return
            for k, v in o.items():
                if isinstance(v, dict) and ("q" in v or "peak_q" in v):
                    add_candidate(k, v.get("t") or v.get("time") or v.get("peak_t"), v.get("q") or v.get("peak_q"))
                elif isinstance(v, (list, tuple)):
                    walk(v)
        elif isinstance(o, (list, tuple)):
            for it in o:
                if isinstance(it, dict):
                    if any(k in it for k in ["veh", "vehicle", "id"]) and any(k in it for k in ["t", "time"]) and "q" in it:
                        add_candidate(it.get("veh") or it.get("vehicle") or it.get("id"), it.get("t") or it.get("time"), it.get("q"))
                    else:
                        walk(it)
                elif isinstance(it, (list, tuple)) and len(it) >= 3:
                    add_candidate(it[0], it[1], it[2])
                else:
                    if isinstance(it, str):
                        for m in _TRIPLET_RE.finditer(it):
                            add_candidate(m.group("veh"), m.group("t"), m.group("q"))

    if obj is not None:
        walk(obj)

    m = _KV_RE.search(s)
    if m:
        add_candidate(m.group("veh"), m.group("t"), m.group("q"))

    for m in _TRIPLET_RE.finditer(s):
        add_candidate(m.group("veh"), m.group("t"), m.group("q"))

    if not candidates:
        return None, None, None

    best = max(candidates, key=lambda x: (-1e18 if x[2] is None else x[2]))
    return best[0], best[1], best[2]


def parse_peak_q_from_row(row_dict: dict) -> Tuple[Optional[str], Optional[float], Optional[float], str]:
    """
    Return (peak_veh, peak_t, peak_q, raw_q_field_used).
    Priority:
      1) explicit Peak_* columns if present
      2) parse from Q_i_Over_Time-like text
    """
    veh = first_present(row_dict, ["Peak_Veh", "Peak_veh", "Peak Vehicle", "Peak_Vehicle", "Q_Peak_Veh", "Q_peak_veh"])
    tt  = first_present(row_dict, ["Peak_T", "Peak_t", "Peak Time", "Peak_Time", "Q_Peak_T", "Q_peak_t"])
    qq  = first_present(row_dict, ["Peak_Q", "Peak_q", "Peak Q", "Peak_Q_Value", "Q_Peak", "Q_peak"])

    peak_q = safe_float(qq)
    peak_t = safe_float(tt)
    peak_veh = safe_str(veh).strip() if veh is not None else None

    if peak_q is not None:
        return peak_veh or None, peak_t, peak_q, "Peak_*"

    q_field = first_present(row_dict, ["Q_i_Over_Time", "Q_i_over_time", "Q_i", "Q_Over_Time", "Q_over_time", "Q"])
    if q_field is None:
        return None, None, None, ""

    pv, pt, pq = parse_peak_from_q_text(q_field)
    return pv, pt, pq, "Q_i_Over_Time"


# ------------------------ Actions JSON locating ------------------------ #

def scene_to_token(scene: str) -> str:
    """
    'interaction_multi/DR_USA_Intersection_GL_train_9' -> 'interaction_multi_DR_USA_Intersection_GL_train_9'
    """
    return safe_str(scene).strip().replace("\\", "/").strip("/").replace("/", "_")


def build_actions_base_id(scene: str, row_pos: int, t_start: Optional[float], t_end: Optional[float], raw_interval: str) -> str:
    token = scene_to_token(scene) if scene else ""
    if t_start is not None and t_end is not None:
        interval = f"{_format_time_1dp(t_start)}-{_format_time_1dp(t_end)}"
    else:
        interval = raw_interval if raw_interval else "unknown"
    return f"{token}_row{row_pos:04d}_t{interval}"


def load_actions_json(actions_dir: str, base_id: str) -> Optional[dict]:
    """
    Exact match:
      {actions_dir}/{base_id}.actions.json
    Fallback:
      glob {actions_dir}/{sceneToken}_rowXXXX_t*.actions.json
    """
    if not actions_dir or not os.path.isdir(actions_dir) or not base_id:
        return None

    exact = os.path.join(actions_dir, f"{base_id}.actions.json")
    if os.path.isfile(exact):
        try:
            with open(exact, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    prefix = base_id.split("_t", 1)[0]
    pattern = os.path.join(actions_dir, f"{prefix}_t*.actions.json")
    matches = sorted(glob.glob(pattern))
    for p in matches:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return None


def summarize_vehicle_action(actions_json: dict, veh_id: str) -> str:
    if not actions_json or not veh_id:
        return "not available"
    vehicles = (((actions_json.get("actions") or {}).get("vehicles")) or {})
    v = vehicles.get(veh_id)
    if not isinstance(v, dict):
        return "not available"

    summ = v.get("summary") or {}
    dlong = summ.get("dominant_long")
    dlat = summ.get("dominant_lat")
    if dlong and dlat:
        return f"{dlong}+{dlat}"

    segs = v.get("segments") or []
    if isinstance(segs, list) and segs and isinstance(segs[0], dict):
        act = segs[0].get("action")
        if act:
            return str(act)

    return "not available"


# ------------------------ ShareGPT formatting ------------------------ #

def build_system_prompt(ego_id: str) -> str:
    return (
        "You are a Vision Language Model specialized in understanding a traffic interaction scenario "
        "from the perspective of the bird's eye view.\n"
        f"You MUST reason from the ego vehicle (ID: {ego_id}) viewpoint.\n\n"
        "The interaction type is one of four categories:\n"
        "- car follow: the ego and another vehicle travel in the same direction, with one following the other.\n"
        "- merging: two vehicles approach and then join into the same lane/stream.\n"
        "- crossing: two vehicles' paths cross (often at an intersection) and they must negotiate right-of-way.\n"
        "- head on: two vehicles approach each other in opposite directions with potential conflict.\n\n"
        "Answer each question briefly (1–2 sentences) without long step-by-step reasoning."
    )


def build_conversations(
    scene: str,
    ego_id: str,
    opp_id: str,
    interaction_type_norm: str,
    peak_veh: Optional[str],
    peak_t: Optional[float],
    peak_q: Optional[float],
    e_veh_map: Dict[str, float],
    ego_action: str,
    opp_action: str,
) -> List[Dict[str, str]]:
    conv: List[Dict[str, str]] = []

    conv.append({"from": "human", "value": "<video>\nWhich vehicle is the ego vehicle interacting with?"})
    if opp_id:
        conv.append({"from": "gpt", "value": f"In scene {scene}, the ego vehicle ({ego_id}) is interacting with vehicle {opp_id}."})
    else:
        conv.append({"from": "gpt", "value": f"In scene {scene}, the ego vehicle ({ego_id}) is interacting with another vehicle (opponent ID not available)."})

    conv.append({"from": "human", "value": "<video>\nWhich type of interaction is this (car follow, merging, crossing, or head on)?"})
    conv.append({"from": "gpt", "value": f"This interaction is classified as **{interaction_type_norm}**."})

    conv.append({"from": "human", "value": "<video>\nWhat interaction action does the ego vehicle take, and what action does the opponent vehicle take?"})
    if opp_id:
        conv.append({"from": "gpt", "value": f"Ego ({ego_id}) mainly performs **{ego_action}**, while the opponent ({opp_id}) mainly performs **{opp_action}**."})
    else:
        conv.append({"from": "gpt", "value": f"Ego ({ego_id}) mainly performs **{ego_action}**. The opponent action is not available."})


    if isinstance(peak_q, (int, float)) and peak_q is not None:
        conv.append({"from": "human", "value": "<video>\nWhen does the interaction intensity (Q) peak, and what is the peak value?"})
        risk_level = "low"
        if peak_q >= 0.9:
            risk_level = "high"
        elif peak_q >= 0.6:
            risk_level = "medium"
        who = peak_veh or ego_id
        if peak_t is None:
            conv.append({"from": "gpt", "value": f"Q peaks for vehicle {who} with Q ≈ {peak_q:.2f}, indicating a {risk_level}-strength interaction."})
        else:
            conv.append({"from": "gpt", "value": f"Q peaks for vehicle {who} at about t = {peak_t:.1f} s with Q ≈ {peak_q:.2f}, indicating a {risk_level}-strength interaction."})

    conv.append({"from": "human", "value": "<video>\nHow do the efficiency values (E) of the ego and opponent vehicles compare?"})
    ego_eff = e_veh_map.get(ego_id)
    opp_eff = e_veh_map.get(opp_id) if opp_id else None
    if ego_eff is not None and opp_eff is not None:
        diff = abs(ego_eff - opp_eff)
        if diff < 1e-3:
            conv.append({"from": "gpt", "value": f"The ego and opponent have very similar efficiency values (E ≈ {ego_eff:.3f} vs {opp_eff:.3f})."})
        elif ego_eff > opp_eff:
            conv.append({"from": "gpt", "value": f"The ego has higher efficiency (E ≈ {ego_eff:.3f}) than the opponent (E ≈ {opp_eff:.3f})."})
        else:
            conv.append({"from": "gpt", "value": f"The ego has lower efficiency (E ≈ {ego_eff:.3f}) than the opponent (E ≈ {opp_eff:.3f})."})
    else:
        conv.append({"from": "gpt", "value": "Efficiency values are incomplete, so the ego–opponent E comparison is not available."})

    return conv


# ------------------------ Main processing ------------------------ #

def find_interaction_csvs(root_dir: str) -> List[str]:
    csv_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "interaction.csv" in filenames:
            csv_paths.append(os.path.join(dirpath, "interaction.csv"))
    return csv_paths


def maybe_video_path(videos_dir: Optional[str], base_id: str) -> str:
    if not videos_dir:
        return ""
    mp4 = os.path.join(videos_dir, f"{base_id}.mp4")
    return mp4 if os.path.isfile(mp4) else ""


def process_root_to_sharegpt_json(
    root_dir: str,
    actions_dir: str,
    videos_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    debug_q: bool = False,
):
    if output_path is None:
        output_path = os.path.join(root_dir, "waymo_test_sharegpt_en.json")

    csv_paths = find_interaction_csvs(root_dir)
    if not csv_paths:
        raise RuntimeError(f"No interaction.csv files found under {root_dir}")

    all_samples: List[dict] = []
    bad_q = 0

    for csv_path in csv_paths:
        df = read_csv_smart(csv_path)

        for row_pos, (_, row) in enumerate(df.iterrows()):
            row_dict = dict(row)

            scene = safe_str(row_dict.get("Scene", ""))
            t_start, t_end, raw_interval = parse_time_interval(safe_str(row_dict.get("Time Interval", "")))

            peak_veh, peak_t, peak_q, q_used = parse_peak_q_from_row(row_dict)
            if peak_q is None:
                bad_q += 1
                if debug_q and bad_q <= 30:
                    raw_q = first_present(row_dict, ["Q_i_Over_Time", "Q_i_over_time", "Q_i", "Q_Over_Time", "Q_over_time", "Q"])
                    print(f"[debug_q] parse failed: csv={csv_path}, row={row_pos}, q_used={q_used}, raw_q={safe_str(raw_q)[:200]}")

            v1, v2 = parse_vehicle_pair(safe_str(row_dict.get("Vehicle Pair", "")))

            ego_id = safe_str(peak_veh or v2 or v1 or "ego")
            opp_id = ""
            for vid in [v1, v2]:
                if vid and safe_str(vid) != ego_id:
                    opp_id = safe_str(vid)
                    break

            interaction_type_norm = normalize_interaction_type(safe_str(row_dict.get("Interaction Type", "crossing")))
            e_veh_map = parse_e_veh(safe_str(row_dict.get("E_Veh", "")))

            base_id = build_actions_base_id(
                scene=scene,
                row_pos=row_pos,
                t_start=t_start,
                t_end=t_end,
                raw_interval=raw_interval,
            )

            actions_json = load_actions_json(actions_dir, base_id)
            ego_action = summarize_vehicle_action(actions_json, ego_id)
            opp_action = summarize_vehicle_action(actions_json, opp_id) if opp_id else "not available"

            sample = {
                "id": base_id,
                "video": maybe_video_path(videos_dir, base_id),  
                "conversations": build_conversations(
                    scene=scene,
                    ego_id=ego_id,
                    opp_id=opp_id,
                    interaction_type_norm=interaction_type_norm,
                    peak_veh=peak_veh,
                    peak_t=peak_t,
                    peak_q=peak_q,
                    e_veh_map=e_veh_map,
                    ego_action=ego_action,
                    opp_action=opp_action,
                ),
                "system": build_system_prompt(ego_id=ego_id),
            }
            all_samples.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_samples)} ShareGPT samples to {output_path}")
    if debug_q:
        print(f"[debug_q] peak_q missing in {bad_q} rows (printed up to 30 examples).")


def main():
    parser = argparse.ArgumentParser(
        description="Convert interaction.csv rows to ShareGPT-format QA JSON, "
                    "and locate actions json strictly by Scene+row+Time Interval."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="root_dir",
    )
    parser.add_argument(
        "--actions_dir",
        type=str,
        default="/actions",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="/videos",
        help="Optional: if {videos_dir}/{id}.mp4 exists, write it into 'video'."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="XXX.json",
        help="If empty, defaults to {root_dir}/XXX.json"
    )
    parser.add_argument(
        "--debug_q",
        action="store_true",
        help="Print raw Q field examples when peak parsing fails (up to 30)."
    )
    args = parser.parse_args()

    out = args.output_json.strip() or None
    process_root_to_sharegpt_json(
        root_dir=args.root_dir,
        actions_dir=args.actions_dir,
        videos_dir=args.videos_dir if args.videos_dir.strip() else None,
        output_path=out,
        debug_q=args.debug_q,
    )


if __name__ == "__main__":
    main()
