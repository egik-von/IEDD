import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import argparse
import logging
import math
import re
import gc
import glob
import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm


from trajdata import UnifiedDataset, MapAPI
from utils.IEDD_InteractionProcessor import SceneProcessor
from utils.IEDD_compute import generate_random_process


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CounterfactualGen")




def rect_corners(x: float, y: float, w: float, l: float, heading: float) -> np.ndarray:
  
    c, s = math.cos(heading), math.sin(heading)
    dx, dy = l / 2.0, w / 2.0
    # local corners
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts @ R.T) + np.array([x, y], dtype=np.float64)

def _project(poly: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    vals = poly @ axis
    return float(vals.min()), float(vals.max())

def obb_intersect(poly1: np.ndarray, poly2: np.ndarray, eps: float = 1e-12) -> bool:
  

    for poly in (poly1, poly2):
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            edge = p1 - p0
            
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm < eps:
                continue
            axis /= norm
            min1, max1 = _project(poly1, axis)
            min2, max2 = _project(poly2, axis)
            if max1 < min2 - eps or max2 < min1 - eps:
                return False
    return True


def run_counterfactual_simulation(process, ego_idx, partner_idx, t_start, t_end, sim_duration=4.0):
 
    
    if t_start not in process.states:
        times = sorted(process.states.keys())
        if not times:
            return {"valid": False, "reason": "Empty states"}
        t_start = min(times, key=lambda t: abs(t - t_start))

    ego_init = process.states[t_start].get(ego_idx)
    partner_init = process.states[t_start].get(partner_idx)

    if (ego_init is None) or (partner_init is None):
        return {"valid": False, "reason": f"State missing for idx {ego_idx}/{partner_idx} at t={t_start:.1f}"}

    if np.isnan(ego_init.v) or np.isnan(ego_init.x) or np.isnan(ego_init.y):
        return {"valid": False, "reason": "NaN values in ego state"}

    ego_meta = process.vehicle_metadata.get(ego_idx, {"length": 4.5, "width": 2.0})
    partner_meta = process.vehicle_metadata.get(partner_idx, {"length": 4.5, "width": 2.0})

    v_init = float(ego_init.v)
    heading_init = float(ego_init.phi)

    if v_init < 0.1:
        return {"valid": False, "reason": "Ego stopped"}

    dt = 0.1
    steps = int(sim_duration / dt)
    collision_detected = False
    collision_time = None

    sorted_times = sorted([t for t in process.states.keys() if t >= t_start])
    if not sorted_times:
        return {"valid": False, "reason": "No future states for partner"}

    for i in range(steps):
        sim_time = t_start + i * dt

        pred_x = ego_init.x + v_init * math.cos(heading_init) * (i * dt)
        pred_y = ego_init.y + v_init * math.sin(heading_init) * (i * dt)

     
        closest_t = min(sorted_times, key=lambda t: abs(t - sim_time))
        if abs(closest_t - sim_time) > 0.5:
            break

        partner_state = process.states[closest_t].get(partner_idx)
        if (partner_state is None) or np.isnan(partner_state.x) or np.isnan(partner_state.y):
            continue

    
        try:
            ego_poly = rect_corners(pred_x, pred_y, ego_meta["width"], ego_meta["length"], heading_init)
            partner_poly = rect_corners(
                partner_state.x, partner_state.y,
                partner_meta["width"], partner_meta["length"], float(partner_state.phi)
            )
            if obb_intersect(ego_poly, partner_poly):
                collision_detected = True
                collision_time = sim_time
                break
        except Exception:
            continue

    return {
        "valid": True,
        "collision": collision_detected,
        "ttc_counterfactual": (collision_time - t_start) if collision_time is not None else None,
        "initial_speed": v_init
    }



def parse_scene_key(video_path: str) -> Optional[Tuple[int, int, float, float]]:
    base = os.path.basename(video_path)
    name_no_ext = os.path.splitext(base)[0]
    m = re.search(r"scene_(\d+)_row(\d+)_t([\d\.]+)-([\d\.]+)", name_no_ext)
    if not m:
        return None
    scene_num = int(m.group(1))
    row_idx = int(m.group(2))
    t0 = float(m.group(3))
    t1 = float(m.group(4))
    return scene_num, row_idx, t0, t1





def find_csv_for_scene(input_dir: str, scene_num: int) -> Optional[str]:
    candidate = os.path.join(input_dir, f"scene_{scene_num}", "interaction.csv")
    if os.path.isfile(candidate):
        return candidate
    pattern = os.path.join(input_dir, "**", f"scene_{scene_num}", "interaction.csv")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None

def parse_vehicle_pair(pair_str: str) -> Optional[Tuple[str, str]]:
    s = str(pair_str).strip()
    if not s:
        return None
    toks = [t for t in re.split(r"[;:,\s]+", s) if t]
    if len(toks) < 2:
        return None
    return toks[0].strip(), toks[1].strip()




def discover_leaf_cache_dirs(cache_root: str, depth: int = 3) -> List[str]:
  
    if not os.path.isdir(cache_root):
        return []

    cands = set()


    patterns = [
        os.path.join(cache_root, "*"),
        os.path.join(cache_root, "*", "*"),
        os.path.join(cache_root, "*", "*", "*"),
    ]
    for pat in patterns[:max(1, min(depth, 3))]:
        for p in glob.glob(pat):
            if os.path.isdir(p):
                cands.add(os.path.abspath(p))

    return sorted(cands)

def build_scene_index(cache_leaf_dirs: List[str], desired_data: str) -> Dict[str, str]:

    scene_to_cache: Dict[str, str] = {}
    logger.info("Indexing scenes from cache directories...")

    for cache_path in cache_leaf_dirs:
        try:
            ds = UnifiedDataset(
                desired_data=[desired_data],
                cache_location=cache_path,
                rebuild_cache=False,
                rebuild_maps=False,
                standardize_data=False,
                centric="scene",
                verbose=False,
                num_workers=0,
                incl_vector_map=False,
                data_dirs={desired_data: ""}
            )
          
            scenes = list(ds.scenes())
            if len(scenes) == 0:
                del ds
                gc.collect()
                continue

            for s in scenes:
                s_str = str(s)
                short_name = s_str.split("/")[-1] if "/" in s_str else s_str
                scene_to_cache[short_name] = cache_path
                scene_to_cache[s_str] = cache_path

            del ds, scenes
            gc.collect()

        except Exception as e:
          
            logger.warning(f"Failed to index {cache_path}: {e}")

    return scene_to_cache




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-input", type=str,
                        default="XXX.json")
    parser.add_argument("--json-output", type=str,
                        default="XXX.json")
    parser.add_argument("--input-dir", type=str,
                        default="csv_dir")
    parser.add_argument("--cache-root", type=str, default="cache_root")
    parser.add_argument("--desired-data", type=str, default="desired_data")
    parser.add_argument("--timerange", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    logger.info("Reading JSON...")
    with open(args.json_input, "r") as f:
        data = json.load(f)


    scene_tasks: Dict[str, List[Dict]] = {}
    for item in data:
        video_path = item.get("video", "")
        parsed = parse_scene_key(video_path)
        if not parsed:
            continue
        scene_num, row_idx, t0, t1 = parsed
        scene_key = f"scene_{scene_num}"
        scene_tasks.setdefault(scene_key, []).append({
            "item": item,
            "meta": {"row_idx": row_idx, "t_start": t0, "t_end": t1},
            "scene_num": scene_num
        })

    logger.info(f"Total items: {len(data)}, Unique scenes: {len(scene_tasks)}")

    
    cache_leaf_dirs = discover_leaf_cache_dirs(args.cache_root, depth=3)
    if not cache_leaf_dirs:
        logger.error(f"No cache directories found under: {args.cache_root}")
        return


    scene_to_cache_map = build_scene_index(cache_leaf_dirs, args.desired_data)


    tasks_by_cache: Dict[str, List] = {}
    for scene_key, tasks in scene_tasks.items():
        found_cache = None
        candidates = [scene_key, f"{args.desired_data}/{scene_key}", scene_key.replace("_", "-")]
        for c in candidates:
            if c in scene_to_cache_map:
                found_cache = scene_to_cache_map[c]
                break
        if found_cache:
            tasks_by_cache.setdefault(found_cache, []).append((scene_key, tasks))
        else:
            logger.warning(f"Scene {scene_key} not found in any cache.")

    processed_count = 0
    fail_stat: Dict[str, int] = {}
    map_fail_pair = 0

   
    for cache_path, scene_groups in tasks_by_cache.items():
        logger.info(f"Loading Dataset from: {os.path.basename(cache_path)}")
        try:
            dataset = UnifiedDataset(
                desired_data=[args.desired_data],
                standardize_data=False,
                rebuild_cache=False,
                rebuild_maps=False,
                centric="scene",
                verbose=False,
                cache_location=cache_path,
                num_workers=args.num_workers,
                incl_vector_map=False,
                data_dirs={args.desired_data: ""}
            )
            map_api = MapAPI(dataset.cache_path)

            scenes_list = list(dataset.scenes())
            scene_lookup: Dict[str, Tuple[int, Any]] = {}
            for idx, s in enumerate(scenes_list):
                s_str = str(s)
                short = s_str.split("/")[-1] if "/" in s_str else s_str
                scene_lookup[short] = (idx, s)
                scene_lookup[s_str] = (idx, s)

        except Exception as e:
            logger.error(f"Failed to load dataset at {cache_path}: {e}")
            continue

        for scene_key, task_list in tqdm(scene_groups, desc=f"Scenes in {os.path.basename(cache_path)}"):
            scene_num = task_list[0]["scene_num"]

            csv_path = find_csv_for_scene(args.input_dir, scene_num)
            if not csv_path:
                continue
            try:
                df_csv = pd.read_csv(csv_path)
            except Exception:
                continue

      
            found = None
            for key in (scene_key, f"{args.desired_data}/{scene_key}", scene_key.replace("_", "-")):
                if key in scene_lookup:
                    found = scene_lookup[key]
                    break
            if found is None:
                continue
            found_idx, found_scene = found

            try:
                gc.collect()
                sp = SceneProcessor(args.desired_data, found_idx, found_scene, map_api, dataset.cache_path, args.timerange)
                _t, _a, _m, _d, _r = {}, {}, {}, {}, {}
                sp.process_scene(_t, _a, _m, _d, _r)
                process = generate_random_process(sp)

                # real_id -> internal idx
                id_map: Dict[Any, int] = {}
                for idx_v, meta in process.vehicle_metadata.items():
                    rid = str(meta.get("id", "")).strip()
                    if rid:
                        id_map[rid] = idx_v
                        if rid.isdigit():
                            id_map[int(rid)] = idx_v

                for task in task_list:
                    original_item = task["item"]
                    meta = task["meta"]
                    row_idx = meta["row_idx"]

                    if row_idx >= len(df_csv):
                        continue

                    csv_row = df_csv.iloc[row_idx]
                    pair = parse_vehicle_pair(csv_row.get("Vehicle Pair", ""))
                    if not pair:
                        map_fail_pair += 1
                        continue

                    ego_real_id, partner_real_id = pair[0], pair[1]
                    ego_idx = id_map.get(ego_real_id)
                    partner_idx = id_map.get(partner_real_id)

                    if ego_idx is None and str(ego_real_id).isdigit():
                        ego_idx = id_map.get(int(ego_real_id))
                    if partner_idx is None and str(partner_real_id).isdigit():
                        partner_idx = id_map.get(int(partner_real_id))

                    if ego_idx is None or partner_idx is None:
                        continue

                    sim_res = run_counterfactual_simulation(
                        process, ego_idx, partner_idx, meta["t_start"], meta["t_end"]
                    )
                    if not sim_res.get("valid", False):
                        r = sim_res.get("reason", "unknown")
                        fail_stat[r] = fail_stat.get(r, 0) + 1
                        continue

                    if "conversations" not in original_item or not isinstance(original_item["conversations"], list):
                        original_item["conversations"] = []

                    v_kmh = sim_res["initial_speed"] * 3.6
                    t_str = f"{meta['t_start']:.1f}"

                    q_text = (
                        f"<video>\nAt t={t_str}s, vehicle {ego_real_id} was traveling at {v_kmh:.1f} km/h. "
                        f"If it had maintained this constant velocity without reacting, what would likely happen regarding vehicle {partner_real_id}?"
                    )

                    if sim_res["collision"]:
                        a_text = (
                            f"Based on constant-velocity extrapolation for vehicle {ego_real_id} and the real trajectory of vehicle {partner_real_id}, "
                            f"a collision would likely occur about {sim_res['ttc_counterfactual']:.1f}s later. "
                            f"This indicates the observed maneuver was safety-critical."
                        )
                    else:
                        a_text = (
                            f"Under constant-velocity extrapolation for vehicle {ego_real_id}, no collision with vehicle {partner_real_id} "
                            f"would occur within the next 4s. The short-horizon risk appears manageable."
                        )

                    original_item["conversations"].append({"from": "human", "value": q_text})
                    original_item["conversations"].append({"from": "gpt", "value": a_text})
                    processed_count += 1

            except Exception as e:
                logger.error(f"Error processing {scene_key}: {e}")
                traceback.print_exc()

            try:
                del sp, process, df_csv, id_map
            except Exception:
                pass
            gc.collect()

  
        try:
            del dataset, map_api, scenes_list, scene_lookup
        except Exception:
            pass
        gc.collect()

    logger.info(f"Done. Processed {processed_count} QA pairs.")
    if fail_stat:
        logger.info(f"Invalid reasons (top 10): {sorted(fail_stat.items(), key=lambda x: -x[1])[:10]}")
    if map_fail_pair > 0:
        logger.info(f"Vehicle Pair parse failed count: {map_fail_pair}")

    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
