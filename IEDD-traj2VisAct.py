import os
import re
import argparse
import logging
import traceback
import copy
import shutil
import json
import math
from typing import Dict, Tuple, Any, Optional, List

import pandas as pd
import numpy as np

from trajdata import UnifiedDataset, MapAPI
from utils.IEDD_InteractionProcessor import SceneProcessor
from utils.IEDD_compute import generate_random_process, plot_2d_animation


logger = logging.getLogger("render_folder_csvs_to_mp4")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ==============================
# General utilities
# ==============================

def sanitize_tag(s: str, max_len: int = 140) -> str:
    s = str(s).strip().replace(os.sep, "__")
    s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len]
    return s or "tag"


def read_csv_with_encoding_fallback(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="ISO-8859-1")


def discover_csv_files(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"--input-dir does not exist or is not a directory：{input_dir}")
    out: List[str] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(".csv"):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def parse_time_interval(interval_str: str) -> Tuple[float, float]:
    if interval_str is None:
        raise ValueError("Time Interval is empty.")
    s = str(interval_str).strip()
    if "-" not in s:
        raise ValueError(f"Time Interval string format error：{interval_str}")
    a, b = s.split("-", 1)
    t0, t1 = float(a), float(b)
    if t1 < t0:
        t0, t1 = t1, t0
    return t0, t1


# ==============================
# Map helper (unchanged)
# ==============================

def try_get_vec_map_for_scene(scene_obj: Any, map_api: MapAPI) -> Optional[Any]:

    scene_str = str(scene_obj)

    ds_tag = None
    if "/" in scene_str:
        ds_tag = scene_str.split("/")[0]

    candidate_names: List[str] = []

    for attr in ["map_id", "map_name", "env_name"]:
        if hasattr(scene_obj, attr):
            val = getattr(scene_obj, attr)
            if isinstance(val, str) and val:
                candidate_names.append(val)

    meta = getattr(scene_obj, "metadata", None)
    if isinstance(meta, dict):
        for key in ["map_id", "map_name", "map", "hd_map"]:
            if key in meta and isinstance(meta[key], str) and meta[key]:
                candidate_names.append(meta[key])

    try:
        _, scene_token = scene_str.split("/")          # "scene_6577"
        scene_idx_str = scene_token.split("_")[-1]     # "6577"
        if scene_idx_str:
            candidate_names.append(scene_idx_str)
        if ds_tag:
            candidate_names.append(f"{ds_tag}_{scene_idx_str}")
    except Exception:
        pass

    seen = set()
    unique_candidates = []
    for name in candidate_names:
        if name not in seen:
            seen.add(name)
            unique_candidates.append(name)

    if not unique_candidates or ds_tag is None:
        return None

    for map_name in unique_candidates:
        key = f"{ds_tag}:{map_name}"
        try:
            return map_api.get_map(key)
        except Exception:
            continue
    return None


# ==============================
# Cache discovery / dataset init
# ==============================

def get_cache_locations(cache_root: str, explicit_subdirs: Optional[List[str]]) -> List[Tuple[str, str]]:

    if not os.path.isdir(cache_root):
        raise FileNotFoundError(f"--cache-root does not exist：{cache_root}")

    if explicit_subdirs:
        out = []
        for sub in explicit_subdirs:
            p = os.path.join(cache_root, sub)
            if os.path.isdir(p):
                out.append((sub, p))
            else:
                logger.warning("Specified cache subdirectory does not exist, skip：%s", p)
        if not out:
            raise RuntimeError("Explicitly specified --cache-subdirs, but none of them exist.")
        return out

    subdirs = [d for d in os.listdir(cache_root)
               if os.path.isdir(os.path.join(cache_root, d)) and not d.startswith(".")]
    subdirs.sort()


    if len(subdirs) > 0:
        return [(d, os.path.join(cache_root, d)) for d in subdirs]

    return [("SINGLE_CACHE", cache_root)]



def init_datasets_and_lookups(
    cache_locations: List[Tuple[str, str]],
    desired_data: str,
    num_workers: int
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Tuple[int, Any]]]]:
    datasets_info: Dict[str, Dict[str, Any]] = {}
    scene_lookups: Dict[str, Dict[str, Tuple[int, Any]]] = {}

    for label, cache_location in cache_locations:
        try:
            logger.info("Initialize cache (label=%s, cache_location=%s, desired_data=%s)...",
                        label, cache_location, desired_data)

            dataset = UnifiedDataset(
                desired_data=[desired_data],
                standardize_data=False,
                rebuild_cache=False,
                rebuild_maps=False,
                centric="scene",
                verbose=True,
                cache_location=cache_location,
                num_workers=num_workers,
                incl_vector_map=True,
                data_dirs={desired_data: ""},   
            )
            map_api = MapAPI(dataset.cache_path)
            cache_path = dataset.cache_path

            scenes = list(dataset.scenes())
            logger.info("  label=%s has %d scenes.", label, len(scenes))

            lookup: Dict[str, Tuple[int, Any]] = {}
            for idx, scene in enumerate(scenes):
                key_full = str(scene)
                lookup[key_full] = (idx, scene)
                base = os.path.basename(key_full)  # "scene_XXX"
                if base not in lookup:
                    lookup[base] = (idx, scene)

            datasets_info[label] = {
                "dataset": dataset,
                "map_api": map_api,
                "cache_path": cache_path,
                "scenes": scenes,
                "cache_location": cache_location,
            }
            scene_lookups[label] = lookup

        except Exception as e:
            logger.error("Initialize cache failed (label=%s, path=%s)：%s", label, cache_location, e)
            logger.debug(traceback.format_exc())
            continue

    if not datasets_info:
        raise RuntimeError("Failed to initialize any UnifiedDataset, please check cache_root / desired_data.")

    return datasets_info, scene_lookups



def resolve_scene(
    scene_name: str,
    scene_lookups: Dict[str, Dict[str, Tuple[int, Any]]],
) -> Tuple[str, int, Any]:
    scene_name = str(scene_name).strip()
    base = os.path.basename(scene_name)

    for label, lookup in scene_lookups.items():
        if scene_name in lookup:
            idx, scene_obj = lookup[scene_name]
            return label, idx, scene_obj

        if base in lookup:
            idx, scene_obj = lookup[base]
            return label, idx, scene_obj

        for key_str, (idx, scene_obj) in lookup.items():
            if key_str.endswith(scene_name):
                return label, idx, scene_obj

    raise KeyError(f"Don't find Scene='{scene_name}'")


# ==============================
# Action semantic extraction
# ==============================

def _wrap_to_pi(angle: float) -> float:
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    return a


def _moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(arr) < 3:
        return arr
    win = int(win)
    win = max(3, win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / win
    out = np.convolve(padded, kernel, mode="valid")
    return out


def _rolling_median(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(arr) < 3:
        return arr
    win = int(win)
    win = max(3, win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        out[i] = np.median(padded[i:i + win])
    return out


def _get_state_for_vid(frame_state: Any, vid: Any) -> Any:
    """
    frame_state might be:
      - dict[vid] -> state
      - dict with keys as int/str/others
    """
    if frame_state is None:
        return None
    if isinstance(frame_state, dict):
        if vid in frame_state:
            return frame_state[vid]
        # try int key
        try:
            iv = int(str(vid))
            if iv in frame_state:
                return frame_state[iv]
        except Exception:
            pass
        # try stringified keys
        for k, v in frame_state.items():
            if str(k) == str(vid):
                return v
    return None


def _extract_xy_heading_speed(state: Any) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Try to extract:
      x, y, heading(rad), speed(m/s), vx, vy
    """
    if state is None:
        return None, None, None, None, None, None

    if isinstance(state, dict):
        if "position" in state and isinstance(state["position"], (list, tuple, np.ndarray)) and len(state["position"]) >= 2:
            x, y = float(state["position"][0]), float(state["position"][1])
        elif "pos" in state and isinstance(state["pos"], (list, tuple, np.ndarray)) and len(state["pos"]) >= 2:
            x, y = float(state["pos"][0]), float(state["pos"][1])
        else:
            x = state.get("x", None)
            y = state.get("y", None)
            x = float(x) if x is not None else None
            y = float(y) if y is not None else None

        heading = None
        for hk in ["heading", "yaw", "psi", "h"]:
            if hk in state and state[hk] is not None:
                try:
                    heading = float(state[hk])
                    break
                except Exception:
                    pass

        speed = None
        vx = state.get("vx", None)
        vy = state.get("vy", None)
        if vx is not None and vy is not None:
            try:
                vx = float(vx)
                vy = float(vy)
                speed = math.sqrt(vx * vx + vy * vy)
            except Exception:
                vx, vy = None, None

        if speed is None:
            for sk in ["speed", "v", "vel"]:
                if sk in state and state[sk] is not None:
                    try:
                        speed = float(state[sk])
                        break
                    except Exception:
                        pass

        return x, y, heading, speed, (float(vx) if vx is not None else None), (float(vy) if vy is not None else None)

    for attrx, attry in [("x", "y"), ("pos_x", "pos_y")]:
        if hasattr(state, attrx) and hasattr(state, attry):
            try:
                x = float(getattr(state, attrx))
                y = float(getattr(state, attry))
                heading = float(getattr(state, "heading")) if hasattr(state, "heading") else None
                if heading is None and hasattr(state, "yaw"):
                    heading = float(getattr(state, "yaw"))
                speed = float(getattr(state, "speed")) if hasattr(state, "speed") else None
                vx = float(getattr(state, "vx")) if hasattr(state, "vx") else None
                vy = float(getattr(state, "vy")) if hasattr(state, "vy") else None
                if speed is None and vx is not None and vy is not None:
                    speed = math.sqrt(vx * vx + vy * vy)
                return x, y, heading, speed, vx, vy
            except Exception:
                pass

    if isinstance(state, (list, tuple, np.ndarray)) and len(state) >= 2:
        try:
            x = float(state[0])
            y = float(state[1])
        except Exception:
            return None, None, None, None, None, None

        heading = None
        speed = None
        if len(state) >= 4:
            a2 = float(state[2])
            a3 = float(state[3])
            if abs(a2) <= 3.5 and a3 >= 0:
                heading, speed = a2, a3
            elif abs(a3) <= 3.5 and a2 >= 0:
                heading, speed = a3, a2
        return x, y, heading, speed, None, None

    return None, None, None, None, None, None


def _collect_vehicle_series(process: Any, vid: Any, times: List[float]) -> Dict[str, np.ndarray]:
    xs, ys, hs, vs = [], [], [], []
    for t in times:
        frame_state = process.states.get(t, None)
        st = _get_state_for_vid(frame_state, vid)
        x, y, h, v, vx, vy = _extract_xy_heading_speed(st)
        xs.append(np.nan if x is None else x)
        ys.append(np.nan if y is None else y)
        hs.append(np.nan if h is None else h)
        if v is None and (vx is not None and vy is not None):
            v = math.sqrt(vx * vx + vy * vy)
        vs.append(np.nan if v is None else v)

    return {
        "x": np.array(xs, dtype=np.float64),
        "y": np.array(ys, dtype=np.float64),
        "heading": np.array(hs, dtype=np.float64),
        "speed": np.array(vs, dtype=np.float64),
    }


def _interp_nans(arr: np.ndarray) -> np.ndarray:
    if np.all(np.isnan(arr)):
        return arr
    x = np.arange(len(arr))
    mask = ~np.isnan(arr)
    return np.interp(x, x[mask], arr[mask]).astype(np.float64)


def _compute_features(times: np.ndarray, x: np.ndarray, y: np.ndarray, heading: np.ndarray, speed: np.ndarray) -> Dict[str, np.ndarray]:
    if len(times) >= 2:
        dts = np.diff(times)
        dt = float(np.median(dts[dts > 1e-6])) if np.any(dts > 1e-6) else 0.1
    else:
        dt = 0.1

    x = _interp_nans(x)
    y = _interp_nans(y)

    win_pos = 7
    xs = _moving_average(x, win_pos)
    ys = _moving_average(y, win_pos)

    if np.all(np.isnan(speed)) or (np.nanmean(speed) < 1e-6):
        vx = np.gradient(xs, dt)
        vy = np.gradient(ys, dt)
        v = np.sqrt(vx * vx + vy * vy)
    else:
        v = _interp_nans(speed)

    v = _moving_average(v, 5)

    if np.all(np.isnan(heading)):
        psi = np.arctan2(np.gradient(ys, dt), np.gradient(xs, dt))
    else:
        psi = _interp_nans(heading)

    psi_unwrap = np.unwrap(psi)
    psi_s = _moving_average(psi_unwrap, 5)

    yaw_rate = np.gradient(psi_s, dt)
    yaw_rate = _rolling_median(yaw_rate, 5)

    accel = np.gradient(v, dt)
    accel = _rolling_median(accel, 5)

    return {
        "dt": np.full_like(v, dt, dtype=np.float64),
        "x": xs,
        "y": ys,
        "v": v,
        "accel": accel,
        "psi": psi_s,
        "yaw_rate": yaw_rate,
    }


def _classify_frame_actions(
    v: np.ndarray,
    accel: np.ndarray,
    yaw_rate: np.ndarray,
    v_stop: float = 0.3,
    a_cruise: float = 0.25,
    a_thr: float = 0.6,
    yaw_straight: float = 0.05,
    yaw_turn: float = 0.15,
) -> Tuple[List[str], List[str]]:
    long_actions: List[str] = []
    lat_actions: List[str] = []

    for i in range(len(v)):
        if v[i] < v_stop:
            long_actions.append("STOPPED")
        else:
            if accel[i] > a_thr:
                long_actions.append("ACCELERATE")
            elif accel[i] < -a_thr:
                long_actions.append("DECELERATE")
            elif abs(accel[i]) <= a_cruise:
                long_actions.append("CRUISE")
            else:
                long_actions.append("CRUISE")

        if abs(yaw_rate[i]) < yaw_straight:
            lat_actions.append("STRAIGHT")
        else:
            if yaw_rate[i] > yaw_turn:
                lat_actions.append("TURN_LEFT")
            elif yaw_rate[i] < -yaw_turn:
                lat_actions.append("TURN_RIGHT")
            else:
                lat_actions.append("STRAIGHT")

    return long_actions, lat_actions


def _merge_segments(labels: List[str], times: np.ndarray, min_dur: float = 0.5) -> List[Dict[str, Any]]:
    if not labels:
        return []

    segs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segs.append({"t0": float(times[start]), "t1": float(times[i - 1]), "label": labels[start]})
            start = i

    changed = True
    while changed and len(segs) >= 2:
        changed = False
        new = []
        i = 0
        while i < len(segs):
            dur = segs[i]["t1"] - segs[i]["t0"]
            if dur < min_dur:
                if new:
                    new[-1]["t1"] = segs[i]["t1"]
                    changed = True
                    i += 1
                    continue
                elif i + 1 < len(segs):
                    segs[i + 1]["t0"] = segs[i]["t0"]
                    changed = True
                    i += 1
                    continue
            new.append(segs[i])
            i += 1
        segs = new

    return segs


def _detect_lane_change_from_straight_segments(
    segs_lat: List[Dict[str, Any]],
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    psi: np.ndarray,
    straight_only: bool = True,
    min_dur: float = 1.0,
    lat_thr: float = 1.2
) -> List[Dict[str, Any]]:
    if not segs_lat:
        return segs_lat

    t_to_idx = {float(t): i for i, t in enumerate(times.tolist())}

    for seg in segs_lat:
        if straight_only and seg["label"] != "STRAIGHT":
            continue
        dur = seg["t1"] - seg["t0"]
        if dur < min_dur:
            continue

        i0 = t_to_idx.get(float(seg["t0"]), None)
        i1 = t_to_idx.get(float(seg["t1"]), None)
        if i0 is None or i1 is None or i1 <= i0:
            continue

        h0 = psi[i0]
        dx = x[i1] - x[i0]
        dy = y[i1] - y[i0]
        lat = (-math.sin(h0) * dx + math.cos(h0) * dy)
        if abs(lat) >= lat_thr:
            seg["label"] = "LANE_CHANGE_LEFT" if lat > 0 else "LANE_CHANGE_RIGHT"

    return segs_lat


def _parse_involved_vehicle_ids(row: pd.Series) -> List[str]:
    """
    Parse involved vehicles from a csv row:
    - Vehicle Pair: "1014;1015"
    - Multi-Vehicle Group: may contain ids
    """
    vids: List[str] = []

    if "Vehicle Pair" in row and not pd.isna(row["Vehicle Pair"]):
        pair_str = str(row["Vehicle Pair"]).strip()
        parts = [p.strip() for p in pair_str.split(";") if p.strip()]
        vids.extend(parts)

    if "Multi-Vehicle Group" in row and not pd.isna(row["Multi-Vehicle Group"]):
        g = str(row["Multi-Vehicle Group"]).strip()
        if g:
            
            digits = re.findall(r"\d+", g)
            if digits:
                vids.extend(digits)
            else:
                
                tokens = re.findall(r"[A-Za-z0-9_]+", g)
                vids.extend(tokens)

    out = []
    seen = set()
    for v in vids:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# -------- NEW: resolve CSV raw id -> internal state key --------
def _resolve_vid_key(process: Any, raw_vid: Any, times: List[float]) -> Optional[Any]:

    raw_str = str(raw_vid)

    
    for t in times:
        frame_state = process.states.get(t, None)
        if not isinstance(frame_state, dict):
            continue
        if raw_vid in frame_state:
            return raw_vid
        for k in frame_state.keys():
            if str(k) == raw_str:
                return k


    meta = getattr(process, "vehicle_metadata", None)
    if isinstance(meta, dict):
        for k, m in meta.items():
            if not isinstance(m, dict):
                continue
            mid = m.get("id", None)
            if mid is not None and str(mid) == raw_str:
                return k

    return None


def extract_actions_for_row(
    process: Any,
    row: pd.Series,
    t_start: float,
    t_end: float,
    pad_s: float = 1.0,
    min_seg_dur: float = 0.5
) -> Dict[str, Any]:

    involved_raw = _parse_involved_vehicle_ids(row)

    all_times = sorted(process.states.keys())
    if not all_times:
        return {"time_window": {"t_start": t_start, "t_end": t_end, "pad_s": pad_s}, "vehicles": {}}

    t0 = t_start - pad_s
    t1 = t_end + pad_s
    times = [t for t in all_times if t0 <= t <= t1]
    if len(times) < 3:
        return {"time_window": {"t_start": t_start, "t_end": t_end, "pad_s": pad_s}, "vehicles": {}}

    times_np = np.array(times, dtype=np.float64)

    out: Dict[str, Any] = {
        "time_window": {
            "t_start": float(t_start),
            "t_end": float(t_end),
            "pad_s": float(pad_s),
            "t_extract0": float(times_np[0]),
            "t_extract1": float(times_np[-1]),
        },
        "vehicles": {},
        "debug": {
            "involved_raw": involved_raw,
            "raw_to_internal_key": {},  # str(raw)->str(key)
        }
    }

    # 1) resolve CSV raw ids -> internal keys
    resolved_keys: List[Any] = []
    raw_to_key: Dict[str, Any] = {}
    for rv in involved_raw:
        k = _resolve_vid_key(process, rv, times)
        if k is not None:
            resolved_keys.append(k)
            raw_to_key[str(rv)] = k
    out["debug"]["raw_to_internal_key"] = {k: str(v) for k, v in raw_to_key.items()}

    # 2) fallback
    if not resolved_keys:
        pool: List[Any] = []
        seen = set()
        for t in times:
            frame_state = process.states.get(t, None)
            if isinstance(frame_state, dict):
                for k in frame_state.keys():
                    if k not in seen:
                        seen.add(k)
                        pool.append(k)
            if len(pool) >= 20:
                break
        resolved_keys = pool

    # 3) inside mask
    eps = 1e-3
    inside_mask = (times_np >= (t_start - eps)) & (times_np <= (t_end + eps))
    if inside_mask.sum() < 2:
        return out

    for vid_key in resolved_keys:
        series = _collect_vehicle_series(process, vid_key, times)
        x = series["x"]
        y = series["y"]
        heading = series["heading"]
        speed = series["speed"]

       
        if np.all(np.isnan(x)) or np.all(np.isnan(y)):
            continue

        feats = _compute_features(times_np, x, y, heading, speed)
        v = feats["v"]
        accel = feats["accel"]
        yaw_rate = feats["yaw_rate"]
        psi = feats["psi"]
        xs = feats["x"]
        ys = feats["y"]

        long_labels, lat_labels = _classify_frame_actions(v, accel, yaw_rate)

        idxs = np.where(inside_mask)[0]
        if len(idxs) < 2:
            continue

        times_in = times_np[idxs]
        long_in = [long_labels[i] for i in idxs]
        lat_in = [lat_labels[i] for i in idxs]

        segs_long = _merge_segments(long_in, times_in, min_dur=min_seg_dur)
        segs_lat = _merge_segments(lat_in, times_in, min_dur=min_seg_dur)

        x_in = xs[idxs]
        y_in = ys[idxs]
        psi_in = psi[idxs]
        segs_lat = _detect_lane_change_from_straight_segments(
            segs_lat, times_in, x_in, y_in, psi_in,
            straight_only=True, min_dur=1.0, lat_thr=1.2
        )

        refined_lat = ["STRAIGHT"] * len(times_in)
        for seg in segs_lat:
            mask = (times_in >= seg["t0"]) & (times_in <= seg["t1"])
            for j in np.where(mask)[0]:
                refined_lat[j] = seg["label"]

        combined_refined = [f"{long_in[i]}+{refined_lat[i]}" for i in range(len(times_in))]
        segs_final = _merge_segments(combined_refined, times_in, min_dur=min_seg_dur)

        segments = []
        for seg in segs_final:
            tseg0, tseg1 = seg["t0"], seg["t1"]
            label = seg["label"]
            if "+" in label:
                llong, llat = label.split("+", 1)
            else:
                llong, llat = label, "STRAIGHT"
            segments.append({
                "t0": float(tseg0),
                "t1": float(tseg1),
                "long": llong,
                "lat": llat,
                "action": label
            })

        def _mode_str(arr: List[str]) -> str:
            if not arr:
                return ""
            from collections import Counter
            return Counter(arr).most_common(1)[0][0]

        summary = {
            "dominant_long": _mode_str(long_in),
            "dominant_lat": _mode_str(refined_lat),
            "num_frames": int(len(times_in)),
            "num_segments": int(len(segments)),
        }


        out_vid = vid_key
        meta = getattr(process, "vehicle_metadata", None)
        if isinstance(meta, dict) and (vid_key in meta) and isinstance(meta[vid_key], dict):
            out_vid = meta[vid_key].get("id", vid_key)

        out["vehicles"][str(out_vid)] = {
            "segments": segments,
            "summary": summary,
            "internal_key": str(vid_key),
        }

    return out


# ==============================
# Build subprocess (unchanged for video)
# ==============================

def build_subprocess_for_interval(
    process: Any,
    t_start: float,
    t_end: float,
    row_idx: int,
    csv_tag: str,
) -> Optional[Any]:
    all_times = sorted(process.states.keys())
    times_in_window = [t for t in all_times if t_start <= t <= t_end]

    if len(times_in_window) < 2:
        logger.warning("Time interval [%.2f, %.2f] has less than 2 frames, skip.", t_start, t_end)
        return None

    sub_proc = copy.deepcopy(process)
    sub_proc.states = {t: process.states[t] for t in times_in_window}
    sub_proc.duration = times_in_window[-1] - times_in_window[0]

    original_name = str(getattr(process, "name", "scene"))
    tag = sanitize_tag(csv_tag)
    sub_proc.name = f"{tag}__{original_name}_row{row_idx:04d}_t{t_start:.1f}-{t_end:.1f}"
    return sub_proc


# ==============================
# Main CSV processing
# ==============================

def process_one_csv(
    csv_path: str,
    input_dir: str,
    datasets_info: Dict[str, Dict[str, Any]],
    scene_lookups: Dict[str, Dict[str, Tuple[int, Any]]],
    desired_data: str,
    timerange: float,
    output_dir: str,
    limit: Optional[int],
    pad_actions: float,
    save_actions: bool,
) -> None:
    df = read_csv_with_encoding_fallback(csv_path)

    if "Scene" not in df.columns or "Time Interval" not in df.columns:
        logger.warning("Skip：%s（missing 'Scene' or 'Time Interval' columns）", csv_path)
        return

    if limit is not None:
        df = df.iloc[:limit].copy()

    rel_parent = os.path.relpath(os.path.dirname(csv_path), input_dir)
    out_subdir = os.path.join(output_dir, rel_parent)
    os.makedirs(out_subdir, exist_ok=True)

    csv_tag = sanitize_tag(os.path.join(rel_parent, os.path.basename(csv_path)))
    total_rows = len(df)

    logger.info("Start processing CSV：%s（%d rows）", csv_path, total_rows)

    for row_idx, row in df.iterrows():
        try:
            scene_col = row["Scene"]
            interval_col = row["Time Interval"]

            if pd.isna(scene_col) or pd.isna(interval_col):
                continue

            scene_name = str(scene_col).strip()
            t_start, t_end = parse_time_interval(str(interval_col))

            label, scene_idx, scene_obj = resolve_scene(scene_name, scene_lookups)
            ds_info = datasets_info[label]
            map_api = ds_info["map_api"]
            cache_path = ds_info["cache_path"]

            time_dic_output: Dict = {}
            a_min_dict: Dict = {}
            multi_a_min_dict: Dict = {}
            delta_TTCP_dict: Dict = {}
            results: Dict = {}

            scene_processor = SceneProcessor(
                desired_data,
                scene_idx,
                scene_obj,
                map_api,
                cache_path,
                timerange,
            )

            scene_processor.process_scene(
                time_dic_output,
                a_min_dict,
                multi_a_min_dict,
                delta_TTCP_dict,
                results,
            )

            random_process = generate_random_process(scene_processor)

            focus_vid = None
            if "Vehicle Pair" in df.columns and ("Vehicle Pair" in row) and (not pd.isna(row["Vehicle Pair"])):
                pair_str = str(row["Vehicle Pair"]).strip()
                first_str = pair_str.split(";")[0].strip()
                focus_vid = first_str
            setattr(random_process, "focus_vehicle_id", focus_vid)
            setattr(random_process, "focus_half_window", 40.0)

            vec_map = try_get_vec_map_for_scene(scene_obj, map_api)
            if vec_map is not None:
                setattr(random_process, "vec_map", vec_map)

            # -------- extract robust action semantics for this row --------
            actions_payload = None
            if save_actions:
                actions_payload = extract_actions_for_row(
                    process=random_process,
                    row=row,
                    t_start=t_start,
                    t_end=t_end,
                    pad_s=pad_actions,
                    min_seg_dur=0.5
                )

            # -------- render mp4 for the exact interval --------
            sub_proc = build_subprocess_for_interval(random_process, t_start, t_end, row_idx, csv_tag)
            if sub_proc is None:
                continue


            plot_2d_animation(sub_proc, metrics_data={})


            # -------- save actions --------
            if save_actions and actions_payload is not None:
                actions_dir = os.path.join("simulation_results", "actions")
                os.makedirs(actions_dir, exist_ok=True)

                base_name = os.path.basename(scene_name.rstrip("/\\").replace("/", "_"))
                clip_base = (
                    f"{sanitize_tag(base_name)}"
                    f"_row{int(row_idx):04d}"
                    f"_t{t_start:.1f}-{t_end:.1f}"
                )
                dst_json = os.path.join(actions_dir, clip_base + ".actions.json")

                meta = {
                    "scene": scene_name,
                    "time_interval": {"t_start": float(t_start), "t_end": float(t_end)},
                    "source_csv": csv_path,
                    "row_index": int(row_idx),
                    "vehicle_pair": str(row["Vehicle Pair"]) if ("Vehicle Pair" in df.columns and not pd.isna(row.get("Vehicle Pair", np.nan))) else "",
                    "interaction_type": str(row["Interaction Type"]) if ("Interaction Type" in df.columns and not pd.isna(row.get("Interaction Type", np.nan))) else "",
                }
                payload = {"meta": meta, "actions": actions_payload}
                with open(dst_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                logger.info("Action semantics output：%s", dst_json)

        except Exception as e:
            logger.error("Process %s row %d failed：%s", csv_path, row_idx, e)
            logger.debug(traceback.format_exc())
            continue

    logger.info("All done for CSV：%s", csv_path)


# ==============================
# Entrypoint
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="from IEDD trajdata cache generate 2D MP4，and extract action semantics."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="csv_dir",
        help="input csv_dir：recursively traverse all .csv files.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="root_dir",
        help="trajdata cache_location.",
    )
    parser.add_argument(
        "--desired-data",
        type=str,
        default="XXX",
        help="trajdata dataset label.",
    )
    parser.add_argument(
        "--cache-subdirs",
        type=str,
        nargs="*",
        default=None,
        help="If your cache-root has subdirectories, you can specify them explicitly.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="XXX",
        help="output mp4_dir：by CSV relative path.",
    )
    parser.add_argument(
        "--timerange",
        type=float,
        default=10.0,
        help="timerange(seconds) to SceneProcessor.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="number of workers to initialize UnifiedDataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If you want to process only the first N rows of each CSV, you can specify the number here.",
    )

    # -------- NEW args --------
    parser.add_argument(
        "--pad-actions",
        type=float,
        default=1.0,
        help="padding seconds before and after the action semantics extraction to improve robustness.",
    )
    parser.add_argument(
        "--no-save-actions",
        action="store_true",
        help="If set, do not output *.actions.json.",
    )

    args = parser.parse_args()

    csv_files = discover_csv_files(args.input_dir)
    if not csv_files:
        raise RuntimeError(f"Don't find any .csv in input directory：{args.input_dir}")
    logger.info("Find %d CSV.", len(csv_files))

    cache_locations = get_cache_locations(args.cache_root, args.cache_subdirs)

    datasets_info, scene_lookups = init_datasets_and_lookups(
        cache_locations=cache_locations,
        desired_data=args.desired_data,
        num_workers=args.num_workers,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for i, csv_path in enumerate(csv_files, start=1):
        logger.info("=== [%d/%d] %s ===", i, len(csv_files), csv_path)
        try:
            process_one_csv(
                csv_path=csv_path,
                input_dir=args.input_dir,
                datasets_info=datasets_info,
                scene_lookups=scene_lookups,
                desired_data=args.desired_data,
                timerange=args.timerange,
                output_dir=args.output_dir,
                limit=args.limit,
                pad_actions=args.pad_actions,
                save_actions=(not args.no_save_actions),
            )
        except Exception as e:
            logger.error("Process CSV failed (skip)：%s | %s", csv_path, e)
            logger.debug(traceback.format_exc())
            continue

    logger.info("All done.")


if __name__ == "__main__":
    main()
