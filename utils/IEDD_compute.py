import numpy as np
import os
import threading
import csv
from collections import defaultdict
import math
from typing import List, Tuple, Any, Dict, Set, Optional, Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import rcParams

from .data_models import VehicleState, RandomProcess, InteractionInfo
from .para import (
    DELTA_T,
    # below kept for compatibility; overridden constants are local inside functions
    V_DESIRE, V_LIM, ALPHA_Q, BETA_Q, A_LIM,
    GAMMA_Q, GAMMA_B, GAMMA_F, BETA_POTENTIAL, D0, KAPPA_V, V0,
    W_S, W_R, W_P,
    ALPHA_E, BETA_E, A_NORMAL
)

plot_lock = threading.Lock()

# =============== Global tuning params =================
POTENTIAL_NEIGHBOR_RADIUS = 10.0
EGO_SPEED_GAIN = 0.8
# ======================================================

# =================== Base utilities ===================

def normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_distance(state1: VehicleState, state2: VehicleState) -> float:
    return float(np.hypot(state1.x - state2.x, state1.y - state2.y))


def calculate_pet_from_times(t1_enter, t1_exit, t2_enter, t2_exit) -> float:
    """Compute PET by entry/exit times of conflict area (degenerates to time points if enter==exit)."""
    t1_exit = max(t1_enter, t1_exit)
    t2_exit = max(t2_enter, t2_exit)
    # overlap in conflict -> PET=0
    if t1_exit > t2_enter and t2_exit > t1_enter:
        return 0.0
    if t1_exit <= t2_enter:
        return float(t2_enter - t1_exit)
    return float(t1_enter - t2_exit)


def _is_valid_state(st: Optional[VehicleState]) -> bool:
    """Treat (0,0) as disappearance placeholder."""
    if st is None:
        return False
    return not (float(getattr(st, "x", 0.0)) == 0.0 and float(getattr(st, "y", 0.0)) == 0.0)


# ===== Interaction type -> Q weights (convex) =====

def get_q_weights_by_type(inter_type: Optional[str]):
    mapping = {
        "Merging":  (0.25, 0.35, 0.40),
        "Crossing": (0.20, 0.55, 0.25),
        "Head-on":  (0.15, 0.65, 0.20),
    }
    ws, wr, wp = mapping.get((inter_type or "").strip(), (0.25, 0.45, 0.30))
    s = ws + wr + wp
    if abs(s - 1.0) > 1e-9:
        ws, wr, wp = ws / s, wr / s, wp / s
    return ws, wr, wp


# =================== Q (state / risk / potential) ===================

def calculate_q_si(current_state: VehicleState, prev_state: VehicleState) -> float:
    V_LIM_OVR = 13.0   # m/s
    A_LIM_OVR = 3.0    # m/s^2
    BETA_Q_OVR = 0.6

    delta_v = float(current_state.v - prev_state.v)
    q_si = (abs(delta_v) / max(1e-6, V_LIM_OVR)) + BETA_Q_OVR * (abs(float(current_state.a)) / max(1e-6, A_LIM_OVR))
    return float(q_si)


def calculate_q_ri_optional(
    ttc_current: Optional[float],
    ttc_prev: Optional[float],
    pet_current: Optional[float],
    pet_prev: Optional[float],
) -> float:
    """
    Risk term = Δ(invTTC) + GAMMA_Q * Δ(invPET),
    but *skip missing terms* (TTC/PET may be absent).
    To avoid artificial spikes, if prev is missing but current exists, we use prev=current for that term.
    """
    GAMMA_Q_OVR = 0.8

    def inv_or_none(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        x = float(x)
        if x <= 0.0 or x == float("inf") or np.isnan(x):
            return 0.0
        return 1.0 / x

    inv_ttc_c = inv_or_none(ttc_current)
    inv_ttc_p = inv_or_none(ttc_prev)
    inv_pet_c = inv_or_none(pet_current)
    inv_pet_p = inv_or_none(pet_prev)

    # TTC delta
    if inv_ttc_c is None:
        d_ttc = 0.0
    else:
        if inv_ttc_p is None:
            inv_ttc_p = inv_ttc_c
        d_ttc = float(inv_ttc_c - inv_ttc_p)

    # PET delta
    if inv_pet_c is None:
        d_pet = 0.0
    else:
        if inv_pet_p is None:
            inv_pet_p = inv_pet_c
        d_pet = float(inv_pet_c - inv_pet_p)

    return float(d_ttc + GAMMA_Q_OVR * d_pet)


def calculate_q_ri(ttc_current: float, ttc_prev: float, pet_current: float, pet_prev: float) -> float:
    """Backward-compatible wrapper (treat non-positive/inf values as unavailable)."""
    def _clean(x: float) -> Optional[float]:
        try:
            x = float(x)
        except Exception:
            return None
        if x == float("inf") or x <= 0.0 or np.isnan(x):
            return None
        return x
    return calculate_q_ri_optional(_clean(ttc_current), _clean(ttc_prev), _clean(pet_current), _clean(pet_prev))


def _ego_speed_gain(v_i: float) -> float:
    V0_OVR = 5.0  # m/s
    return float(1.0 + EGO_SPEED_GAIN * (1.0 - np.exp(-float(v_i) / max(1e-6, V0_OVR))))


def calculate_potential_kernel(state_i: VehicleState, state_j: VehicleState) -> float:
    D0_OVR = 8.0
    V0_OVR = 5.0
    KAPPA_V_OVR = 0.6
    GAMMA_F_OVR = 1.2
    GAMMA_B_OVR = 0.3
    BETA_POTENTIAL_OVR = 2.0

    d_ij = calculate_distance(state_i, state_j)
    if d_ij < 1e-6:
        return float("inf")

    phi_ij = np.arctan2(state_j.y - state_i.y, state_j.x - state_i.x) - state_i.phi
    phi_ij = normalize_angle(float(phi_ij))

    omega_direction = GAMMA_B_OVR + (GAMMA_F_OVR - GAMMA_B_OVR) * ((1 + np.cos(phi_ij)) / 2.0) ** BETA_POTENTIAL_OVR

    v_j_vec = np.array([state_j.v * np.cos(state_j.phi), state_j.v * np.sin(state_j.phi)], dtype=float)
    v_i_vec = np.array([state_i.v * np.cos(state_i.phi), state_i.v * np.sin(state_i.phi)], dtype=float)
    relative_v_vec = v_j_vec - v_i_vec
    relative_pos_vec = np.array([state_j.x - state_i.x, state_j.y - state_i.y], dtype=float)

    v_ij_parallel = float(np.dot(relative_v_vec, relative_pos_vec) / d_ij) if d_ij > 0 else 0.0
    sigma_z = float(1.0 / (1.0 + np.exp(-v_ij_parallel / V0_OVR)))

    base = float(omega_direction * np.exp(-(d_ij / D0_OVR) ** 2) * (1.0 + KAPPA_V_OVR * sigma_z))
    return float(_ego_speed_gain(state_i.v) * base)


def _potential_kernel_field_vectorized(ego_state: VehicleState, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    D0_OVR = 8.0
    V0_OVR = 5.0
    KAPPA_V_OVR = 0.6
    GAMMA_F_OVR = 1.2
    GAMMA_B_OVR = 0.3
    BETA_POTENTIAL_OVR = 2.0

    dx = X - ego_state.x
    dy = Y - ego_state.y
    d = np.hypot(dx, dy)
    d = np.maximum(d, 1e-6)

    phi_ij = np.arctan2(dy, dx) - ego_state.phi
    phi_ij = (phi_ij + np.pi) % (2 * np.pi) - np.pi

    omega_direction = GAMMA_B_OVR + (GAMMA_F_OVR - GAMMA_B_OVR) * ((1 + np.cos(phi_ij)) / 2.0) ** BETA_POTENTIAL_OVR

    dot_term = (np.cos(ego_state.phi) * dx + np.sin(ego_state.phi) * dy)
    v_ij_parallel = (-ego_state.v * dot_term) / d
    sigma_z = 1.0 / (1.0 + np.exp(-v_ij_parallel / V0_OVR))

    base = omega_direction * np.exp(-(d / D0_OVR) ** 2) * (1.0 + KAPPA_V_OVR * sigma_z)
    return _ego_speed_gain(ego_state.v) * base


def calculate_q_pi(ego_state: VehicleState, partner_vehicles_states: List[VehicleState]) -> float:
    q_pi = 0.0
    for partner_state in partner_vehicles_states:
        q_pi += calculate_potential_kernel(ego_state, partner_state)
    return float(q_pi)


def calculate_q_i(q_si: float, q_ri: float, q_pi: float, inter_type: Optional[str] = None) -> float:
    def _squash_01(x: float) -> float:
        x = max(0.0, float(x))
        return float(1.0 - np.exp(-x))

    q_si_n = _squash_01(q_si)
    q_ri_n = _squash_01(max(0.0, q_ri))
    q_pi_n = _squash_01(q_pi)

    ws, wr, wp = get_q_weights_by_type(inter_type)
    return float(ws * q_si_n + wr * q_ri_n + wp * q_pi_n)


# =================== E (efficiency) ===================

def calculate_e_pi(initial_state: VehicleState, final_state: VehicleState, actual_distance: float) -> float:
    d_ref = float(np.hypot(initial_state.x - final_state.x, initial_state.y - final_state.y))
    if actual_distance < 1e-6:
        return 0.0
    return float(d_ref / actual_distance)


def calculate_e_ti(t_delay: float, t_free: float) -> float:
    ALPHA_E_OVR = 2.1
    if t_free == 0 or t_free == float("inf"):
        return 0.0
    return float(np.exp(-ALPHA_E_OVR * (t_delay / t_free)))


def calculate_e_si(sigma_a: float) -> float:
    BETA_E_OVR = 0.45
    A_NORMAL_OVR = 0.6
    return float(np.exp(-BETA_E_OVR * (sigma_a / max(1e-6, A_NORMAL_OVR))))


def calculate_e_i(e_pi: float, e_ti: float, e_si: float) -> float:
    return float(e_pi * e_ti * e_si)


def calculate_e_scene(e_i_values: List[float]) -> float:
    if not e_i_values:
        return 0.0
    return float(np.mean(e_i_values))


# =================== Interaction detection & classification ===================

def detect_and_classify_interactions(process: RandomProcess):
    """
    Detect & classify pairwise interactions and then build Multi groups.

    Fixes:
    - Pair interaction end_time is clamped by overlap_end to guarantee ego+partner exist.
    - Multi stores *per-pair* sub-interactions (possibly multiple) and CSV/metrics use pair windows.
    - Dynamic constraint: in each pair window, at least one vehicle has motion (by displacement threshold).
    """
    # ---------------- constants ----------------
    DIST_THRESHOLD = 5.0
    TIME_THRESHOLD = 3.0
    ANGLE_SAME_DIR = 30.0
    ANGLE_CROSSING = 160.0
    CAR_FOLLOW_MIN_POINTS = 3
    CAR_FOLLOW_DIR_TOL = 5.0
    WINDOW_SMOOTH = 5
    PRE_WINDOW_NONFOLLOW = 5.0
    POST_WINDOW_NONFOLLOW = 1.0
    MOVING_DIST_THRESHOLD = 0.1  # m per frame segment (~0.1s)

    def _norm_deg(a: float) -> float:
        a = a % 360.0
        return a if a >= 0 else a + 360.0

    def _angle_diff_abs(a: float, b: float) -> float:
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    def _median(vals: List[float]) -> float:
        if not vals:
            return 0.0
        vs = sorted(vals)
        n = len(vs)
        mid = n // 2
        return vs[mid] if n % 2 == 1 else 0.5 * (vs[mid - 1] + vs[mid])

    def _unwrap_and_smooth(deg_list: List[float], win: int = WINDOW_SMOOTH) -> List[float]:
        if not deg_list:
            return []
        unwrapped = [deg_list[0]]
        for k in range(1, len(deg_list)):
            a_prev = unwrapped[-1]
            a = deg_list[k]
            delta = a - a_prev
            while delta > 180.0:
                a -= 360.0
                delta = a - a_prev
            while delta < -180.0:
                a += 360.0
                delta = a - a_prev
            unwrapped.append(a)

        w = max(1, win | 1)
        r = w // 2
        smoothed = []
        n = len(unwrapped)
        prefix = [0.0]
        for v in unwrapped:
            prefix.append(prefix[-1] + v)
        for i in range(n):
            L = max(0, i - r)
            R = min(n - 1, i + r)
            s = prefix[R + 1] - prefix[L]
            smoothed.append(s / (R - L + 1))
        return [_norm_deg(v) for v in smoothed]

    def _compute_headings(xs: List[float], ys: List[float]) -> List[float]:
        n = len(xs)
        if n == 0:
            return []
        raw = [0.0] * n
        for i in range(n):
            if i < n - 1 and (xs[i + 1] != xs[i] or ys[i + 1] != ys[i]):
                dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
            elif i > 0 and (xs[i] != xs[i - 1] or ys[i] != ys[i - 1]):
                dx, dy = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
            else:
                raw[i] = raw[i - 1] if i > 0 else 0.0
                continue
            raw[i] = _norm_deg(math.degrees(math.atan2(dy, dx)))
        for i in range(1, n):
            if xs[i] == xs[i - 1] and ys[i] == ys[i - 1]:
                raw[i] = raw[i - 1]
        return _unwrap_and_smooth(raw, WINDOW_SMOOTH)

    # ---------------- collect trajectories ----------------
    times = sorted(process.states.keys())
    if not times:
        return

    traj_raw: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)  # vid -> [(t,x,y)]
    for t in times:
        frame = process.states.get(t, {})
        for vid, st in frame.items():
            if not _is_valid_state(st):
                continue
            traj_raw[vid].append((float(t), float(st.x), float(st.y)))

    if len(traj_raw) < 2:
        return

    trajectories: Dict[int, List[Tuple[float, float, float, float]]] = {}  # vid -> [(t,x,y,phi_deg)]
    exist_window: Dict[int, Tuple[float, float]] = {}
    for vid, seq in traj_raw.items():
        seq.sort(key=lambda a: a[0])
        ts = [r[0] for r in seq]
        xs = [r[1] for r in seq]
        ys = [r[2] for r in seq]
        phis = _compute_headings(xs, ys)
        trajectories[vid] = [(ts[i], xs[i], ys[i], phis[i]) for i in range(len(ts))]
        exist_window[vid] = (ts[0], ts[-1])

    vids = sorted(trajectories.keys())
    if len(vids) < 2:
        return

    # ---------------- dynamic motion check by displacement ----------------
    def _has_dynamic_motion(vid: int, t_start: float, t_end: float) -> bool:
        traj = trajectories.get(vid)
        if not traj or t_end <= t_start:
            return False
        for i in range(len(traj) - 1):
            t0, x0, y0, _ = traj[i]
            t1, x1, y1, _ = traj[i + 1]
            seg_start = min(t0, t1)
            seg_end = max(t0, t1)
            if seg_end <= t_start or seg_start >= t_end:
                continue
            dx = x1 - x0
            dy = y1 - y0
            if dx * dx + dy * dy >= MOVING_DIST_THRESHOLD * MOVING_DIST_THRESHOLD:
                return True
        return False

    def _pair_has_dynamic(a: int, b: int, t_start: float, t_end: float) -> bool:
        return _has_dynamic_motion(a, t_start, t_end) or _has_dynamic_motion(b, t_start, t_end)

    # ---------------- search cross points and classify ----------------
    class _InterBag:
        __slots__ = ("vehicle_pair", "interaction_type", "start_time", "end_time", "cross_points_info")
        def __init__(self, vehicle_pair, interaction_type, start_time, end_time, cross_points_info):
            self.vehicle_pair = vehicle_pair
            self.interaction_type = interaction_type
            self.start_time = float(start_time)
            self.end_time = float(end_time)
            self.cross_points_info = cross_points_info
        @property
        def type(self):
            return self.interaction_type

    all_inters: List[_InterBag] = []

    for idx_a in range(len(vids)):
        a = vids[idx_a]
        Ta = trajectories[a]
        if not Ta:
            continue
        for idx_b in range(idx_a + 1, len(vids)):
            b = vids[idx_b]
            Tb = trajectories[b]
            if not Tb:
                continue

            found = []  # (cp, t_a, t_b, phi_a, phi_b)
            j0 = 0
            for (t_a, x_a, y_a, phi_a) in Ta:
                while j0 < len(Tb) and Tb[j0][0] < t_a - TIME_THRESHOLD:
                    j0 += 1
                j = j0
                while j < len(Tb) and Tb[j][0] <= t_a + TIME_THRESHOLD:
                    t_b, x_b, y_b, phi_b = Tb[j]
                    dx = x_a - x_b
                    dy = y_a - y_b
                    if dx * dx + dy * dy <= DIST_THRESHOLD * DIST_THRESHOLD:
                        cp = ((x_a + x_b) * 0.5, (y_a + y_b) * 0.5)
                        found.append((cp, t_a, t_b, phi_a, phi_b))
                    j += 1

            if not found:
                continue

            dphis = [_angle_diff_abs(pa, pb) for (_, _, _, pa, pb) in found]
            midtimes = [0.5 * (ta + tb) for (_, ta, tb, _, _) in found]

            inter_type = None
            start_time = None
            end_time = None
            cross_info = None

            # pair co-existence window
            t_a_start, t_a_end = exist_window.get(a, (times[0], times[-1]))
            t_b_start, t_b_end = exist_window.get(b, (times[0], times[-1]))
            overlap_start = max(t_a_start, t_b_start)
            overlap_end = min(t_a_end, t_b_end)
            if overlap_end <= overlap_start:
                continue

            if len(found) >= CAR_FOLLOW_MIN_POINTS and _median(dphis) <= CAR_FOLLOW_DIR_TOL:
                start_time = float(max(overlap_start, min(midtimes)))
                end_time = float(min(overlap_end, max(midtimes)))
                inter_type = "Car-follow"
                cross_info = [(cp, ta, tb) for (cp, ta, tb, _, _) in found]
            else:
                best = min(found, key=lambda r: abs(r[1] - r[2]))
                cp, t_ai, t_bi, phi_ai, phi_bi = best
                delta = _angle_diff_abs(phi_ai, phi_bi)
                if 0.0 <= delta <= ANGLE_SAME_DIR:
                    inter_type = "Merging"
                elif ANGLE_SAME_DIR < delta <= ANGLE_CROSSING:
                    inter_type = "Crossing"
                else:
                    inter_type = "Head-on"

                t_first = min(t_ai, t_bi)

                desired_lower = t_first - PRE_WINDOW_NONFOLLOW
                start_time = float(max(0.0, max(desired_lower, overlap_start)))
                # IMPORTANT FIX: clamp end by overlap_end (both exist), and by process.duration
                end_time = float(min(t_first + POST_WINDOW_NONFOLLOW, overlap_end, getattr(process, "duration", t_first + POST_WINDOW_NONFOLLOW)))

                if end_time <= start_time:
                    continue

                cross_info = [(cp, t_ai, t_bi)]

            if inter_type is None:
                continue
            if end_time <= start_time:
                continue

            if not _pair_has_dynamic(a, b, start_time, end_time):
                continue

            pair = (a, b) if a < b else (b, a)
            all_inters.append(_InterBag(pair, inter_type, start_time, end_time, cross_info))

    if not all_inters:
        return

    # ---------------- Multi grouping ----------------
    types_by_vehicle: Dict[int, Set[str]] = defaultdict(set)
    partners_by_vehicle: Dict[int, Set[int]] = defaultdict(set)
    inter_by_pair: Dict[Tuple[int, int], List[_InterBag]] = defaultdict(list)

    for it in all_inters:
        u, v = it.vehicle_pair
        types_by_vehicle[u].add(it.type)
        types_by_vehicle[v].add(it.type)
        partners_by_vehicle[u].add(v)
        partners_by_vehicle[v].add(u)
        inter_by_pair[(min(u, v), max(u, v))].append(it)

    # sort each pair's interactions by time for stable iteration
    for k in list(inter_by_pair.keys()):
        inter_by_pair[k].sort(key=lambda it: (it.start_time, it.end_time, it.type))

    anchors = {vid for vid, tset in types_by_vehicle.items() if len(tset) >= 2}

    if anchors:
        groups: List[Set[int]] = []
        for a in anchors:
            g = set([a]) | set(partners_by_vehicle.get(a, set()))
            merged = False
            for G in groups:
                if G & g:
                    G |= g
                    merged = True
                    break
            if not merged:
                groups.append(g)

        # merge indirect overlaps
        changed = True
        while changed:
            changed = False
            new_groups = []
            while groups:
                g = groups.pop()
                for i in range(len(groups) - 1, -1, -1):
                    if g & groups[i]:
                        g |= groups[i]
                        groups.pop(i)
                        changed = True
                new_groups.append(g)
            groups = new_groups

        multi_members: Set[int] = set().union(*groups) if groups else set()

        for G in groups:
            related_inters: List[_InterBag] = []
            for u in G:
                for v in G:
                    if u >= v:
                        continue
                    key = (u, v)
                    if key in inter_by_pair:
                        related_inters.extend(inter_by_pair[key])
            if not related_inters:
                continue

            g_start = min(it.start_time for it in related_inters)
            g_end = max(it.end_time for it in related_inters)

            # per member: clamp group interval by member existence to avoid "ego not exist"
            for vid in G:
                v_start, v_end = exist_window.get(vid, (g_start, g_end))
                member_start = float(max(g_start, v_start))
                member_end = float(min(g_end, v_end))
                if member_end <= member_start:
                    continue

                partners_map: Dict[int, List[_InterBag]] = {}
                for other in (G - {vid}):
                    key = (min(vid, other), max(vid, other))
                    if key in inter_by_pair:
                        partners_map[other] = list(inter_by_pair[key])  # all occurrences

                if partners_map:
                    process.interaction_info_by_vehicle[vid].append(
                        InteractionInfo("Multi", member_start, member_end, partners_map)
                    )

            process.interaction_participants.update(G)

        skip_vids = multi_members
    else:
        skip_vids = set()

    # ---------------- write non-Multi pair interactions ----------------
    for it in all_inters:
        u, v = it.vehicle_pair
        if u in skip_vids or v in skip_vids:
            continue
        if it.type == "Car-follow":
            process.interaction_info_by_vehicle[u].append(
                InteractionInfo("Car-follow", it.start_time, it.end_time, {v: [it]})
            )
            process.interaction_info_by_vehicle[v].append(
                InteractionInfo("Car-follow", it.start_time, it.end_time, {u: [it]})
            )
        else:
            process.interaction_info_by_vehicle[u].append(
                InteractionInfo(it.type, it.start_time, it.end_time, {v: [it]})
            )
            process.interaction_info_by_vehicle[v].append(
                InteractionInfo(it.type, it.start_time, it.end_time, {u: [it]})
            )
        process.interaction_participants.update([u, v])


# =================== E over full lifecycle ===================

def _calculate_full_trajectory_e_metrics(process: RandomProcess) -> Dict[int, Dict[str, float]]:
    V_DESIRE_OVR = 11.0
    e_metrics_by_vehicle: Dict[int, Dict[str, float]] = {}
    all_time_points = sorted(process.states.keys())

    for vid in process.interaction_participants:
        veh_times = []
        for t in all_time_points:
            st = process.states.get(t, {}).get(vid)
            if _is_valid_state(st):
                veh_times.append(t)
        if len(veh_times) < 2:
            continue

        initial_state = process.states[veh_times[0]][vid]
        final_state = process.states[veh_times[-1]][vid]

        actual_distance = 0.0
        for i in range(1, len(veh_times)):
            a = process.states[veh_times[i - 1]][vid]
            b = process.states[veh_times[i]][vid]
            actual_distance += calculate_distance(a, b)

        e_pi = calculate_e_pi(initial_state, final_state, actual_distance)

        t_delay = 0.0
        for t in veh_times:
            v = float(process.states[t][vid].v)
            if V_DESIRE_OVR - v > 0:
                t_delay += float(DELTA_T)
        t_free = actual_distance / V_DESIRE_OVR if V_DESIRE_OVR > 1e-6 else float("inf")

        e_ti = calculate_e_ti(t_delay, t_free)
        sigma_a = float(np.std([float(process.states[t][vid].a) for t in veh_times])) if veh_times else 0.0
        e_si = calculate_e_si(sigma_a)
        e_i = calculate_e_i(e_pi, e_ti, e_si)

        e_metrics_by_vehicle[vid] = {"e_i": e_i, "e_pi": e_pi, "e_ti": e_ti, "e_si": e_si}

    return e_metrics_by_vehicle


# =================== Q + E unified metrics ===================

def calculate_all_metrics(process: RandomProcess):
    """
    Q: computed only inside non Car-follow interaction windows, but strictly on *pair-level* windows
       (including each sub-interaction inside Multi).
    E: computed for interacting vehicles over full lifecycle.

    TTC/PET:
    - TTC requires ego+partner states at that frame and a meaningful approach speed.
    - PET requires cross_points_info; if missing, PET is treated as unavailable and skipped in q_ri.
    """
    q_i_over_time, q_si_over_time, q_ri_over_time, q_pi_over_time = [defaultdict(dict) for _ in range(4)]
    all_time = sorted(process.states.keys())

    # ---- raw TTC/PET rows (append) ----
    ttc_pet_rows = []
    EPS_APPROACH = 0.30
    TTC_KEEP_LT = 5.0
    PET_KEEP_LT = 5.0

    def _robust_ttc(distance: float, approach_speed: float) -> Optional[float]:
        if approach_speed <= EPS_APPROACH:
            return None
        if distance < 1e-9:
            return 0.0
        return float(max(0.0, distance / approach_speed))

    def _vx_vy_from(state: VehicleState) -> Tuple[float, float]:
        if hasattr(state, "vx") and hasattr(state, "vy"):
            return float(state.vx), float(state.vy)
        phi = float(getattr(state, "phi", 0.0))
        return float(state.v * np.cos(phi)), float(state.v * np.sin(phi))

    def _store_max(d: Dict[int, Dict[float, float]], vid: int, t: float, val: float):
        prev = d[vid].get(t)
        if prev is None or float(val) > float(prev):
            d[vid][t] = float(val)

    # ---- iterate all ego interactions, flattened to pair-level ----
    # we de-duplicate by (ego, partner, start, end, type)
    seen_inter_keys: Set[Tuple[int, int, float, float, str]] = set()

    for ego_id in process.interaction_participants:
        info_list = process.interaction_info_by_vehicle.get(ego_id, [])
        for info in info_list:
            if info.type == "Car-follow":
                continue

            # flatten: for non-Multi, partners map has one key; for Multi, multiple keys;
            # values are lists of _InterBag
            partners_map = getattr(info, "partners", {}) or {}
            for partner_id, inter_list in partners_map.items():
                if not inter_list:
                    continue
                for inter in inter_list:
                    inter_type = getattr(inter, "type", info.type)
                    start_t = float(getattr(inter, "start_time", info.start_time))
                    end_t = float(getattr(inter, "end_time", info.end_time))

                    # ensure ego exists + partner exists in this window (basic clamp)
                    # (pair windows from detection already satisfy; this is extra safety)
                    if end_t <= start_t:
                        continue

                    key = (int(ego_id), int(partner_id), start_t, end_t, str(inter_type))
                    if key in seen_inter_keys:
                        continue
                    seen_inter_keys.add(key)

                    # time span: ego & partner must exist; also keep only motion frames (at least one moving)
                    t_raw = [t for t in all_time if start_t <= t <= end_t]
                    t_span: List[float] = []
                    prev_valid_t: Optional[float] = None
                    prev_ego: Optional[VehicleState] = None
                    prev_par: Optional[VehicleState] = None
                    MOVING_DIST_THRESHOLD = 0.1  # m between frames

                    for t in t_raw:
                        ego_st = process.states.get(t, {}).get(ego_id)
                        par_st = process.states.get(t, {}).get(partner_id)
                        if not _is_valid_state(ego_st) or not _is_valid_state(par_st):
                            continue

                        moving = (float(ego_st.v) > 1e-3) or (float(par_st.v) > 1e-3)

                        # displacement-based motion (preferred)
                        if prev_valid_t is not None and prev_ego is not None and prev_par is not None:
                            de = float(np.hypot(ego_st.x - prev_ego.x, ego_st.y - prev_ego.y))
                            dp = float(np.hypot(par_st.x - prev_par.x, par_st.y - prev_par.y))
                            if de >= MOVING_DIST_THRESHOLD or dp >= MOVING_DIST_THRESHOLD:
                                moving = True

                        if moving:
                            t_span.append(t)

                        prev_valid_t = t
                        prev_ego = ego_st
                        prev_par = par_st

                    if len(t_span) < 2:
                        continue

                    prev_ttc: Optional[float] = None
                    prev_pet: Optional[float] = None

                    for idx, t in enumerate(t_span):
                        ego_now = process.states[t][ego_id]
                        par_now = process.states[t][partner_id]

                        # state term (uses prev frame in this filtered span)
                        if idx > 0:
                            prev_state = process.states[t_span[idx - 1]][ego_id]
                        else:
                            prev_state = ego_now
                        q_si_val = calculate_q_si(ego_now, prev_state)

                        # TTC current (optional)
                        dx, dy = float(par_now.x - ego_now.x), float(par_now.y - ego_now.y)
                        dist = float(np.hypot(dx, dy))
                        ttc_current: Optional[float] = None
                        if dist >= 1e-9:
                            evx, evy = _vx_vy_from(ego_now)
                            pvx, pvy = _vx_vy_from(par_now)
                            rvx, rvy = pvx - evx, pvy - evy
                            approach = - (rvx * dx + rvy * dy) / max(dist, 1e-9)
                            ttc_current = _robust_ttc(dist, approach)

                        # PET current (optional, depends on cross_points_info)
                        pet_current: Optional[float] = None
                        cross_info = getattr(inter, "cross_points_info", None)
                        if cross_info:
                            best = min(cross_info, key=lambda rec: abs(rec[1] - rec[2]))
                            _, t_i_cross, t_j_cross = best
                            if inter.vehicle_pair[0] == ego_id:
                                t_ego, t_partner = t_i_cross, t_j_cross
                            else:
                                t_ego, t_partner = t_j_cross, t_i_cross
                            pet_current = calculate_pet_from_times(t_ego, t_ego, t_partner, t_partner)

                        # record TTC/PET raw (only keep <5s)
                        ttc_to_write = ttc_current if (ttc_current is not None and 0.0 < ttc_current < TTC_KEEP_LT) else ""
                        pet_to_write = pet_current if (pet_current is not None and 0.0 < pet_current < PET_KEEP_LT) else ""
                        if ttc_to_write != "" or pet_to_write != "":
                            ttc_pet_rows.append([process.name, t, ego_id, inter_type, ttc_to_write, pet_to_write])

                        # risk term (skip missing)
                        q_ri_val = calculate_q_ri_optional(ttc_current, prev_ttc, pet_current, prev_pet)

                        # potential term: all neighbors within R (exclude invalid)
                        neighbors_states: List[VehicleState] = []
                        R = 13.0
                        for other_vid, other_st in process.states.get(t, {}).items():
                            if other_vid == ego_id:
                                continue
                            if not _is_valid_state(other_st):
                                continue
                            if calculate_distance(ego_now, other_st) <= R:
                                neighbors_states.append(other_st)
                        q_pi_val = calculate_q_pi(ego_now, neighbors_states)

                        q_i_val = calculate_q_i(q_si_val, q_ri_val, q_pi_val, inter_type=inter_type)

                        # store as max across potentially multiple interactions per time
                        _store_max(q_si_over_time, ego_id, t, float(q_si_val))
                        _store_max(q_ri_over_time, ego_id, t, float(q_ri_val))
                        _store_max(q_pi_over_time, ego_id, t, float(q_pi_val))
                        _store_max(q_i_over_time, ego_id, t, float(q_i_val))

                        prev_ttc = ttc_current
                        prev_pet = pet_current

    # ---- E metrics ----
    e_metrics = _calculate_full_trajectory_e_metrics(process)
    e_scene = calculate_e_scene([m["e_i"] for m in e_metrics.values()])

    # ---- write raw TTC/PET ----
    try:
        out_dir = os.path.join("simulation_results", process.name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "ttc_pet_raw.csv")
        file_exists = os.path.isfile(out_path)
        with open(out_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Scene", "Time", "Ego", "InteractionType", "MinTTC", "MinPET"])
            writer.writerows(ttc_pet_rows)
    except Exception as e:
        print(f"[WARN] write ttc_pet_raw.csv failed: {e}")

    return {
        "q_i_over_time": q_i_over_time,
        "q_si_over_time": q_si_over_time,
        "q_ri_over_time": q_ri_over_time,
        "q_pi_over_time": q_pi_over_time,
        "e_metrics": e_metrics,
        "e_scene": e_scene,
    }


# =================== Save results to CSV ===================

def save_results_to_csv(process: RandomProcess, metrics_data: dict):
    """
    Save interaction results to CSV.

    IMPORTANT FIX:
    - For Multi, each row uses the *pair-level* sub-interaction interval (sub_inter.start/end),
      not the group's [info.start, info.end]. This guarantees the "ego" exists in the interval.
    """
    output_dir = os.path.join("simulation_results", process.name)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "interaction.csv")

    q_i_data = metrics_data.get("q_i_over_time", {})
    e_metrics = metrics_data.get("e_metrics", {})

    def vehicle_id_str(vid):
        m = process.vehicle_metadata.get(vid, {})
        return str(m.get("id", vid))

    def format_q_series_for_pair(v1, v2, start_t, end_t):
        parts = []
        for vid in (v1, v2):
            veh_id = vehicle_id_str(vid)
            t2q = q_i_data.get(vid, {})
            ts = sorted(t for t in t2q.keys() if float(start_t) <= float(t) <= float(end_t))
            for t in ts:
                parts.append(f"{veh_id}:{t:.3f}:{float(t2q[t]):.4f}")
        return ",".join(parts)

    def format_e_values_for_vids(vids):
        items = []
        for vid in vids:
            veh_id = vehicle_id_str(vid)
            e_val = float(e_metrics.get(vid, {}).get("e_i", 0.0))
            items.append((veh_id, e_val))
        items.sort(key=lambda x: x[0])
        return ",".join([f"{veh}:{e:.4f}" for veh, e in items])

    def _pair_window_has_presence_and_motion(v1: int, v2: int, start_t: float, end_t: float) -> bool:
        """Extra safety: within [start,end], there are >=2 frames where both exist and at least one is dynamic."""
        times = sorted(process.states.keys())
        kept = 0
        prev_s1 = None
        prev_s2 = None
        MOVING_DIST_THRESHOLD = 0.1

        for t in times:
            if t < start_t or t > end_t:
                continue
            s1 = process.states.get(t, {}).get(v1)
            s2 = process.states.get(t, {}).get(v2)
            if not _is_valid_state(s1) or not _is_valid_state(s2):
                continue

            moving = (float(s1.v) > 1e-3) or (float(s2.v) > 1e-3)
            if prev_s1 is not None and prev_s2 is not None:
                d1 = float(np.hypot(s1.x - prev_s1.x, s1.y - prev_s1.y))
                d2 = float(np.hypot(s2.x - prev_s2.x, s2.y - prev_s2.y))
                if d1 >= MOVING_DIST_THRESHOLD or d2 >= MOVING_DIST_THRESHOLD:
                    moving = True

            if moving:
                kept += 1
                if kept >= 2:
                    return True

            prev_s1, prev_s2 = s1, s2

        return False

    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Scene", "Vehicle Pair", "Time Interval", "Interaction Type",
                "Q_i_Over_Time", "Multi-Vehicle Group", "E_Veh"
            ])

        processed_rows: Set[Tuple[Tuple[int, int], float, float, str]] = set()

        for ego_vid, info_list in process.interaction_info_by_vehicle.items():
            for info in info_list:
                if info.type == "Multi":
                    group_vids = sorted(set([ego_vid] + list(getattr(info, "partners", {}).keys())))
                    group_ids_str = ";".join([vehicle_id_str(v) for v in group_vids])
                    e_group_str = format_e_values_for_vids(group_vids)

                    for partner_vid, inter_list in (getattr(info, "partners", {}) or {}).items():
                        for sub_inter in (inter_list or []):
                            pair_indices = tuple(sorted((ego_vid, partner_vid)))
                            start_t = float(getattr(sub_inter, "start_time", info.start_time))
                            end_t = float(getattr(sub_inter, "end_time", info.end_time))
                            itype = str(getattr(sub_inter, "type", "Interaction"))

                            row_key = (pair_indices, start_t, end_t, itype)
                            if row_key in processed_rows:
                                continue
                            processed_rows.add(row_key)

                            if not _pair_window_has_presence_and_motion(pair_indices[0], pair_indices[1], start_t, end_t):
                                continue

                            v1_idx, v2_idx = pair_indices
                            pair_str = f"{vehicle_id_str(v1_idx)};{vehicle_id_str(v2_idx)}"
                            time_interval = f"{start_t:.1f}-{end_t:.1f}"
                            q_series = format_q_series_for_pair(v1_idx, v2_idx, start_t, end_t)

                            writer.writerow([
                                process.name,
                                pair_str,
                                time_interval,
                                itype,
                                q_series,
                                group_ids_str,
                                e_group_str
                            ])
                else:
                    partners = list((getattr(info, "partners", {}) or {}).keys())
                    if not partners:
                        continue
                    partner_vid = partners[0]
                    inter_list = (getattr(info, "partners", {}) or {}).get(partner_vid, [])
                    if not inter_list:
                        continue
                    # non-multi should have exactly one sub_inter (we keep first for safety)
                    sub_inter = inter_list[0]

                    pair_indices = tuple(sorted((ego_vid, partner_vid)))
                    start_t = float(getattr(sub_inter, "start_time", info.start_time))
                    end_t = float(getattr(sub_inter, "end_time", info.end_time))
                    itype = str(getattr(sub_inter, "type", info.type))

                    row_key = (pair_indices, start_t, end_t, itype)
                    if row_key in processed_rows:
                        continue
                    processed_rows.add(row_key)

                    if not _pair_window_has_presence_and_motion(pair_indices[0], pair_indices[1], start_t, end_t):
                        continue

                    v1_idx, v2_idx = pair_indices
                    pair_str = f"{vehicle_id_str(v1_idx)};{vehicle_id_str(v2_idx)}"
                    time_interval = f"{start_t:.1f}-{end_t:.1f}"

                    q_series = format_q_series_for_pair(v1_idx, v2_idx, start_t, end_t)
                    e_veh_str = format_e_values_for_vids([v1_idx, v2_idx])

                    writer.writerow([
                        process.name,
                        pair_str,
                        time_interval,
                        itype,
                        q_series,
                        "",
                        e_veh_str
                    ])


# =================== Plot helpers (unchanged except output path naming) ===================

def get_rotated_rect_corners(x, y, width, length, angle):
    dx, dy = length / 2, width / 2
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (R @ corners.T).T + [x, y]


def plot_2d_animation(process: RandomProcess, metrics_data: dict):
    with plot_lock:
        times = sorted(process.states.keys())
        if not times:
            print("No available state time series.")
            return

        active_intervals = defaultdict(list)
        for vid, info_list in process.interaction_info_by_vehicle.items():
            for info in info_list:
                # For Multi, this interval is only for "group presence", not for per-pair rows.
                active_intervals[vid].append((float(info.start_time), float(info.end_time)))

        def is_active(vid: int, t: float) -> bool:
            for s, e in active_intervals.get(vid, []):
                if s <= t <= e:
                    return True
            return False

        def safe_meta(vid: int):
            m = process.vehicle_metadata.get(vid, {})
            width = float(m.get("width", 2.0))
            length = float(m.get("length", 4.0))
            size_scale = getattr(process, "vehicle_size_scale", 1.8)
            width *= size_scale
            length *= size_scale
            return {"width": width, "length": length, "id": m.get("id", vid)}

        def safe_corners(x, y, w, l, phi):
            try:
                return get_rotated_rect_corners(x, y, w, l, phi)
            except Exception:
                hw, hl = w * 0.5, l * 0.5
                return [(x - hl, y - hw), (x + hl, y - hw), (x + hl, y + hw), (x - hl, y + hw)]

        minx, miny = float("inf"), float("inf")
        maxx, maxy = float("-inf"), float("-inf")
        any_valid = False
        for t in times:
            frame_states = process.states.get(t, {})
            for vid, st in frame_states.items():
                if not _is_valid_state(st):
                    continue
                meta = safe_meta(vid)
                corners = safe_corners(st.x, st.y, meta["width"], meta["length"], st.phi)
                any_valid = True
                for (cx, cy) in corners:
                    minx, maxx = min(minx, cx), max(maxx, cx)
                    miny, maxy = min(miny, cy), max(maxy, cy)
        if not any_valid:
            minx, maxx, miny, maxy = -50.0, 50.0, -50.0, 50.0

        extra_margin = 10.0
        minx -= extra_margin
        maxx += extra_margin
        miny -= extra_margin
        maxy += extra_margin

        global_x_center = 0.5 * (minx + maxx)
        global_y_center = 0.5 * (miny + maxy)
        global_half_w = 0.5 * (maxx - minx)
        global_half_h = 0.5 * (maxy - miny)

        focus_raw = getattr(process, "focus_vehicle_id", None)
        focus_half_window = float(getattr(process, "focus_half_window", 120.0))
        focus_radius = float(getattr(process, "focus_radius", 80.0))
        focus_half_window = max(focus_half_window, focus_radius)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_xlim(global_x_center - global_half_w, global_x_center + global_half_w)
        ax.set_ylim(global_y_center - global_half_h, global_y_center + global_half_h)

        time_text = ax.text(
            0.02, 0.97, "", transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

        vehicle_patches = {}
        vehicle_text = {}

        all_vids = set()
        for t in times:
            all_vids.update(process.states.get(t, {}).keys())

        focus_vid = None
        if focus_raw is not None:
            if focus_raw in all_vids:
                focus_vid = focus_raw
            else:
                for vid in all_vids:
                    m = process.vehicle_metadata.get(vid, {})
                    if str(m.get("id", vid)) == str(focus_raw):
                        focus_vid = vid
                        break
            if focus_vid is None:
                print(f"[plot_2d_animation] Warning: No internal vid found for focus_vehicle_id={focus_raw} in states.")

        for vid in all_vids:
            patch = Polygon([[0, 0]], closed=True, visible=False, facecolor="none", edgecolor="black", linewidth=0.8, zorder=10)
            ax.add_patch(patch)
            vehicle_patches[vid] = patch
            vehicle_text[vid] = ax.text(0, 0, "", fontsize=8, color="red", fontweight="bold", visible=False, zorder=20)

        def update(frame_idx):
            t = times[frame_idx]
            time_text.set_text(f"Time: {t:.1f}s")

            for txt in vehicle_text.values():
                txt.set_visible(False)

            frame_states = process.states.get(t, {})
            st_focus = None
            if focus_vid is not None:
                st_focus = frame_states.get(focus_vid)
                if _is_valid_state(st_focus):
                    ax.set_xlim(st_focus.x - focus_half_window, st_focus.x + focus_half_window)
                    ax.set_ylim(st_focus.y - focus_half_window, st_focus.y + focus_half_window)
                else:
                    st_focus = None
                    ax.set_xlim(global_x_center - global_half_w, global_x_center + global_half_w)
                    ax.set_ylim(global_y_center - global_half_h, global_y_center + global_half_h)
            else:
                ax.set_xlim(global_x_center - global_half_w, global_x_center + global_half_w)
                ax.set_ylim(global_y_center - global_half_h, global_y_center + global_half_h)

            for vid in all_vids:
                st = frame_states.get(vid)
                if not _is_valid_state(st):
                    vehicle_patches[vid].set_visible(False)
                    continue

                if focus_vid is not None and st_focus is not None:
                    dx = st.x - st_focus.x
                    dy = st.y - st_focus.y
                    if dx * dx + dy * dy > focus_radius * focus_radius:
                        vehicle_patches[vid].set_visible(False)
                        continue

                meta = safe_meta(vid)
                corners = safe_corners(st.x, st.y, meta["width"], meta["length"], st.phi)
                patch = vehicle_patches[vid]
                patch.set_xy(corners)

                if focus_vid is not None:
                    patch.set_edgecolor("red" if vid == focus_vid else "black")
                else:
                    patch.set_edgecolor("red" if is_active(vid, t) else "black")

                patch.set_facecolor("none")
                patch.set_linewidth(0.8)
                patch.set_visible(True)

                txt = vehicle_text[vid]
                txt.set_position((st.x, st.y + meta["width"] / 2 + 2.5))
                txt.set_text(f"{meta['id']}")
                if focus_vid is not None:
                    txt.set_color("red" if vid == focus_vid else "black")
                txt.set_visible(True)

            return list(vehicle_patches.values()) + list(vehicle_text.values()) + [time_text]

        ani = animation.FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)

        videos_dir = os.path.join("simulation_results", "videos")
        os.makedirs(videos_dir, exist_ok=True)
        base_name = os.path.basename(str(process.name).rstrip("/\\"))
        video_path = os.path.join(videos_dir, f"waymo_train_{base_name}.mp4")

        try:
            writer = animation.FFMpegWriter(fps=10, bitrate=8000)
            ani.save(video_path, writer=writer)
            print(f"2D animation saved to: {video_path}")
        except Exception as e:
            print(f"Save mp4 failed: {e}")
            gif_path = video_path.replace(".mp4", ".gif")
            try:
                ani.save(gif_path, writer="pillow", fps=10)
                print(f"GIF animation saved: {gif_path}")
            except Exception as gif_e:
                print(f"Save GIF also failed: {gif_e}")
        plt.close(fig)


# =================== Potential field video (kept) ===================

def plot_potential_field_video(process: "RandomProcess", metrics_data: dict):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Polygon
    from matplotlib.ticker import MaxNLocator

    with plot_lock:

        q_i_data = metrics_data.get("q_i_over_time", {})
        ego_id = None
        best = -1.0
        for vid, series in q_i_data.items():
            if not series:
                continue
            avg_q = float(np.mean(list(series.values())))
            if avg_q > best:
                best = avg_q
                ego_id = vid
        if ego_id is None:
            print("No vehicle found for generating potential field video.")
            return

        out_dir = os.path.join("simulation_results", process.name)
        os.makedirs(out_dir, exist_ok=True)

        all_times = sorted(process.states.keys())
        q_ts = sorted(q_i_data.get(ego_id, {}).keys())
        if len(q_ts) < 2:
            return

        start_t, end_t = float(q_ts[0]), float(q_ts[-1])
        interaction_times = [t for t in all_times if start_t <= t <= end_t]
        if len(interaction_times) < 2:
            return

        ego_xy = []
        for t in interaction_times:
            st = process.states.get(t, {}).get(ego_id)
            if _is_valid_state(st):
                ego_xy.append((float(st.x), float(st.y)))
        if len(ego_xy) < 2:
            return

        ego_xy = np.array(ego_xy, dtype=float)
        ego_min_x, ego_max_x = float(np.min(ego_xy[:, 0])), float(np.max(ego_xy[:, 0]))
        ego_min_y, ego_max_y = float(np.min(ego_xy[:, 1])), float(np.max(ego_xy[:, 1]))

        zoom_half = float(max(POTENTIAL_NEIGHBOR_RADIUS + 8.0, 30.0))
        grid_margin = float(zoom_half + 12.0)

        x_min, x_max = ego_min_x - grid_margin, ego_max_x + grid_margin
        y_min, y_max = ego_min_y - grid_margin, ego_max_y + grid_margin


        res = 1.0
        x_vals = np.arange(x_min, x_max, res)
        y_vals = np.arange(y_min, y_max, res)
        X, Y = np.meshgrid(x_vals, y_vals)

        frames = [np.zeros_like(X, dtype=float) for _ in interaction_times]
        for k, t in enumerate(interaction_times):
            ego_state = process.states.get(t, {}).get(ego_id)
            if not _is_valid_state(ego_state):
                continue
            frames[k] = _potential_kernel_field_vectorized(ego_state, X, Y)

        try:
            vmax_global = float(np.max([np.max(f) for f in frames]))
            vmax_global = max(vmax_global, 1e-6)
        except Exception:
            vmax_global = 1.0


        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
        })

        fig = plt.figure(figsize=(7.4, 5.6), dpi=220)
        gs = fig.add_gridspec(1, 2, width_ratios=[22, 1], wspace=0.06)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

 
        cmap = plt.cm.Reds.copy()
        cmap.set_bad(color="white")
        extent = (x_min, x_max, y_min, y_max)

        im = ax.imshow(
            np.ma.masked_less_equal(frames[0], 1e-10),
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax_global,
            interpolation="bilinear",
            alpha=0.95,
            zorder=0,
        )

        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Potential")
        cb.locator = MaxNLocator(nbins=6)
        cb.update_ticks()

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.tick_params(direction="in", length=3, width=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        
        def safe_meta(vid: int):
            m = process.vehicle_metadata.get(vid, {}) if hasattr(process, "vehicle_metadata") else {}
            width = float(m.get("width", 2.0))
            length = float(m.get("length", 4.0))

            base_scale = float(getattr(process, "vehicle_size_scale", 1.8))
            size_scale = 0.8 * base_scale

            width *= size_scale
            length *= size_scale
            return {"width": width, "length": length, "id": m.get("id", vid)}

        def _safe_float_attr(st, name):
            if hasattr(st, name):
                try:
                    v = float(getattr(st, name))
                    if np.isfinite(v):
                        return v
                except Exception:
                    return None
            return None

        def phi_rad(st):
 
            phi = _safe_float_attr(st, "phi")

            if phi is None:
                for nm in ("yaw", "heading", "theta", "psi"):
                    phi = _safe_float_attr(st, nm)
                    if phi is not None:
                        break

            if phi is None:
                vx = None
                vy = None
                for nm in ("vx", "v_x", "vel_x"):
                    vx = _safe_float_attr(st, nm)
                    if vx is not None:
                        break
                for nm in ("vy", "v_y", "vel_y"):
                    vy = _safe_float_attr(st, nm)
                    if vy is not None:
                        break
                if vx is not None and vy is not None and (abs(vx) + abs(vy) > 1e-6):
                    phi = float(np.arctan2(vy, vx))
                else:
                    phi = 0.0

  
            if abs(phi) > (2.0 * np.pi + 0.5):
                phi = float(np.deg2rad(phi))

      
            phi = (phi + np.pi) % (2.0 * np.pi) - np.pi
            return float(phi)

        def safe_corners(x, y, w, l, phi):
         
            try:
                return get_rotated_rect_corners(x, y, w, l, phi)
            except Exception:
                
                hw, hl = 0.5 * w, 0.5 * l
                pts = np.array([
                    [ hl,  hw],
                    [ hl, -hw],
                    [-hl, -hw],
                    [-hl,  hw],
                ], dtype=float)
                c, s = float(np.cos(phi)), float(np.sin(phi))
                R = np.array([[c, -s], [s, c]], dtype=float)
                pts = pts @ R.T
                pts[:, 0] += float(x)
                pts[:, 1] += float(y)
                return [(float(px), float(py)) for px, py in pts]

        def label_for_vid(vid):
            if hasattr(process, "vehicle_metadata"):
                return str(process.vehicle_metadata.get(vid, {}).get("id", vid))
            return str(vid)

   
        all_vids = set()
        for t in interaction_times:
            all_vids.update(process.states.get(t, {}).keys())

        vehicle_patch = {}
        heading_line = {}
        vehicle_text = {}

        for vid in all_vids:
            p = Polygon([[0.0, 0.0]], closed=True, facecolor="none",
                        edgecolor="black", linewidth=1.0, zorder=10, visible=False)
            ax.add_patch(p)
            vehicle_patch[vid] = p

            ln, = ax.plot([], [], color="black", linewidth=1.0, alpha=0.95, zorder=11, visible=False)
            heading_line[vid] = ln

            tx = ax.text(
                0.0, 0.0, "",
                fontsize=8, ha="left", va="center",
                zorder=12, visible=False,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", alpha=0.85, lw=0.6),
            )
            vehicle_text[vid] = tx

  
        def update(frame_idx):
            t = interaction_times[frame_idx]
            frame_states = process.states.get(t, {})
            ego_state = frame_states.get(ego_id)

            if not _is_valid_state(ego_state):
                im.set_data(np.ma.masked_less_equal(np.zeros_like(frames[0]), 1e-10))
                for vid in all_vids:
                    vehicle_patch[vid].set_visible(False)
                    heading_line[vid].set_visible(False)
                    vehicle_text[vid].set_visible(False)
                ax.set_title("2D Potential Field")
                return [im] + list(vehicle_patch.values()) + list(heading_line.values()) + list(vehicle_text.values())

            ax.set_xlim(float(ego_state.x) - zoom_half, float(ego_state.x) + zoom_half)
            ax.set_ylim(float(ego_state.y) - zoom_half, float(ego_state.y) + zoom_half)

     
            im.set_data(np.ma.masked_less_equal(frames[frame_idx], 1e-10))

            neighbors = []
            for vid, st in frame_states.items():
                if vid == ego_id or not _is_valid_state(st):
                    continue
                if calculate_distance(ego_state, st) <= POTENTIAL_NEIGHBOR_RADIUS:
                    neighbors.append(vid)
            neighbor_ids = set(neighbors)
            q_pi_curr = calculate_q_pi(
                ego_state,
                [frame_states[v] for v in neighbors if _is_valid_state(frame_states.get(v))]
            )

         
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()

            def in_view(st):
                return _is_valid_state(st) and (x0 <= float(st.x) <= x1) and (y0 <= float(st.y) <= y1)

         
            for vid in all_vids:
                st = frame_states.get(vid)
                if not in_view(st):
                    vehicle_patch[vid].set_visible(False)
                    heading_line[vid].set_visible(False)
                    vehicle_text[vid].set_visible(False)
                    continue

                meta = safe_meta(vid)
                phi = phi_rad(st)

              
                if vid == ego_id:
                    color = "blue"
                    lw = 2.0
                    z = 20
                elif vid in neighbor_ids:
                    color = "red"
                    lw = 1.6
                    z = 18
                else:
                    color = "black"
                    lw = 1.1
                    z = 16

              
                corners = safe_corners(st.x, st.y, meta["width"], meta["length"], phi)
                p = vehicle_patch[vid]
                p.set_xy(corners)
                p.set_edgecolor(color)
                p.set_linewidth(lw)
                p.set_zorder(z)
                p.set_visible(True)

         
                L = float(meta["length"])
                hx = float(st.x) + 0.55 * L * float(np.cos(phi))
                hy = float(st.y) + 0.55 * L * float(np.sin(phi))
                ln = heading_line[vid]
                ln.set_data([float(st.x), hx], [float(st.y), hy])
                ln.set_color(color)
                ln.set_linewidth(max(1.0, lw - 0.1))
                ln.set_zorder(z + 0.1)
                ln.set_visible(True)

          
                nx = -float(np.sin(phi))
                ny =  float(np.cos(phi))
                offset = 0.5 * float(meta["width"]) + 2.0
                tx = vehicle_text[vid]
                tx.set_text(f"{meta['id']}")
                tx.set_position((float(st.x) + nx * offset, float(st.y) + ny * offset))
                tx.set_color(color if color != "black" else "black")
                tx.set_zorder(z + 0.2)
                tx.set_visible(True)

            veh_id_str = label_for_vid(ego_id)
      
            ax.set_title(f"Vehicle {veh_id_str} 2D Potential Field (Q_pi={q_pi_curr:.3f}, N={len(neighbor_ids)})")

            return [im] + list(vehicle_patch.values()) + list(heading_line.values()) + list(vehicle_text.values())

        ani = animation.FuncAnimation(fig, update, frames=len(interaction_times), blit=False)

  
        veh_id_str = label_for_vid(ego_id)
        video_path = os.path.join(out_dir, f"potential_field_vehicle_{veh_id_str}_2d_sci.mp4")
        try:
            writer = animation.FFMpegWriter(
                fps=10,
                bitrate=8000,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p"]
            )
            ani.save(video_path, writer=writer, dpi=220)
            print(f"2D video saved to: {video_path}")
        except Exception as e:
            print(f"Save mp4 failed: {e}")
            gif_path = video_path.replace(".mp4", ".gif")
            try:
                ani.save(gif_path, writer="pillow", fps=10, dpi=220)
                print(f"GIF animation saved: {gif_path}")
            except Exception as gif_e:
                print(f"Save GIF also failed: {gif_e}")

        plt.close(fig)


# =================== Metrics plots (kept) ===================

def plot_state_metrics_over_time(process: "RandomProcess", metrics_data: dict):
    """
    Plot ONLY ego vehicle's 4 Q-metrics (Q_i, Q_si, Q_ri, Q_pi) on ONE axis,
    SCI-paper-friendly style + high-quality saving.

    Output:
      simulation_results/<process.name>/ego_state_metrics_over_time.png
      simulation_results/<process.name>/ego_state_metrics_over_time.pdf
    """
    with plot_lock:
        def _squash_01(x: float) -> float:
            x = max(0.0, float(x))
            return 1.0 - np.exp(-x)

        q_i_data  = metrics_data.get("q_i_over_time", {})
        q_si_data = metrics_data.get("q_si_over_time", {})
        q_ri_data = metrics_data.get("q_ri_over_time", {})
        q_pi_data = metrics_data.get("q_pi_over_time", {})

        # ---- Determine ego id robustly ----
        ego_id = None
        # common patterns
        for k in ["ego_id", "ego_vid", "ego_vehicle_id"]:
            if hasattr(process, k):
                ego_id = getattr(process, k)
                break
        if ego_id is None:
            for k in ["ego_id", "ego_vid", "ego_vehicle_id"]:
                if k in process.vehicle_metadata:
                    ego_id = process.vehicle_metadata[k]
                    break
        # fallback: the one named "ego"
        if ego_id is None:
            for vid, meta in getattr(process, "vehicle_metadata", {}).items():
                if isinstance(meta, dict) and str(meta.get("role", "")).lower() == "ego":
                    ego_id = vid
                    break
        # final fallback: pick the first vehicle that has Q_i series
        if ego_id is None:
            for vid in sorted(q_i_data.keys()):
                if q_i_data.get(vid, {}):
                    ego_id = vid
                    break

        if ego_id is None:
            print("No ego ID found, cannot plot.")
            return

        # ---- Fetch ego series ----
        series_qi  = q_i_data.get(ego_id, {})
        series_qsi = q_si_data.get(ego_id, {})
        series_qri = q_ri_data.get(ego_id, {})
        series_qpi = q_pi_data.get(ego_id, {})

        if not any([series_qi, series_qsi, series_qri, series_qpi]):
            print("Ego has no available Q metrics time series, cannot plot.")
            return

        # unify time axis by union of timestamps
        t_all = sorted(set(series_qi.keys()) | set(series_qsi.keys()) | set(series_qri.keys()) | set(series_qpi.keys()))
        if len(t_all) < 2:
            print("Ego Q metrics time points too few, cannot plot.")
            return

        def _get_or_nan(series: dict, t: float):
            return series.get(t, np.nan)

        y_qi  = np.array([_get_or_nan(series_qi,  t) for t in t_all], dtype=float)
        y_qsi = np.array([_squash_01(_get_or_nan(series_qsi, t)) if np.isfinite(_get_or_nan(series_qsi, t)) else np.nan for t in t_all], dtype=float)
        y_qri = np.array([_squash_01(max(0.0, _get_or_nan(series_qri, t))) if np.isfinite(_get_or_nan(series_qri, t)) else np.nan for t in t_all], dtype=float)
        y_qpi = np.array([_squash_01(_get_or_nan(series_qpi, t)) if np.isfinite(_get_or_nan(series_qpi, t)) else np.nan for t in t_all], dtype=float)

        # ---- SCI-paper style ----
        paper_style = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "savefig.bbox": "tight",
        }

        with plt.rc_context(paper_style):
            fig, ax = plt.subplots(figsize=(7.2, 3.6))  # good for 2-column figure

            # 4 curves on one axis
            ax.plot(t_all, y_qi,  label=r"$Q_i(t)$")
            ax.plot(t_all, y_qsi, label=r"$Q_{si}(t)$")
            ax.plot(t_all, y_qri, label=r"$Q_{ri}(t)$")
            ax.plot(t_all, y_qpi, label=r"$Q_{pi}(t)$")

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Q (0–1)")
            ax.set_title("Ego Vehicle Q Metrics Over Time")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, linestyle="--", alpha=0.25)

            # Legend: upper-right inside axis (clean for papers)
            ax.legend(loc="upper right", frameon=True, borderpad=0.6, handlelength=2.0)

            out_dir = os.path.join("simulation_results", process.name)
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, "ego_state_metrics_over_time.png")
            out_pdf = os.path.join(out_dir, "ego_state_metrics_over_time.pdf")

            fig.savefig(out_png, dpi=600)
            fig.savefig(out_pdf)
            plt.close(fig)

            print(f"Ego state metrics curve saved: {out_png}")
            print(f"Ego state metrics curve saved: {out_pdf}")



def plot_efficiency_metrics(process: RandomProcess, metrics_data: dict):

    with plot_lock:
        e_metrics = metrics_data.get("e_metrics", {})
        if not e_metrics:
            print("No efficiency metrics can be plotted.")
            return

 
        out_dir = os.path.join("simulation_results", process.name)
        os.makedirs(out_dir, exist_ok=True)

        # build reverse map: meta_id(str) -> vid
        id_to_vid = {}
        for vid, meta in getattr(process, "vehicle_metadata", {}).items():
            if isinstance(meta, dict) and "id" in meta:
                id_to_vid[str(meta["id"])] = vid

      
        red_vids = set()
        try:
            import glob
            for ext in ("mp4", "gif"):
                for fp in glob.glob(os.path.join(out_dir, f"potential_field_vehicle_*.{ext}")):
                    base = os.path.basename(fp)
                    tag = base.replace("potential_field_vehicle_", "").replace(f".{ext}", "")
                    if tag in id_to_vid:
                        red_vids.add(id_to_vid[tag])
        except Exception:
            pass

    
        highest = {"avg_q": -1.0, "ego": None, "info": None}
        q_i_data = metrics_data.get("q_i_over_time", {}) or {}
        for ego_id, info_list in getattr(process, "interaction_info_by_vehicle", {}).items():
            for info in info_list:
                if getattr(info, "type", None) == "Car-follow":
                    continue
                series = q_i_data.get(ego_id, {})
                if not series:
                    continue
                q_vals = [
                    series.get(t, 0.0) for t in series.keys()
                    if getattr(info, "start_time", -1e9) <= t <= getattr(info, "end_time", 1e9)
                ]
                if not q_vals:
                    continue
                avg_q = float(np.mean(q_vals))
                if avg_q > highest["avg_q"]:
                    highest = {"avg_q": avg_q, "ego": ego_id, "info": info}

        if highest["ego"] is not None:
            red_vids.add(highest["ego"])


        blue_vids = set()

        def _add_partner_ids_from_info(info_obj):
            if info_obj is None:
                return
            candidate_attrs = [
                "other_vehicle", "other_vehicle_id", "other_id",
                "partner_id", "partner_vid", "target_id",
                "target_vehicle", "target_vehicle_id",
                "agent_id", "agent", "vid_j", "vehicle_j",
                "other_vid", "other_vehicle_ids", "participant_ids",
                "participants", "vehicles", "vehicle_ids"
            ]
            for a in candidate_attrs:
                if not hasattr(info_obj, a):
                    continue
                v = getattr(info_obj, a)
                if v is None:
                    continue
                if isinstance(v, dict):
                    vals = list(v.values())
                elif isinstance(v, (list, tuple, set)):
                    vals = list(v)
                else:
                    vals = [v]
                for x in vals:
                    try:
                        if isinstance(x, (int, np.integer)):
                            blue_vids.add(int(x))
                        elif isinstance(x, str) and x.isdigit():
                            blue_vids.add(int(x))
                    except Exception:
                        continue

        info = highest.get("info", None)
        _add_partner_ids_from_info(info)

        # overlap fallback
        if (not blue_vids) and (highest["ego"] is not None) and (info is not None):
            s0 = float(getattr(info, "start_time", -1e9))
            e0 = float(getattr(info, "end_time", 1e9))
            for vid2, info_list in getattr(process, "interaction_info_by_vehicle", {}).items():
                if vid2 == highest["ego"]:
                    continue
                for inf2 in info_list:
                    if getattr(inf2, "type", None) == "Car-follow":
                        continue
                    s1 = float(getattr(inf2, "start_time", -1e9))
                    e1 = float(getattr(inf2, "end_time", 1e9))
                    if max(s0, s1) <= min(e0, e1):
                        blue_vids.add(vid2)
                        break

        highlight_vids = set(red_vids) | set(blue_vids)
        if not highlight_vids:
            print("Unable to infer the (red/blue) vehicle set to plot from plot_potential_field_video.")
            return

  
        import re

        def _clean_vehicle_label(tag) -> str:
       
            s = str(tag).strip()

            s = re.sub(r"\(.*\)\s*$", "", s).strip()
       
            s = re.sub(r"^[Vv]\s*", "", s).strip()

           
            m = re.search(r"(\d+)\s*_\s*(\d+)", s)
            if m:
                return f"{m.group(1)}_{m.group(2)}"

            nums = re.findall(r"\d+", s)
            if len(nums) == 1:
                return nums[0]

         
            return s

       
        candidate_labels = {}
        for vid in sorted(highlight_vids):
            meta = process.vehicle_metadata.get(vid, {}) if hasattr(process, "vehicle_metadata") else {}
            raw = None
            if isinstance(meta, dict):
                
                for k in ("id", "vehicle_id", "vid", "name", "tag"):
                    if k in meta and meta[k] is not None:
                        raw = meta[k]
                        break
            if raw is None:
                raw = vid
            candidate_labels[vid] = _clean_vehicle_label(raw)

      
        label_counts = {}
        for vid, lab in candidate_labels.items():
            label_counts[lab] = label_counts.get(lab, 0) + 1
        final_labels = {}
        for vid, lab in candidate_labels.items():
            if (lab is None) or (str(lab).strip() == "") or (label_counts.get(lab, 0) > 1):
                final_labels[vid] = str(vid)  
            else:
                final_labels[vid] = str(lab)

        plot_rows = []
        for vid in sorted(highlight_vids):
            if vid not in e_metrics:
                continue
            em = e_metrics[vid]
            plot_rows.append((
                final_labels.get(vid, str(vid)),
                float(em.get("e_i", 0.0)),
                float(em.get("e_pi", 0.0)),
                float(em.get("e_ti", 0.0)),
                float(em.get("e_si", 0.0)),
            ))

        if not plot_rows:
            print("Highlighted vehicle set has no available efficiency metrics (e_metrics missing).")
            return

        labels = [r[0] for r in plot_rows]
        e_i  = [r[1] for r in plot_rows]
        e_pi = [r[2] for r in plot_rows]
        e_ti = [r[3] for r in plot_rows]
        e_si = [r[4] for r in plot_rows]

     
        paper_style = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "savefig.bbox": "tight",
        }

        
        x = np.arange(len(labels))
        width = 0.2

        
        c_ei  = "#0072B2"  # blue
        c_epi = "#E69F00"  # orange
        c_eti = "#009E73"  # green
        c_esi = "#CC79A7"  # magenta

        fig_w = max(7.2, 0.65 * len(labels) + 3.0)
        fig_h = 3.6

        with plt.rc_context(paper_style):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            ax.bar(x - 1.5 * width, e_i,  width, label=r"$E_i$",    color=c_ei)
            ax.bar(x - 0.5 * width, e_pi, width, label=r"$E_{pi}$", color=c_epi)
            ax.bar(x + 0.5 * width, e_ti, width, label=r"$E_{ti}$", color=c_eti)
            ax.bar(x + 1.5 * width, e_si, width, label=r"$E_{si}$", color=c_esi)

            ax.set_xlabel("Vehicle ID")
            ax.set_ylabel("E (0–1)")
            ax.set_title("E Metrics of Vehicles")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0, ha="center")

            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, axis="y", linestyle="--", alpha=0.25)

         
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=True,
                borderpad=0.6,
                handlelength=2.0,
                borderaxespad=0.0
            )

         
            fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])

            out_png = os.path.join(out_dir, "efficiency_metrics.png")
            out_pdf = os.path.join(out_dir, "efficiency_metrics.pdf")
            fig.savefig(out_png, dpi=600)
            fig.savefig(out_pdf)
            plt.close(fig)

            print(f"Efficiency metrics bar chart saved: {out_png}")
            print(f"Efficiency metrics bar chart saved: {out_pdf}")



# =================== Process generation ===================

def generate_random_process(scene_processor: Any) -> RandomProcess:
    """
    Build RandomProcess from a SceneProcessor and run interaction detection.
    """
    process = RandomProcess()
    process.name = str(scene_processor.desired_scene)

    agent_states = scene_processor.agent_states
    num_vehicles = agent_states.shape[0]
    total_steps = agent_states.shape[1]
    dt = scene_processor.dt
    process.duration = (total_steps - 1) * dt
    process.num_participants = num_vehicles

    agent_ids = list(scene_processor.all_agents.keys())
    column_keys = ["x", "y", "vx", "vy", "ax", "ay", "heading"]
    indices = {k: scene_processor.column_dict[k] for k in column_keys}

    def _get_dim(ext, *candidates):
        for c in candidates:
            if hasattr(ext, c):
                return float(getattr(ext, c))
        return 1.0

    for i in range(num_vehicles):
        agent_id = agent_ids[i]
        meta = scene_processor.all_agents[agent_id]
        ext = meta.extent
        process.vehicle_metadata[i] = {
            "id": agent_id,
            "length": _get_dim(ext, "length", "length_m", "l", "x"),
            "width": _get_dim(ext, "width", "width_m", "w", "y"),
            "height": _get_dim(ext, "height", "height_m", "h", "z"),
        }

    for step in range(total_steps):
        t = step * dt
        current = {}
        for i in range(num_vehicles):
            s = agent_states[i, step]
            vx, vy = float(s[indices["vx"]]), float(s[indices["vy"]])
            ax_, ay_ = float(s[indices["ax"]]), float(s[indices["ay"]])
            current[i] = VehicleState(
                x=float(s[indices["x"]]),
                y=float(s[indices["y"]]),
                phi=float(s[indices["heading"]]),
                v=float(np.hypot(vx, vy)),
                a=float(np.hypot(ax_, ay_)),
            )
        process.states[t] = current

    print(f"Running interaction detection for scene: {process.name}...")
    detect_and_classify_interactions(process)
    print(f"Detection completed. Found {len(process.interaction_participants)} interacting vehicles.")
    return process
