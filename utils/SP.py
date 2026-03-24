import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from shapely.geometry import LineString, Point
from trajdata.data_structures import AgentType
from .trajdata_utils import DataFrameCache




class SceneProcessor:

    def __init__(
        self,
        desired_data: List[str],
        idx: int,
        desired_scene: Any,
        map_api: Optional[Any] = None,  
        cache_path: str = "",
        timerange: Optional[int] = None,
    ):
        self.timerange = timerange or 0
        self.desired_data = desired_data
        self.idx = idx
        self.desired_scene = desired_scene
        self.cache_path = cache_path
        self.dt = desired_scene.dt

        
        self.all_agents = {
            agent.name: agent
            for agent in desired_scene.agents
            if agent.type == AgentType.VEHICLE
        }
        self.length_timesteps = desired_scene.length_timesteps
        self.all_timesteps = list(range(self.length_timesteps))

        
        self.scene_cache = DataFrameCache(cache_path=cache_path, scene=desired_scene)
        self.column_dict: Dict[str, int] = self.scene_cache.column_dict

        
        self.vec_map = None
        self.lane_id = []
        self.lanes = {}

        
        self.scene_cache.load_kdtrees()

    
    def process_scene(
        self,
        time_dic_output: Dict,
        a_min_dict: Dict,
        multi_a_min_dict: Dict,
        delta_TTCP_dict: Dict,
        results: Dict,
    ) -> List[List]:
        
        self.get_agent_states()
        return []

    
    def get_agent_states(self):
        
        x_idx = self.column_dict["x"]
        y_idx = self.column_dict["y"]
        z_idx = self.column_dict["z"]
        h_idx = self.column_dict["heading"]

        n_agents = len(self.all_agents)
        n_steps = self.length_timesteps
        
        self.agent_states = np.zeros((n_agents, n_steps, 8 + 1))
        self.agent_lane_ids = {}

        for name, agent in self.all_agents.items():
            self.agent_lane_ids[name] = [None] * n_steps
            agent_idx = list(self.all_agents.keys()).index(name)
            for t in range(agent.first_timestep, agent.last_timestep + 1):
                raw = self.scene_cache.get_raw_state(agent_id=name, scene_ts=t)
                time_idx = self.all_timesteps.index(t)
                self.agent_states[agent_idx, time_idx, :-1] = raw
                self.agent_states[agent_idx, time_idx, -1] = -1.0  

    
    def get_move_agent(self) -> Dict[int, List[str]]:
        
        move_agent = {}
        vx_idx = self.column_dict["vx"]
        for t_idx, t in enumerate(self.all_timesteps):
            move_agent[t] = [
                list(self.all_agents.keys())[aidx]
                for aidx, vx in enumerate(self.agent_states[:, t_idx, vx_idx])
                if vx != 0
            ]
        return move_agent

    def get_fut_line(self, move_agent: Dict[int, List[str]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
       
        return {}

    def get_results(self, *args, **kwargs) -> List[List]:
        return []

    def get_line(self, timestamp, track_id1, track_id2):
        return None, None

    
    def find_last_nonzero_index(self, agent_states: np.ndarray, track_id_index: int, position_index: List[int]) -> Optional[int]:
        xs = agent_states[track_id_index, :, position_index[0]]
        nz = xs != 0
        return int(np.where(nz)[0][-1]) if nz.any() else None

    def process_tracks(self, all_tracks: np.ndarray, track_id_index: int, timestamp_index: int,
                       timerange: int, position_index: List[int], v_index: List[int], dt: float,
                       additional_trajectory: Optional[np.ndarray] = None) -> Dict[str, Any]:
        track = all_tracks[track_id_index, timestamp_index:timestamp_index + int(timerange / dt), position_index]
        valid = np.any(track != 0, axis=1)
        track = track[valid]
        if track.shape[0] < 2:
            return {"line": None, "velocity": 0.0}
        line = LineString(track)
        vx, vy = all_tracks[track_id_index, timestamp_index, v_index]
        v = float(np.hypot(vx, vy))
        return {"line": line, "velocity": v}

    def concatenate_trajectory(self, original_line: LineString, additional_trajectory: np.ndarray) -> LineString:
        if additional_trajectory.size == 0:
            return LineString()
        return LineString(np.vstack([original_line.coords, additional_trajectory[:, :2]]))

    def get_downstream_lane_ids(self, *args, **kwargs) -> List[int]:
        return []

    def get_complete_lane(self, *args, **kwargs) -> np.ndarray:
        return np.array([])

    def extract_path_from_current_position(self, *args, **kwargs) -> np.ndarray:
        return np.array([])

    def process_vehicle_tracks(self, track_id_index: int, timestamp_index: int, lanes: List) -> Tuple:
        return [], 0.0, 0.0, 0.0, [], 0.0, 0.0

    def calculate_future_time(self, timestamp: int) -> float:
        return 0.0


    def dic_to_array(self, msaa_dict: Dict[int, Dict[Tuple[int, int], List[float]]]
                    ) -> Tuple[np.ndarray, List, List]:
        ts_labels = list(msaa_dict.keys())
        row_labels = sorted({pair for d in msaa_dict.values() for pair in d.keys()})
        col_labels = list(range(min(ts_labels), max(ts_labels) + 1))
        mat = np.zeros((len(row_labels), len(col_labels)))
        for j, t in enumerate(col_labels):
            for i, p in enumerate(row_labels):
                mat[i, j] = msaa_dict.get(t, {}).get(p, [0])[0]
        return mat, row_labels, col_labels

    def get_segments(self, row_labels, col_labels, a_min_matrix: np.ndarray,
                     acceleration_thresh: float) -> Dict:
        segments = {}
        for i, pair in enumerate(row_labels):
            start = None
            for j, val in enumerate(a_min_matrix[i, :]):
                if val >= acceleration_thresh:
                    if start is None:
                        start = col_labels[j]
                    end = col_labels[j]
                elif start is not None:
                    segments.setdefault(pair, []).append((start, end))
                    start = None
            if start is not None:
                segments.setdefault(pair, []).append((start, end))
        return segments