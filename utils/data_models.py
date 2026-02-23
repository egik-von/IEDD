

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict
from typing import List
from typing import Any

@dataclass
class VehicleState:
    def __init__(self, x, y, phi, v, a):
        self.x = x
        self.y = y
        self.phi = phi
        self.v = v
        self.a = a

class RandomProcess:
    def __init__(self):
        self.name = "default_scene"
        self.states: Dict[float, Dict[int, VehicleState]] = {}
        self.num_participants = 0
        self.duration = 0
        self.vehicle_metadata: Dict[int, Dict[str, Any]] = {}
        self.interaction_info_by_vehicle: Dict[int, List['InteractionInfo']] = defaultdict(list)
        self.interaction_participants: set = set()

class Interaction:

    def __init__(self, vehicle_pair: Tuple[int, int], interaction_type: str, start_time: float, end_time: float, cross_points_info: List[Tuple[Tuple[float, float], float, float]]):
        self.vehicle_pair = vehicle_pair
        self.type = interaction_type
        self.start_time = start_time
        self.end_time = end_time
        self.cross_points_info = cross_points_info

class InteractionInfo:
   
    def __init__(self, interaction_type: str, start_time: float, end_time: float, partners: Dict[int, Interaction]):
        self.type = interaction_type
        self.start_time = start_time
        self.end_time = end_time
        self.partners = partners
        self.e_i: float = 0.0
        self.e_pi: float = 0.0
        self.e_ti: float = 0.0
        self.e_si: float = 0.0




