import numpy as np
from pathlib import Path

from trajdata.data_structures import Scene
from trajdata.caching import  EnvCache
from trajdata import  VectorMap
from trajdata.data_structures import AgentType
from trajdata.caching.df_cache import DataFrameCache

def load_random_scene(cache_path: Path, env_name: str, scene_dt: float) -> Scene:
    env_cache = EnvCache(cache_path)
    scenes_list = env_cache.load_env_scenes_list(env_name)
    random_scene_name = scenes_list[np.random.randint(0, len(scenes_list))].name
    print(scenes_list)
    print(random_scene_name)

    return env_cache.load_scene(env_name, random_scene_name, scene_dt)


def print_lane_connections(vector_map: VectorMap, lane_id: str):
    lane = vector_map.get_road_lane(lane_id)

    print("Previous Lanes:")
    for prev_lane_id in lane.prev_lanes:
        print(f"  - {prev_lane_id}")


    print("Next Lanes:")
    for next_lane_id in lane.next_lanes:
        print(f"  - {next_lane_id}")

    print("Adjacent Lanes Left:")
    for left_lane_id in lane.adj_lanes_left:
        print(f"  - {left_lane_id}")

 
    print("Adjacent Lanes Right:")
    for right_lane_id in lane.adj_lanes_right:
        print(f"  - {right_lane_id}")

def current_lane_id(lane_kd_tree, query_point, distance_threshold = 3, heading_threshold = 20):  
    heading_threshold = np.pi /heading_threshold 

    lane_indices = lane_kd_tree.current_lane_inds(
        xyzh=query_point,
        distance_threshold=distance_threshold,
        heading_threshold=heading_threshold
    )
    return lane_indices


def get_agent_states(interact_ids, all_agents, vec_map, lane_kd_tree, sc, desired_scene, column_dict, all_timesteps):

    agent_states = np.zeros((len(all_agents), desired_scene.length_timesteps, 8))
    
    agent_lane_ids = {agent.name: [0] * len(all_timesteps) for agent in desired_scene.agents}


    for agent in desired_scene.agents:
        current_lane = None
        
        x_index = column_dict['x']
        y_index = column_dict['y']
        z_index = column_dict['z']
        heading_index = column_dict['heading']

        
        for t in range(agent.first_timestep, agent.last_timestep + 1):
            
            raw_state = sc.get_raw_state(agent_id=agent.name, scene_ts=t)
            query_point = np.array([raw_state[x_index], raw_state[y_index], raw_state[z_index], raw_state[heading_index]])
            
         
            lane_indices = current_lane_id(lane_kd_tree, query_point)
            lane_indices = [vec_map.lanes[i].id for i in lane_indices]

            
            if len(lane_indices) > 1:
                query_point = np.array([raw_state[x_index], raw_state[y_index], raw_state[z_index]])
                closest = lane_kd_tree.closest_polyline_ind(query_point)
                closest_lane_id = vec_map.lanes[int(closest)].id
                if closest_lane_id in lane_indices:
                    chosen_lane = closest_lane_id
                else:
                    chosen_lane = next((lan for lan in lane_indices if lan == current_lane), lane_indices[0])
            elif len(lane_indices) == 0:
             
                query_point = np.array([raw_state[x_index], raw_state[y_index], raw_state[z_index]])
                closest = lane_kd_tree.closest_polyline_ind(query_point)
                chosen_lane = vec_map.lanes[int(closest)].id
            else:

                chosen_lane = lane_indices[0]

            current_lane = chosen_lane

            try:

                agent_index = all_agents.index(agent.name)
                timestep_index = all_timesteps.index(t)
                agent_states[agent_index, timestep_index, :] = raw_state
            except:
                continue


            agent_lane_ids[agent.name][timestep_index] = chosen_lane

    return agent_states, agent_lane_ids