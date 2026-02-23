import logging
import os
import gc
import traceback

from tqdm import tqdm
from concurrent.futures import as_completed
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yaml
import time

from trajdata import UnifiedDataset, MapAPI

from .SP import SceneProcessor
from .IEDD_compute import generate_random_process, calculate_all_metrics, plot_state_metrics_over_time, plot_efficiency_metrics, plot_potential_field_video, plot_2d_animation,save_results_to_csv
import threading
from concurrent.futures import as_completed

# Set up the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_config(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

class InteractionProcessorTraj:

    def __init__(self, desired_data: str, param: Optional[Any], cache_location: str, save_path: str, num_workers: int):
        # Set configuration and instance variables
        self.desired_data = desired_data
        self.save_path = save_path
        self.timerange = param


        # Initialize internal state variables for data processing
        self.scene_counter = 0
        self.time_dic_output = {}
        self.a_min_dict = {}
        self.multi_a_min_dict = {}
        self.delta_TTCP_dict = {}
        self.results = {}

        # Initialize dataset with configuration settings
        self.dataset = UnifiedDataset(
            desired_data=[self.desired_data],
            standardize_data=False,
            rebuild_cache=False,
            rebuild_maps=False,
            centric="scene",
            verbose=True,
            cache_location=cache_location,
            num_workers=num_workers,
            incl_vector_map=True,
            data_dirs={self.desired_data:''}
        )

        # Set up MapAPI and cache paths
        self.map_api = MapAPI(self.dataset.cache_path)
        self.cache_path = self.dataset.cache_path

    def process(self):
        results_list = []
        scenes_list = list(self.dataset.scenes())  # Convert generator to list
        print(self.dataset.scenes())
        num_scenes = len(scenes_list)  # Get total number of scenes
        batch_size = 100  # Number of scenes to process per batch

        # Initialize tqdm progress bar
        with tqdm(total=num_scenes, desc="Processing Scenes", unit="scene") as pbar:
            # Submit tasks in batches
            for start_idx in range(0, num_scenes, batch_size):
                end_idx = min(start_idx + batch_size, num_scenes)
                batch_scenes = scenes_list[start_idx:end_idx]

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(self.process_single_scene, idx, desired_scene): (idx, desired_scene)
                        for idx, desired_scene in enumerate(batch_scenes, start=start_idx)
                    }

                    for future in as_completed(futures):
                        idx, scene = futures[future]
                        try:
                            scene_results = future.result()
                            results_list.extend(scene_results)
                        except Exception as e:
                            logger.error(f"Scene {idx} failed with error: {e}")
                            traceback.print_exc()
                        finally:
                            pbar.update(1)


    def process_single_scene(self, idx, desired_scene) -> List[List]:
        scene_processor = SceneProcessor(
            self.desired_data, idx, desired_scene, self.map_api, self.cache_path,
            self.timerange
        )
        ts1 = time.time()
        # Process the scene and collect results
        scene_results = scene_processor.process_scene(
            self.time_dic_output, self.a_min_dict, self.multi_a_min_dict, self.delta_TTCP_dict, self.results
        )


        random_process = generate_random_process(scene_processor)

   
        num_vehicles = random_process.num_participants
        total_time = random_process.duration
        metrics = calculate_all_metrics(random_process)

        # save_results_to_csv(random_process, metrics)
        # plot_state_metrics_over_time(random_process, metrics)
        # plot_efficiency_metrics(random_process, metrics)
        # plot_potential_field_video(random_process,metrics)
        # plot_2d_animation(random_process, metrics)



        del scene_processor, desired_scene
        gc.collect()

        return scene_results



