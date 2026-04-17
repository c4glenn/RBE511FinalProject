#TODO: Move this file out and make it a library so linking works correctly
from typing import Optional, Tuple, List, Dict, Any
import datetime
import logging
import argparse
from dataclasses import dataclass
import copy
import numpy as np
import tqdm
import jsonpickle
import mesa
import pandas as pd
from model import SwarmModel

logger = logging.getLogger()

@dataclass
class ModelParams():
    n_robots:int | List[int] = 20
    n_tasks:int | List[int]=1
    speed:float | List[float] = 3.0
    seed: Optional[int] = None
    arena_width:float | List[float] = 600.0
    arena_height:float | List[float] = 200.0
    interface_gap:float | List[float] = 25.0
    task_dist_calc: int| List[int] = 50
    task_distribution: Optional[np.ndarray] | List[np.ndarray] = None
    robot_initial_placements: Optional[np.ndarray] | List[np.ndarray] = None
    allowed_to_switch:bool = True
    gamma: float | List[float] = 0.1
    k:float | List[float]= 5
    m:float | List[float]= 8
    switching_cost: int | List[int]= 30
    delay_random_range: Tuple[float, float] | List[Tuple[float, float]] = (0.0, 10.0)
    transfer_time: int | List[int]= 3
    pickup_time: int | List[int] = 1
    dropoff_time: int | List[int] = 1        
    
    def __post_init__(self):
        pass
        
    def create_swarm_model(self) -> SwarmModel:
        return SwarmModel(
            n_robots=self.n_robots if not isinstance(self.n_robots, list) else self.n_robots[0],
            n_tasks=self.n_tasks if not isinstance(self.n_tasks, list) else self.n_tasks[0],
            speed=self.speed if not isinstance(self.speed, list) else self.speed[0],
            seed=self.seed, # type: ignore
            arena_width=self.arena_width if not isinstance(self.arena_width, list) else self.arena_width[0],
            arena_height=self.arena_height if not isinstance(self.arena_height, list) else self.arena_height[0],
            interface_gap=self.interface_gap if not isinstance(self.interface_gap, list) else self.interface_gap[0],
            task_dist_calc=self.task_dist_calc if not isinstance(self.task_dist_calc, list) else self.task_dist_calc[0],
            task_distribution=self.task_distribution if not isinstance(self.task_distribution, list) else self.task_distribution[0],
            robot_initial_placements=self.robot_initial_placements if not isinstance(self.robot_initial_placements, list) else self.robot_initial_placements[0],
            allowed_to_switch=self.allowed_to_switch if not isinstance(self.allowed_to_switch, list) else self.allowed_to_switch[0],
            gamma=self.gamma if not isinstance(self.gamma, list) else self.gamma[0],
            k=self.k if not isinstance(self.k, list) else self.k[0],
            m=self.m if not isinstance(self.m, list) else self.m[0],
            switching_cost=self.switching_cost if not isinstance(self.switching_cost, list) else self.switching_cost[0],
            delay_random_range=self.delay_random_range if not isinstance(self.delay_random_range, list) else self.delay_random_range[0],
            transfer_time=self.transfer_time if not isinstance(self.transfer_time, list) else self.transfer_time[0],
            pickup_time=self.pickup_time if not isinstance(self.pickup_time, list) else self.pickup_time[0],
            dropoff_time=self.dropoff_time if not isinstance(self.dropoff_time, list) else self.dropoff_time[0],
        )
    
    def create_param_dict(self) -> Dict[str, Any]:
        return {
            "n_robots": self.n_robots,
            "n_tasks": self.n_tasks,
            "speed": self.speed,
            "seed": self.seed, # type: ignore
            "arena_width": self.arena_width,
            "arena_height": self.arena_height,
            "interface_gap": self.interface_gap,
            "task_dist_calc": self.task_dist_calc,
            "task_distribution": self.task_distribution,
            "robot_initial_placements": self.robot_initial_placements,
            "allowed_to_switch": self.allowed_to_switch,
            "gamma": self.gamma,
            "k": self.k,
            "m": self.m,
            "switching_cost": self.switching_cost,
            "delay_random_range": [self.delay_random_range],
            "transfer_time": self.transfer_time,
            "pickup_time": self.pickup_time,
            "dropoff_time": self.dropoff_time
        }

    def _optimal_concerns(self) -> str:
        return f"ModelParams({self.n_robots=},{self.n_tasks=}{self.speed=},{self.arena_width=},{self.arena_height=},{self.interface_gap=},{self.task_dist_calc=},{self.task_distribution=},{self.delay_random_range=},{self.transfer_time=},{self.pickup_time=},{self.dropoff_time=}"

    def __hash__(self) -> int:
        return hash(f"{self.n_robots}{self.n_tasks}{self.speed}{self.arena_width}{self.arena_height}{self.interface_gap}{self.task_dist_calc}{self.task_distribution}{self.robot_initial_placements}{self.delay_random_range}{self.transfer_time}{self.pickup_time}{self.dropoff_time}")

    def is_singleton(self) -> bool:
        return not ( isinstance(self.n_robots, list) or isinstance(self.n_tasks, list) or isinstance(self.speed, list) or isinstance(self.arena_width, list) or isinstance(self.arena_height, list) or isinstance(self.interface_gap, list) or isinstance(self.task_dist_calc, list) or isinstance(self.task_distribution, list) or isinstance(self.robot_initial_placements, list) or isinstance(self.allowed_to_switch, list) or isinstance(self.gamma, list) or isinstance(self.k, list) or isinstance(self.m, list) or isinstance(self.switching_cost, list) or isinstance(self.delay_random_range, list) or isinstance(self.transfer_time, list) or isinstance(self.pickup_time, list) or isinstance(self.dropoff_time, list) )


@dataclass
class RunResult(ModelParams):
    RunId: Optional[int] = None
    iteration: Optional[int] = None
    Step: Optional[int] = None
    total_deliveries: Optional[int] = None
    total_crossings: Optional[int] = None
    throughput: Optional[float] = None
    allocation: Optional[List[np.ndarray]] = None
    
    optimal_delivery_count: Optional[float] = None
    mae: Optional[float] = None
    
    def __post_init__(self):
        self.creation_time = datetime.datetime.now() #to help ensure uniqueness between runs? we shall see

    def save(self, filename: str):
        with open(filename, "a+") as f:
            f.write(f"{'\t'.join(str(v) for k,v in self.__dict__.items() if k != "allocation")}\n")
    
def assignments(num_robots, num_tasks, depth=0):
    if num_robots == 0: raise ValueError("wtf")
    if num_tasks == 1: 
        return [num_robots]
    options = []
    for i in range(1, num_robots-num_tasks+1):
        possible_vals = assignments(num_robots-i, num_tasks-1, depth+1)
        for val in possible_vals:
            try:
                options.append([i, *val])
            except:
                options.append([i, val])
                
    if options == []:
        options = [[0]*(num_tasks+1)]
        for i in range(0, num_robots+1):
            options[0][i%(num_tasks+1)] += 1
    return options

def find_optimal(params:ModelParams) -> Tuple[int, np.ndarray]:
    with open("optimal.json", "r+") as f:
        cache = jsonpickle.decode(f.read())
    # print([x for x in cache.keys()][-1])
    # print(params.__repr__())
    
    if params._optimal_concerns() in cache: 
        if cache[params._optimal_concerns()][1] is not None: # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]
            return (cache[params._optimal_concerns()][0], np.array(cache[params._optimal_concerns()][1])) # pyright: ignore[reportReturnType, reportIndexIssue, reportCallIssue, reportArgumentType, reportOperatorIssue]
    params = copy.deepcopy(params)
    params.allowed_to_switch = False
    assert params.is_singleton, "cant find the optimal for a sweep"
    num_robots = params.n_robots
    num_segments = params.n_tasks + 1 # pyright: ignore[reportOperatorIssue] the assertion catches it
    
    
    best_delivery_count = 0
    best_allocation = None
    for possible_assignment in tqdm.tqdm(assignments(num_robots, num_segments)):
        params.robot_initial_placements = np.array(possible_assignment)
        model = params.create_swarm_model()
        model.run_for(3600)
        deliveries = model.delivery_log[-1]
        if deliveries > best_delivery_count:
            best_delivery_count = deliveries
            best_allocation = np.array(possible_assignment)
    
    cache[params._optimal_concerns()] = (best_delivery_count, best_allocation) # type: ignore
    with open("optimal.json", "w+") as f:
        f.write(jsonpickle.encode(cache)) # type: ignore
    
    return best_delivery_count, best_allocation # pyright: ignore[reportReturnType]
        

def run_and_save(params: ModelParams, filename: str, number_process:int = 1, itterations_per_combo:int=100, seeds: Optional[np.ndarray] = None, ):
    if seeds is None:
        seeds = np.random.randint(0, 100000, itterations_per_combo).tolist()
    results = mesa.batch_run(
        SwarmModel,
        number_processes=number_process,
        parameters=params.create_param_dict(),
        max_steps=3600,
        display_progress=True,
        rng=seeds #pyright: ignore[reportArgumentType] THIS IS LITERALLY WHAT THE DOCS SAY TO DO https://mesa.readthedocs.io/latest/migration_guide.html#batch-run
    )
    for test in results:
        test_result=RunResult(**{k:v for k,v in test.items() if "Segment" not in k})
        oc, oa = find_optimal(test_result)
        test_result.optimal_delivery_count = oc
        assert test_result.allocation is not None, "Cleaner than type ignore"
        total = 0.0
        for allocation in test_result.allocation:
            total += np.linalg.norm(allocation-oa)
        test_result.mae = float(total / len(test_result.allocation)) / test_result.n_robots # pyright: ignore[reportOperatorIssue] #its an np floating type not a python float so convert it over
        test_result.save(filename)

def main():
    params = ModelParams()
    params.n_tasks = [2,3,4,5]
    params.n_robots = [20,30,40,50]
    
    
    run_and_save(params, "results.tsv", 5, itterations_per_combo=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, choices=["debug", "info", "warning", "error"], default="error")
    parser.add_argument("-v", dest="log_level", action="store_const", const="info", help="Set log level to info")
    parser.add_argument("-d", dest="log_level", action="store_const", const="debug", help="Set log level to debug")

    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }

    args = parser.parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[{asctime}] [{levelname:<7}] [{name}] : {message}", datefmt="%y/%m/%d-%H:%M:%S", style="{"))
    logger.addHandler(handler)
    logger.setLevel(log_level_map[args.log_level])
    
    main()
