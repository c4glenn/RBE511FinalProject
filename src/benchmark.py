#TODO: Move this file out and make it a library so linking works correctly
from typing import Optional, Tuple, List
import logging
import argparse
from dataclasses import dataclass
import copy
import numpy as np
import tqdm
import jsonpickle

from model import SwarmModel

logger = logging.getLogger()

@dataclass
class ModelParams():
    n_robots:int = 20
    n_tasks:int=1
    speed:float = 3.0
    seed: int = 42
    arena_width:float= 600.0
    arena_height:float = 200.0
    interface_gap:float = 25.0
    task_dist_calc: int = 50
    task_distribution: Optional[np.ndarray] = None
    robot_initial_placements: Optional[np.ndarray] = None
    allowed_to_switch:bool = True
    gamma: float = 0.1
    k:float = 5
    m:float = 8
    switching_cost: int = 30
    delay_random_range: Tuple[float, float] = (0.0, 10.0)
    transfer_time: int = 3
    pickup_time: int = 2
    dropoff_time: int = 26
    
    def create_swarm_model(self) -> SwarmModel:
        return SwarmModel(
            n_robots=self.n_robots,
            n_tasks=self.n_tasks,
            speed=self.speed,
            seed=self.seed,
            arena_width=self.arena_width,
            arena_height=self.arena_height,
            interface_gap=self.interface_gap,
            task_dist_calc=self.task_dist_calc,
            task_distribution=self.task_distribution,
            robot_initial_placements=self.robot_initial_placements,
            allowed_to_switch=self.allowed_to_switch,
            gamma=self.gamma,
            k=self.k,
            m=self.m,
            switching_cost=self.switching_cost,
            delay_random_range=self.delay_random_range,
            transfer_time=self.transfer_time,
            pickup_time=self.pickup_time,
            dropoff_time=self.dropoff_time
        )

    def __repr__(self) -> str:
        return f"ModelParams({self.n_robots=},{self.n_tasks=}{self.speed=},{self.arena_width=},{self.arena_height=},{self.interface_gap=},{self.task_dist_calc=},{self.task_distribution=},{self.delay_random_range=},{self.transfer_time=},{self.pickup_time=},{self.dropoff_time=}"

    def __hash__(self) -> int:
        return hash(f"{self.n_robots}{self.n_tasks}{self.speed}{self.arena_width}{self.arena_height}{self.interface_gap}{self.task_dist_calc}{self.task_distribution}{self.robot_initial_placements}{self.delay_random_range}{self.transfer_time}{self.pickup_time}{self.dropoff_time}")

def assignments(num_robots, num_tasks, depth=0):
    if num_robots == 0: raise ValueError("wtf")
    if num_tasks == 1: 
        return [num_robots]
    options = []
    for i in range(1, num_robots-num_tasks):
        possible_vals = assignments(num_robots-i, num_tasks-1, depth+1)
        for val in possible_vals:
            try:
                options.append([i, *val])
            except:
                options.append([i, val])
    return options

def find_optimal(params:ModelParams) -> Tuple[int, List[int]]:
    with open("optimal.json", "r+") as f:
        cache = jsonpickle.decode(f.read())
    # print([x for x in cache.keys()][-1])
    # print(params.__repr__())
    
    if params.__repr__() in cache: return cache[params.__repr__()] # pyright: ignore[reportReturnType, reportIndexIssue, reportCallIssue, reportArgumentType, reportOperatorIssue]
    
    params = copy.deepcopy(params)
    params.allowed_to_switch = False
    num_robots = params.n_robots
    num_segments = params.n_tasks + 1
    
    
    best_delivery_count = 0
    best_allocation = None
    for possible_assignment in tqdm.tqdm(assignments(num_robots, num_segments)):
        params.robot_initial_placements = np.array(possible_assignment)
        model = params.create_swarm_model()
        model.run_for(1000)
        deliveries = model.delivery_log[-1]
        if deliveries > best_delivery_count:
            best_delivery_count = deliveries
            best_allocation = possible_assignment
    
    cache[params] = (best_delivery_count, best_allocation) # type: ignore
    with open("optimal.json", "r+") as f:
        f.write(jsonpickle.encode(cache)) # type: ignore
    
    return best_delivery_count, best_allocation # pyright: ignore[reportReturnType]
        

def main():
    params = ModelParams(task_dist_calc=70)
    params.seed = int(np.random.randint(0, 100000))
    dc, ba = find_optimal(params)
    model = params.create_swarm_model()
    model.run_for(1000)
    deliveries = model.delivery_log[-1]
    print(f"this trials efficiency was {deliveries/dc:.2%}")


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
