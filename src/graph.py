import pandas as pd
import argparse
import matplotlib.pyplot as plt
from benchmark import RunResult

column_names = [k for k,v in RunResult().__dict__.items() if k != "allocation"]

def load_file(filename: str):
    df = pd.read_csv(filename, sep="\t", names=column_names)
    print(column_names)
    return df

"""['n_robots', 'n_tasks', 'speed', 'seed', 'arena_width', 'arena_height', 'interface_gap', 'task_dist_calc', 'task_distribution', 'robot_initial_placements', 'allowed_to_switch', 'gamma', 'k', 'm', 'switching_cost', 'delay_random_range', 'transfer_time', 'pickup_time', 'dropoff_time', 'RunId', 'iteration', 'Step', 'total_deliveries', 'total_crossings', 'throughput', 'optimal_delivery_count', 'mae', 'creation_time']"""

def example(filename: str):
    df = load_file(filename)
    print(df)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="results.tsv")
    parser.add_argument("--mode", type=str, choices=["example"], default="example")
    
    
    args = parser.parse_args()
    
    match args.mode:
        case "example":
            example(args.filename)
    