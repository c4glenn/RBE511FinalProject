import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import argparse
import matplotlib.pyplot as plt
from benchmark import RunResult

column_names = [k for k,v in RunResult().__dict__.items() if k != "allocation"]

def load_file(filename: str):
    df = pd.read_csv(filename, sep="\t", names=column_names)
    df = df[df["n_tasks"] == 3]
    print(column_names)
    return df

"""['n_robots', 'n_tasks', 'speed', 'seed', 'arena_width', 'arena_height', 'interface_gap', 'task_dist_calc', 'task_distribution', 'robot_initial_placements', 'allowed_to_switch', 'gamma', 'k', 'm', 'switching_cost', 'delay_random_range', 'transfer_time', 'pickup_time', 'dropoff_time', 'RunId', 'iteration', 'Step', 'total_deliveries', 'total_crossings', 'throughput', 'optimal_delivery_count', 'mae', 'creation_time']"""

def example(filename: str):
    df = load_file(filename)
    print(df)

def plot_graph(df: pd.DataFrame, start_row: int = 0, end_row: int = -1, sort_by: str | None = None, filter_num: List[int] | None = None, group_by: str | None = None):
    if end_row == -1:
        end_row = len(df)
    df_slice = df.iloc[start_row:end_row]

    if filter_num is not None:
        if sort_by is None:
            raise ValueError("filter_num requires sort_by to be set")
        if sort_by not in df_slice.columns:
            raise ValueError(f"sort_by column '{sort_by}' not found in dataframe")
        df_slice = df_slice[df_slice[f"{sort_by}"].isin(filter_num)]

    if sort_by is not None:
        if sort_by not in df_slice.columns:
            raise ValueError(f"sort_by column '{sort_by}' not found in dataframe")
        df_slice = df_slice.sort_values(by=sort_by)
        print(df_slice)
    
    plt.figure(figsize=(10, 6))

    
    if group_by is not None:
        if group_by not in df_slice.columns:
            raise ValueError(f"group_by column '{group_by}' not found in dataframe")
        groups = df_slice.groupby(group_by)
        colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax = plt.gca())
        cbar.set_label(f"{group_by}")
        for i, (name, group) in enumerate(groups):
            plt.scatter(group[f"{args.xlabel}"], group[f"{args.ylabel}"], marker='o', color=colors[i], label=f"{group_by}={name}")
        # plt.legend()
    else:
        plt.plot(df_slice[f"{args.xlabel}"], df_slice[f"{args.ylabel}"], marker='o')
    
    title = f"{args.ylabel} vs {args.xlabel}"
    if sort_by is not None:
        title += f" (sorted by {sort_by})"
    if group_by is not None:
        title += f" (grouped by {group_by})"
    
    plt.title(title)
    plt.xlabel(f"{args.xlabel}")
    plt.ylabel(f"{args.ylabel}")
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="results.tsv")
    parser.add_argument("--xlabel", type=str, default="iteration")
    parser.add_argument("--ylabel", type=str, default="throughput")
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--end_row", type=int, default=-1)
    parser.add_argument("--sort_by", type=str, default=None)
    parser.add_argument("--filter_num", type=int, nargs='+', default=None,
                        help="One or more values to filter the sort_by column")
    parser.add_argument("--group_by", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["example", "load_file", "plot_graph"], default="example")
    
    # tasks = [3]
    # n_robots = [4, 8, 12, 16, 20, 24, 28, 32]
    
    # args = parser.parse_args(["--filename", "results.tsv", "--mode", "load_file"])
    args = parser.parse_args(["--filename", "results.tsv", "--xlabel", "n_robots", "--ylabel", "mae", "--mode", "plot_graph", "--sort_by", "n_robots", "--filter_num", "4", "8", "12", "16", "20", "24", "28", "32", "--group_by", "total_deliveries"])
    
    match args.mode:
        case "example":
            example(args.filename)
        case "load_file":
            load_file(args.filename)
        case "plot_graph":
            df = load_file(args.filename)
            plot_graph(df, args.start_row, args.end_row, args.sort_by, args.filter_num, args.group_by)
        
# Bad Test Data:
# 20	3	3.0	39205	600.0	200.0	25.0	50	[10 20 30 40]	None	True	0.1	5	8	30	(0.0, 10.0)	3	1	1	0	0	3600.0	163	274	0.08163265306122448	105	0.20787495519032198	2026-04-23 12:12:46.395372
# 20	3	3.0	39205	600.0	200.0	25.0	50	[40 30 20 10]	None	True	0.1	5	8	30	(0.0, 10.0)	3	1	1	2	0	3600.0	159	275	0.04081632653061224	124	0.22948188726729996	2026-04-23 12:12:46.425790
# 20	3	3.0	39205	600.0	200.0	25.0	50	[15 35 35 15]	None	True	0.1	5	8	30	(0.0, 10.0)	3	1	1	3	0	3600.0	162	268	0.04081632653061224	148	0.21100701497876062	2026-04-23 12:14:48.508152
# 20	3	3.0	39205	600.0	200.0	25.0	50	[35 15 15 35]	None	True	0.1	5	8	30	(0.0, 10.0)	3	1	1	4	0	3600.0	153	303	0.061224489795918366	157	0.2920151903174631	2026-04-23 12:18:25.130361
# 20	3	3.0	39205	600.0	200.0	25.0	50	[25 25 25 25]	None	True	0.1	5	8	30	(0.0, 10.0)	3	1	1	1	0	3600.0	159	275	0.02040816326530612	183	0.21186722811079597	2026-04-23 12:21:54.420419