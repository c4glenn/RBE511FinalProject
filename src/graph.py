from typing import Optional
import random
import math
from itertools import product

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import argparse
import matplotlib.pyplot as plt
from benchmark import RunResult

column_names = [k for k,v in RunResult().__dict__.items() if k != "allocation"]

def load_file(filename: str):
    df = pd.read_csv(filename, sep="\t", names=column_names)
    return df


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
    
def violin_categorical(filename: str, category: str, value: str, block_x_name: Optional[str], block_y_name: Optional[str], remove_cats: Optional[List[Any]]):
    df = load_file(filename)
    df["total_deliveries"] = [int(x) for x in df["total_deliveries"]]
    df["optimal_delivery_count"] = [int(x) for x in df["optimal_delivery_count"]]
    df["delivery_ratio"] = df["total_deliveries"] / df["optimal_delivery_count"]
    
    if block_x_name and block_y_name:
        x_options = sorted(df[block_x_name].unique())
        y_options = sorted(df[block_y_name].unique())
        ncols = len(x_options)
        nrows = len(y_options)
    elif block_x_name or block_y_name:
        name = block_x_name if block_x_name else block_y_name
        try:
            x_options = sorted(df[name].unique())
        except:
            x_options = df[name].unique()
        print(x_options)
        y_options = [None]
        length = len(x_options)
        ncols = math.ceil(math.sqrt(length))
        nrows = math.ceil(length / ncols)
    else:
        x_options = [None]
        y_options = [None]
        ncols = 1
        nrows = 1
    
    print(ncols, nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    for x_col, y_col in product(x_options, y_options):
        print(x_col, y_col)
        x_idx = list(x_options).index(x_col)
        y_idx = list(y_options).index(y_col)
        
        if x_col is not None and y_col is not None:
            if remove_cats and str(x_col) in remove_cats: continue
            if remove_cats and str(y_col) in remove_cats: continue
            df_filtered = df[(df[block_x_name] == x_col) & (df[block_y_name] == y_col)].copy()
            ax = axes[y_idx][x_idx]
        elif x_col is not None or y_col is not None:
            name = block_x_name if block_x_name else block_y_name
            col_result = x_col if block_x_name else y_col
            if remove_cats and str(col_result) in remove_cats: continue
            print(col_result, remove_cats)
            if isinstance(col_result, float) and math.isnan(col_result):
                df_filtered = df[df[name].isna()].copy()
            else:
                df_filtered = df[df[name] == col_result].copy()
            ax = axes_flat[x_idx if block_x_name else y_idx]
        else:
            df_filtered = df.copy()
            ax = axes

        if df_filtered.empty:
            print(x_col, "is empty")
            ax.set_visible(False)
            continue
        
        try:
            categories = sorted(df_filtered[category].unique())
        except:
            categories = df_filtered[category].unique()
        n_cats = len(categories)

        if n_cats > 10 and pd.api.types.is_numeric_dtype(df_filtered[category]):
            df_filtered[category] = pd.cut(df_filtered[category], bins=10)
            categories = sorted(df_filtered[category].unique())
            n_cats = len(categories)
        
        grouped_data = []
        for cat in categories:
            if remove_cats and cat in remove_cats: continue
            if isinstance(cat, float) and math.isnan(cat):
                if remove_cats and 'nan' in remove_cats: continue # pyright: ignore[reportOperatorIssue]
                grouped_data.append((df_filtered[df_filtered[category].isna()][value].dropna().values, cat))
            else:
                grouped_data.append((df_filtered[df_filtered[category] == cat][value].dropna().values, cat))
        
        grouped_data = [x for x in grouped_data if not x[0].size == 0]
        names = [x[1] for x in grouped_data]
        grouped_data = [x[0] for x in grouped_data]
        n_cats = len(grouped_data)
        print(n_cats)
        colors = plt.cm.Set2(np.linspace(0, 1, n_cats)) # pyright: ignore[reportAttributeAccessIssue]

        
        parts = ax.violinplot(
            grouped_data,
            positions=range(1, n_cats + 1),
            showmedians=True,
            showextrema=True,
        )

        for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.75)

        for partname in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[partname].set_edgecolor("black")
            parts[partname].set_linewidth(2.4)

        for i, (data, color) in enumerate(zip(grouped_data, colors)):
            jitter = np.random.uniform(-0.1, 0.1, size=len(data)) #for the x so that nearby data doesnt occlude as much
            ax.scatter(
                np.full_like(data, i + 1) + jitter,
                data,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                s=20,
                alpha=0.6,
                zorder=3,
            )

    # Labels and formatting
        ax.set_xticks(range(1, n_cats + 1))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
        ax.set_xlabel(category, fontsize=12)
        ax.set_ylabel(value, fontsize=12)
        ax.set_title(f"{block_x_name}={x_col} : {block_y_name}={y_col}", fontsize=14, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    
def scatter_graph(filename):
    df = load_file(filename)

    df = df[df["task_distribution"].isna()]
    # print(df["n_tasks"])
    df["total_deliveries"] = [int(x) for x in df["total_deliveries"]]
    df["optimal_delivery_count"] = [int(x) for x in df["optimal_delivery_count"]]
    df["delivery_ratio"] = df["total_deliveries"] / df["optimal_delivery_count"]

    plt.scatter(df["n_tasks"], df["total_crossings"] / df["n_robots"])
    plt.show()


    
column_names = ['n_robots', 'n_tasks', 'speed', 'seed', 'arena_width', 'arena_height', 'interface_gap', 'task_dist_calc', 'task_distribution', 'robot_initial_placements', 'allowed_to_switch', 'gamma', 'k', 'm', 'switching_cost', 'delay_random_range', 'transfer_time', 'pickup_time', 'dropoff_time', 'RunId', 'iteration', 'Step', 'total_deliveries', 'total_crossings', 'throughput', 'optimal_delivery_count', 'mae', 'creation_time', "delivery_ratio"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="results.tsv")
    parser.add_argument("--mode", type=str, choices=["example", "load_file", "plot_graph", "violin", "scatter"], default="example")
    
    
    violin_args = parser.add_argument_group("violin args", description="Arguments for the categorical violin graph")
    violin_args.add_argument("--x_block", type=str, default=None, choices=column_names, help="Select a category to split on the x axis")
    violin_args.add_argument("--y_block", type=str, default=None, choices=column_names, help="Select a category to split on the y axis")
    violin_args.add_argument("--category", type=str, default="n_tasks", choices=column_names, help="Select a category for each violin type")
    violin_args.add_argument("--remove_cats", nargs='+')
    violin_args.add_argument("--value",  type=str, default="delivery_ratio", choices=column_names, help="Select a column to use as the y axis for the violins")
    
    
    plot_graph_args = parser.add_argument_group("plot_graph_args")
    plot_graph_args.add_argument("--xlabel", type=str, default="iteration")
    plot_graph_args.add_argument("--ylabel", type=str, default="throughput")
    plot_graph_args.add_argument("--start_row", type=int, default=20)
    plot_graph_args.add_argument("--end_row", type=int, default=35)
    plot_graph_args.add_argument("--sort_by", type=str, default=None)
    plot_graph_args.add_argument("--filter_num", type=int, default=None)
    plot_graph_args.add_argument("--group_by", type=str, default=None)
    
    # tasks = [3]
    # n_robots = [4, 8, 12, 16, 20, 24, 28, 32]
    
    # args = parser.parse_args(["--filename", "results.tsv", "--mode", "load_file"])
    # args = parser.parse_args(["--filename", "results.tsv", "--xlabel", "n_robots", "--ylabel", "mae", "--mode", "plot_graph", "--sort_by", "n_robots", "--filter_num", "4", "8", "12", "16", "20", "24", "28", "32", "--group_by", "total_deliveries"])
    

    args = parser.parse_args()
        
    match args.mode:
        case "example":
            example(args.filename)
        case "load_file":
            load_file(args.filename)
        case "plot_graph":
            df = load_file(args.filename)
            plot_graph(df, args.start_row, args.end_row, args.sort_by, args.filter_num, args.group_by)
        case "violin":
            violin_categorical(filename=args.filename,category=args.category, value=args.value, block_x_name=args.x_block, block_y_name=args.y_block, remove_cats=args.remove_cats)
        case "scatter":
            scatter_graph(args.filename)
# Bad Test Data:
# 20	3	3.0	39205	600.0	200.0	25.0	50	[10 20 30 40]	None	True	0.1	5	8	30	(0.0, 10.0)	3	1	1	0	0	3600.0	163	274	0.08163265306122448	105	0.20787495519032198	2026-04-23 12:12:46.395372
# 20	3	3.0	39205	600.0	200.0	25.0	50	[40 30 20 10]	None	True	0.1	5	8	30	(0.0, 10.0)	3	1	1	2	0	3600.0	159	275	0.04081632653061224	124	0.22948188726729996	2026-04-23 12:12:46.425790
# 20	3	3.0	39205	600.0	200.0	25.0	50	[15 35 35 15]	None	True	0.1	5	8	30	(0.0, 10.0)	3	1	1	3	0	3600.0	162	268	0.04081632653061224	148	0.21100701497876062	2026-04-23 12:14:48.508152
# 20	3	3.0	39205	600.0	200.0	25.0	50	[35 15 15 35]	None	True	0.1	5	8	30	(0.0, 10.0)	3	1	1	4	0	3600.0	153	303	0.061224489795918366	157	0.2920151903174631	2026-04-23 12:18:25.130361
# 20	3	3.0	39205	600.0	200.0	25.0	50	[25 25 25 25]	None	True	0.1	5	8	30	(0.0, 10.0)	3	1	1	1	0	3600.0	159	275	0.02040816326530612	183	0.21186722811079597	2026-04-23 12:21:54.420419