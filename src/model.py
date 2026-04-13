from typing import Any, Optional

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from pipeline import Pipeline
from agents import RobotAgent, State

class SwarmModel(Model):
    def __init__(self, n_robots:int = 20, n_tasks:int=1, speed:float = 3.0, seed: int = 42, arena_width:float= 600.0, arena_height:float = 200.0, interface_gap:float = 25.0, task_dist_calc: int = 50, task_distribution: Optional[np.ndarray] = None) -> None:
        """you have to pass in a task distribution if n_tasks > 1"""
        super().__init__(rng=seed)
            
        self.pipeline = Pipeline(
            n_tasks       = n_tasks,
            arena_width   = arena_width,
            arena_height  = arena_height,
            interface_gap = interface_gap,
            task_distribution = np.array([task_dist_calc, 100 - task_dist_calc]) if not task_distribution.any() else task_distribution  # type: ignore
        )
        
        self.n_tasks = n_tasks
        
        self.total_deliveries = 0
        self.delivery_log = [0]
        
        spacing = 0.9 * arena_height / n_robots
        
        n_segs = self.pipeline.n_segments
        for i in range(n_robots):
            seg = i % n_segs
            jitter = self.rng.uniform(-20, 20, size=1)
            robot = RobotAgent(model=self, segment=seg, speed=speed)
            robot.pos = np.array([self.pipeline.left_end(seg)[0] + 0.5 * (self.pipeline.right_end(seg)[0] - self.pipeline.left_end(seg)[0]) + jitter[0], spacing * (i+1)])
        
        segment_reporters = {
            f"Segment {s}": (lambda m, s=s: sum(1 for a in m.agents if a.segment == s))
            for s in range(n_segs)
        }
        self.datacollector = DataCollector(
                model_reporters={
                    "Total Deliveries": lambda m: m.total_deliveries,
                    "Throughput":       self._throughput,
                    "Waiting":          lambda m: sum(1 for a in m.agents if a.state == State.WAITING),
                    "Crossing":         lambda m: sum(1 for a in m.agents if a.state == State.CROSSING),
                    **segment_reporters,
                },
                agent_reporters={
                    "State":   lambda a: a.state.name,
                    "Segment": "segment",
                    "X":       lambda a: float(a.pos[0]),
                    "Y":       lambda a: float(a.pos[1]),
                    "Observed Delay": lambda a: float(a.observed_delay)
                }
            )

    def _throughput(self, _=None):
        log = self.delivery_log
        if len(log) < 2:
            return 0.0
        window = log[-50:]
        return (window[-1] - window[0]) / max(len(window) - 1, 1)
 
    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")
        self.delivery_log.append(self.total_deliveries)
