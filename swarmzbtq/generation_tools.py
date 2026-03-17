from __future__ import annotations
import logging

import numpy as np

from swarmzbtq.problems import Problem

logger = logging.getLogger(__name__)


class ProblemBuilder():
    def __init__(self, num_robots: int, time_unit: str = "sec") -> None:
        self.num_robots = num_robots
        self.stages = []
        self.num_stages = 0
        self.time_unit = time_unit
    
    def add_stage(self, average_task_duration: float) -> ProblemBuilder:
        self.stages.append(1/average_task_duration)
        self.num_stages += 1
        return self
        
    def build(self) -> Problem:
        return Problem(self.num_robots, self.num_stages, np.array(self.stages), self.time_unit)
        