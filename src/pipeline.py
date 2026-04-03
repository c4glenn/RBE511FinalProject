from dataclasses import dataclass, field
import logging

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TaskNode:
    index : int
    entry : np.ndarray
    exit  : np.ndarray
    label : str = ""
    
    def __post_init__(self):
        if not self.label: self.label = f"T{self.index}"
        
@dataclass
class Pipeline:
    n_tasks       : int   = 2
    arena_width   : float = 600.0
    arena_height  : float = 200.0
    interface_gap : float = 25.0
    margin        : float = 60.0
    
    source_pos : np.ndarray = field(init=False)
    nest_pos   : np.ndarray = field(init=False)
    tasks      : list       = field(init=False)
    
    def __post_init__(self):
        h = self.arena_height / 2
        self.source_pos = np.array([self.margin, h])
        self.nest_pos = np.array([self.arena_width - self.margin, h])
        
        if self.n_tasks == 0:
            self.tasks = []
            return
        
        n = self.n_tasks
        corridor = self.arena_width - 2 * self.margin
        slot = corridor / (n+1)
        centers_x = [self.margin + slot * (i+1) for i in range(n)]
        self.tasks = [
            TaskNode(index = i, 
                     entry = np.array([cx - self.interface_gap / 2, h]), 
                     exit  = np.array([cx + self.interface_gap / 2, h]))
            for i, cx in enumerate(centers_x)
        ]
    
    def left_end(self, seg:int) -> np.ndarray:
        if seg == 0:
            return self.source_pos.copy()
        return self.tasks[seg-1].exit.copy()
    
    def right_end(self, seg:int) -> np.ndarray:
        if seg == self.n_tasks:
            return self.nest_pos.copy()
        return self.tasks[seg].entry.copy()
    
    def crossing_exit(self, seg: int) -> np.ndarray:
        """Where a robot lands after crossing interface `seg` (= left_end of seg+1)."""
        if seg >= self.n_tasks:
            raise ValueError(f"Segment {seg} has no interface to cross (max {self.n_tasks - 1})")
        return self.tasks[seg].exit.copy()
    
    @property
    def n_segments(self) -> int:
        return self.n_tasks + 1
    
    def describe(self):
        print(f"Pipeline: {self.n_tasks} task(s), {self.n_segments} segment(s)")
        print(f"  SOURCE @ x={self.source_pos[0]:.1f}")
        for t in self.tasks:
            print(f"  {t.label}   entry @ x={t.entry[0]:.1f}  exit @ x={t.exit[0]:.1f}")
        print(f"  NEST   @ x={self.nest_pos[0]:.1f}")

if __name__ == "__main__":
    pipeline = Pipeline(1)
    
    pipeline.describe()
