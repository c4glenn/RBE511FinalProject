from typing import Any
from enum import Enum, auto
import logging

import numpy as np
from mesa import Agent

logger = logging.getLogger(__name__)

class State(Enum):
    MOVING_TO_RIGHT = auto()   # carrying object toward right_end
    WAITING         = auto()   # at interface entry, deciding
    CROSSING        = auto()   # traversing the interface gap
    MOVING_TO_LEFT  = auto()   # returning to left_end
    ACTING          = auto()   # pick-up or deposit (timed)

STATE_COLORS = {
    State.MOVING_TO_RIGHT: "#4FC3F7",
    State.WAITING:         "#FF8A65",
    State.CROSSING:        "#E91E63",
    State.MOVING_TO_LEFT:  "#A5D6A7",
    State.ACTING:          "#FFF176",
}

class RobotAgent(Agent):
    def __init__(self, model: Any, segment=0, speed=3.0) -> None:
        super().__init__(model)
        self.segment = segment
        self.speed = max(speed + model.rng.uniform(-0.5, 0.5), 0.5)
    
        self.state = State.MOVING_TO_LEFT
        self.pos = model.pipeline.left_end(self.segment)
        self.has_object = False
        self.wait_timer = 0
        self._act_is_pickup = True
        self.observed_delay = 0
        
        self.deliveries = 0
        self.crossings_done = 0
        
        self.step_map = {
            State.ACTING: self._step_acting,
            State.MOVING_TO_LEFT: self._step_moving_left,
            State.MOVING_TO_RIGHT: self._step_moving_right,
            State.WAITING: self._step_waiting,
            State.CROSSING: self._step_crossing
        }
    
    def _move_toward(self, target: np.ndarray) -> bool:
        delta = target - self.pos
        dist = np.linalg.norm(delta)
        if dist <= self.speed:
            self.pos = target
            return True
        self.pos += (delta/dist) * self.speed # pyright: ignore[reportOperatorIssue]
        return False
    
    def step(self):
        pipe = self.model.pipeline
        logger.info(f"Agent: {self.unique_id} is {self.state} {self.pos} {self.has_object} {pipe.right_end(self.segment)}")
        if self.state != State.WAITING and self.wait_timer != 0:
            self.observed_delay = np.average([self.wait_timer, self.observed_delay])
            self.wait_timer = 0
        self.step_map[self.state](pipe)

    def _step_acting(self, pipe):
        if self.segment == 0:
            self.has_object = True
            self.state = State.MOVING_TO_RIGHT
        if self.segment >= len(pipe.tasks):
            self.has_object = False
            self.model.total_deliveries += 1
            self.state = State.MOVING_TO_LEFT
    
    def _step_moving_left(self, pipe):
        arrived = self._move_toward(np.array([pipe.left_end(self.segment)[0], self.pos[1]])) # pyright: ignore[reportOptionalSubscript, reportIndexIssue]
        if not arrived:
            return
        if self.segment == 0: #leftmost segment
            self.state = State.ACTING
        else:
            self.state = State.WAITING # waiting on an agent to get here
    def _step_moving_right(self, pipe):
        arrived = self._move_toward(np.array([pipe.right_end(self.segment)[0], self.pos[1]])) # pyright: ignore[reportOptionalSubscript, reportIndexIssue]
        if not arrived:
            return
        if self.segment >= len(pipe.tasks): #rightmost segment
            self.state = State.ACTING
        else:
            self.state = State.WAITING # waiting on an agent to get here

    def _step_waiting(self, pipe):
        self.wait_timer += 1
        if self.has_object: #waiting at the right side to do a handoff
            next_agents = [x for x in self.model.agents if x.segment == self.segment + 1 and x.state == State.WAITING]
            if len(next_agents) > 0:
                next_agent = next_agents[0]
                next_agent.has_object = True
                self.has_object = False
                next_agent.state = State.MOVING_TO_RIGHT
                self.state = State.MOVING_TO_LEFT
        else: #waiting at the left side to pick up an object (let a robot on the other side handle the logic)
            pass
    
    def _step_crossing(self, pipe):
        pass
    
    @property
    def color(self):
        return STATE_COLORS[self.state]
