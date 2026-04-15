from typing import Any, Tuple
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
    def __init__(self, model: Any, segment=0, speed=3.0, allowed_to_switch:bool = True, gamma: float = 0.1, k:float = 5, m: float=8, switching_cost: int = 30, delay_random_range: Tuple[float, float] = (0.0, 10.0), transfer_time: int = 3, pickup_time: int = 2, dropoff_time: int = 2) -> None:
        super().__init__(model)
        self.segment = segment
        self.speed = max(speed + model.rng.uniform(-0.5, 0.5), 0.5)
    
        self.state = State.MOVING_TO_LEFT
        self.pos = model.pipeline.left_end(self.segment)
        self.has_object = False
        self.paired = None
        self.wait_timer = 0
        self.timer = 0
        self._act_is_pickup = True
        self.observed_delay = 0
        
        self.crossing_time = -1
        
        delays_list = [self.model.rng.uniform(*delay_random_range) for _ in range(model.n_tasks + 1)]
        self.observed_delays = np.array(delays_list)
        
        
        self.deliveries = 0
        self.crossings_done = 0
        self.allowed_to_switch = allowed_to_switch
        self._allowed_to_switch = False
        self.gamma = gamma 
        self.k = k
        self.m = m
        self.switching_cost = switching_cost
        self.transfer_time = transfer_time
        self.pickup_time = pickup_time
        self.dropoff_time = dropoff_time
        
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
        if self.state != State.WAITING and self.wait_timer != -1:
            self.observed_delay = np.average([self.wait_timer, self.observed_delay])
            self.observed_delays[self.segment] = self.observed_delay
            logger.debug(self.unique_id, self.observed_delays)
            self._allowed_to_switch = self.allowed_to_switch
            self.wait_timer = -1
        self.step_map[self.state](pipe)

    def _step_acting(self, pipe):
        if self.segment == 0:
            if self.timer == -1: self.timer = self.pickup_time
            if self.timer <= 0:
                self.has_object = True
                self.state = State.MOVING_TO_RIGHT
                self.timer = -1
            else:
                self.timer -= 1
        if self.segment >= len(pipe.tasks):
            if self.timer == -1: self.timer = self.dropoff_time
            if self.timer <= 0:
                self.has_object = False
                self.model.total_deliveries += 1
                self.state = State.MOVING_TO_LEFT
                self.timer = -1          
            else:
                self.timer -= 1

            
    
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
        if self.wait_timer == -1: self.wait_timer = 0
        self.wait_timer += 1
        if self.has_object: #waiting at the right side to do a handoff
            if self.paired != None and self.timer <= 0:
                self.paired.has_object = True
                self.has_object = False
                logging.debug(f"{self.unique_id} passed an object to {self.paired.unique_id}")
                self.paired.state = State.MOVING_TO_RIGHT
                self.state = State.MOVING_TO_LEFT
                self.paired.paired = None
                self.paired = None
                self.timer = -1
                return #handoff happend - dont switch
            elif self.paired != None:
                self.timer -= 1
                return #handoff about to happen - dont switch
            else:
                next_agents = [x for x in self.model.agents if x.segment == self.segment + 1 and x.state == State.WAITING and x.has_object == False and x.paired is None]
                if len(next_agents) > 0:
                    next_agent = self.model.rng.choice(next_agents)
                    self.paired = next_agent
                    next_agent.paired = self
                    self.timer = self.transfer_time
                    return # paired off for handoff 
        else: #waiting at the left side to pick up an object (let a robot on the other side handle the logic)
            if self.paired: return # someone has picked to hand an object, dont switch
            
        if self._allowed_to_switch:
            other_delay = self.observed_delays[self.segment + (1 if self.has_object else -1)] 
            prob = self.switching_probability(self.observed_delay, max(other_delay, 0.01)) #prevent a divide by 0 error by making the delay non-zero
            if self.model.rng.random() < prob:
                self.state = State.CROSSING
                self.wait_timer = 0
                self.crossings_done += 1

    def switching_probability(self, current_state_estimate_delay: float, other_state_estimate_delay: float) -> float:
        theta = self.theta_calc(current_state_estimate_delay, other_state_estimate_delay)
        return 1/(1 + np.exp(-theta)) * self.gamma
    
    def theta_calc(self, current_state_estimate_delay: float, other_state_estimate_delay: float) -> float:
        dij = max(current_state_estimate_delay, 0.0001)
        dji = max(other_state_estimate_delay, 0.0001) 

        return (1/self.k)*((dij / ((dij * max(dij, dji)) / dji)) - self.m)
    
    def _step_crossing(self, pipe):
        if self.crossing_time == -1:
            self.crossing_time = self.switching_cost
            logging.debug(f"{self.unique_id}: switching from {self.segment} by {1 if self.has_object else -1} because {self.has_object}")
            self.segment += 1 if self.has_object else -1
            
        elif self.crossing_time == 0:
            self.crossing_time = -1
            self.state = State.MOVING_TO_RIGHT if self.has_object else State.MOVING_TO_LEFT
            logging.debug(f"{self.unique_id} leaving crossing state - now in {self.state} ")
            return
        
        if self.has_object:
            self._move_toward(np.array([pipe.left_end(self.segment)[0], self.pos[1]])) # pyright: ignore[reportOptionalSubscript, reportIndexIssue]
        else:
            self._move_toward(np.array([pipe.right_end(self.segment)[0], self.pos[1]])) # pyright: ignore[reportOptionalSubscript, reportIndexIssue]

        self.crossing_time -= 1
        
        
    @property
    def color(self):
        return STATE_COLORS[self.state]
