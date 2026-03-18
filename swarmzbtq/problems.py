import logging

import numpy as np

logger = logging.getLogger(__name__)



class Problem():
    def __init__(self, num_robots: int, num_stages: int, service_rate: np.ndarray, time_unit: str = "sec") -> None:
        """Create a problem instance

        Args:
            num_robots (int): number of robots
            num_stages (int): number of coupled stages
            service_rate (np.ndarray): this is a frequency, = 1/average completion time
            time_unit (str): doesn't do anything functional, but allows for unit printing when a number is printed out
        """
        assert service_rate.shape == (num_stages,), "shape of service rate array should be 1 per stage"
        self.num_robots = num_robots
        self.num_stages = num_stages
        self.service_rate = service_rate
        self.time_unit = time_unit