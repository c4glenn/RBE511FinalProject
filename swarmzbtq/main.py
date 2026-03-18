import logging
import argparse
from typing import Optional

import numpy as np

from swarmzbtq.generation_tools import ProblemBuilder

logger = logging.getLogger("swarmzbtq")


def main(seed: Optional[int] = None):
    if not seed: seed = np.random.randint(0, 1000000)
    logger.info(f"{seed=}") 
    rng = np.random.default_rng(seed)
    
    problem = ProblemBuilder(20).add_stage(30).add_stage(30).build()
    
    throughput = evaluate_T([10, 10], problem)
    print(throughput)


if __name__ == "__main__":
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, choices=log_level_map.keys(), default="error")
    parser.add_argument("-v", dest="log_level", action="store_const", const="info", help="Set log level to info")
    parser.add_argument("-d", dest="log_level", action="store_const", const="debug", help="Set log level to debug")
    
    parser.add_argument("--seed", type=int, default=None)
    
    
    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[{asctime}] [{levelname:<7}] [{name}] : {message}", datefmt="%y/%m/%d-%H:%M:%S", style="{"))
    logger.addHandler(handler)
    logger.setLevel(log_level_map[args.log_level])
    
    main()
