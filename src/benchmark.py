#TODO: Move this file out and make it a library so linking works correctly

import logging
import argparse

from model import SwarmModel

logger = logging.getLogger()

def main():
    swarm = SwarmModel(2, 1, 40)
    swarm.run_for(20)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, choices=["debug", "info", "warning", "error"], default="error")
    parser.add_argument("-v", dest="log_level", action="store_const", const="info", help="Set log level to info")
    parser.add_argument("-d", dest="log_level", action="store_const", const="debug", help="Set log level to debug")

    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }

    args = parser.parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[{asctime}] [{levelname:<7}] [{name}] : {message}", datefmt="%y/%m/%d-%H:%M:%S", style="{"))
    logger.addHandler(handler)
    logger.setLevel(log_level_map[args.log_level])
    
    main()
