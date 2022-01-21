from os import path


## Simulation configuartion ##

TIME_STEP = 0.001
REPEAT_INIT = 100
REPEAT_ACTION = 10


## System configuration ##

PATH_SCRIPT = path.dirname(path.realpath(__file__))
PATH_ROOT   = path.dirname(PATH_SCRIPT)
PATH_CONFIG = PATH_ROOT+"/config"
PATH_SAVE = PATH_ROOT+"/save"
PATH_DATA   = PATH_ROOT+"/data"
PATH_PLOT   = PATH_ROOT+"/plot"

SUBPATH_CONFIG = {  "reward":   "reward.yaml",
                    "ppo":      "ppo.yaml",
                    "experiment": "experiment.yaml"}

## Robot configuration

NUM_ACTIONS = 3
NUM_COMMANDS = 3


## Universal constants ##

PI = 3.14159265359
GRAVITY = (0, 0, -9.8)
FIELD_RANGE = 10


## Controller configuartion ##

DEFAULT_ACTION = (0, 0, 0)