from os import path
import numpy as np

## Universal constants ##

PI = 3.14159265359
GRAVITY = (0, 0, -9.8)
FIELD_RANGE = 10 #need to expand, likely to 20

## Simulation configuartion ##

TIME_STEP = 0.001
REPEAT_INIT = 100 #repeats initalization in RLenv REPEAT_INIT times, see RLenv.($CLASS)._steup()
REPEAT_ACTION = 10 #repeats action command REPEAT_ACTION times before updating with new action
MAX_STEPS = 2000 #MAX_STEPS*TIME_STEP = wall time of simulation 
TRAIN_STEPS = 5_000_000#5_000_000
#TODO: currently MAX_STEPS*TIME_STEP=2s, should probably be 30 irl with real max velocities - need real robot params


## System configuration ##

PATH_SCRIPT = path.dirname(path.realpath(__file__))
PATH_ROOT   = path.dirname(PATH_SCRIPT)
PATH_CONFIG = PATH_SCRIPT+"/config"
PATH_SAVE = PATH_ROOT+"/save"
PATH_DATA   = PATH_ROOT+"/data"
PATH_LOG   = PATH_ROOT+"/log"

## Robot configuration

NUM_ACTIONS = 3
NUM_COMMANDS = 3
H_ROBOT = 0.635 #[m]
R_ROBOT = 0.2794 #[m]
M_ROBOT = 4.53592 #[kg]

## Human configuration

POSE = [0,0,0.25]#[0,0,1.112] #[0,0,0.25] #trial and error, works for man, child not tested
ORI = [PI,0,PI]#[PI/2,0,PI] #[PI,0,PI]
#(POSE,ORI) tuple for standing and laying on back config
INIT_POSE_LIST = [
    ([0,0,1.112],[PI/2,0,0]), #standing
    ([0,0,0.15] ,[PI,PI,0])   #laying on back (PI,0,0 for stomach)
    ]
M_HUMAN = 75
TEST_POSE = np.array([0,-5,0])

## Controller configuartion ##

DEFAULT_ACTION = (0, 0, 0)


if __name__=='__main__':
    print(PATH_ROOT)


    