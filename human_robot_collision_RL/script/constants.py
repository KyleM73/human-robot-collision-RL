from os import path

import numpy as np
import datetime

## Universal constants ##

PI = 3.14159265359
GRAVITY = (0, 0, -9.8)
FIELD_RANGE = 10 #may expand (to 20?)

## Simulation params ##

EXP_NUM = 2 #see slides, exps 1,2,3 = simple_nav,human_nav,human_nav_w_RGBD
CPU_NUM = 24 # machine dependent
ENV_IDS = ['simple-v0','human-v0','safety-v0'] # env ids are registered in outermost __init__.py
TRAIN_STEPS = 5_000_000
DT = datetime.datetime.now().strftime('%m%d_%H%M')
MIN_TARGET_DIST = 4
POLICY_KWARGS = dict(net_arch=[64, 64, dict(vf=[], pi=[])]) #vf and pi are layers not shared and unique to the value function and policy, respectively


## Simulation configuartion ##

TIME_STEP = 0.001
REPEAT_INIT = 100 #repeats initalization in RLenv REPEAT_INIT times, see RLenv.($CLASS)._steup()
REPEAT_ACTION = 10 #repeats action command REPEAT_ACTION times before updating with new action
MAX_STEPS = 2000 #MAX_STEPS*TIME_STEP = wall time of simulation #TODO: currently MAX_STEPS*TIME_STEP=2s, should probably be 30 irl with real max velocities - need real robot params 
MAX_HEIGHT_DEVIATION = 0.1

## System configuration ##

PATH_SCRIPT = path.dirname(path.realpath(__file__))
PATH_ROOT   = path.dirname(PATH_SCRIPT)
PATH_CONFIG = PATH_SCRIPT+"/config"
PATH_SAVE = PATH_ROOT+"/save"
PATH_DATA   = PATH_ROOT+"/data"
PATH_LOG   = PATH_ROOT+"/log"

## Robot configuration

H_ROBOT = 0.635 #[m]
R_ROBOT = 0.2794 #[m]
M_ROBOT = 4.53592 #[kg]
ROBOT_POSE = [0,0,0.5]
ROBOT_ORI = [0,0,0]
NUM_ACTIONS = 3
ROBOT_MODEL = "/trikey2.urdf"

## Human configuration

NUM_HUMANS = 1
POSE = [0,0,1.112] #trial and error, works for man, child not tested
ORI = [PI/2,0,PI]
#(POSE,ORI) tuple for standing and laying on back config
INIT_POSE_LIST = [
    ([0,0,1.112],[PI/2,0,0]), #standing
    ([0,0,0.15] ,[PI,PI,0])   #laying on back (PI,0,0 for stomach)
    ]
M_HUMAN = 75
TEST_POSE = np.array([0,-5,0])

## Controller configuartion ##

DEFAULT_ACTION = [0, 0, 0]

## Camera configuration ##

CAM_DIST = 10.0
CAM_YAW = 0
CAM_PITCH = -30
CAM_WIDTH = 480
CAM_HEIGHT = 360
CAM_FPS = 60
CAM_FOV = 60
CAM_NEARVAL = 0.1
CAM_FARVAL = 100.0


if __name__=='__main__':
    print(PATH_ROOT)


    