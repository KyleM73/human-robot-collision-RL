from os import path

import numpy as np
import datetime

from human_robot_collision_RL.script.network import Network as CustomFeatureNetwork

## Universal constants ##

PI = 3.14159265359
GRAVITY = (0, 0, -9.8)
FIELD_RANGE = 10 #may expand (to 20?)

## Simulation params ##

EXP_NUM = 3 # no human, simple human, full human -------------------- #see slides, exps 1,2,3 = simple_nav,human_nav,human_nav_w_RGBD
EXP_NAME = "Empty_Hall_With_Vision"
CPU_NUM = 10 # machine dependent
TRAIN_STEPS = 1_000_000
POLICY_KWARGS = dict(features_extractor_class=CustomFeatureNetwork,net_arch=[128,64, dict(vf=[], pi=[])]) #vf and pi are layers not shared and unique to the value function and policy, respectively
MAX_ACTION_DIFF = 0.5

DT = datetime.datetime.now().strftime('%m%d_%H%M')
MIN_TARGET_DIST = 4
ENV_IDS = ['simple-v0','human-v0','safety-v0'] # env ids are registered in outermost __init__.py


## Simulation configuartion ##

TIME_STEP = 0.001
REPEAT_INIT = 1000 #repeats initalization in RLenv REPEAT_INIT times, see RLenv.($CLASS)._steup()
REPEAT_ACTION = 100 #repeats action command REPEAT_ACTION times before updating with new action
MAX_STEPS = 2000 #MAX_STEPS*TIME_STEP = wall time of simulation #TODO: currently MAX_STEPS*TIME_STEP=2s, should probably be 30 irl with real max velocities - need real robot params 
MAX_HEIGHT_DEVIATION = 0.1

## System configuration ##

PATH_SCRIPT = path.dirname(path.realpath(__file__))
PATH_ROOT   = path.dirname(PATH_SCRIPT)
PATH_CONFIG = PATH_SCRIPT+"/config"
PATH_SAVE   = PATH_ROOT+"/OLDsave"
PATH_DATA   = PATH_ROOT+"/data"
PATH_LOG    = PATH_ROOT+"/log"
PATH_TMP    = PATH_ROOT+"/tmp"

## Robot configuration

H_ROBOT = 0.635 #[m]
R_ROBOT = 0.2794 #[m]
M_ROBOT = 4.53592 #[kg]
ROBOT_POSE = [0,0,0.5]
ROBOT_ORI = [0,0,0]
NUM_ACTIONS = 2
ROBOT_MODEL = "/trikey2.urdf"

## Human configuration

NUM_HUMANS = None
HUMAN_POSE = [0,5,1.112] #trial and error, works for man, child not tested
HUMAN_ORI = [PI/2,0,0]
#(POSE,ORI) tuple for standing and laying on back config
#INIT_POSE_LIST = [
#    ([0,0,1.112],[PI/2,0,0]), #standing
#    ([0,0,0.15] ,[PI,PI,0])   #laying on back (PI,0,0 for stomach)
#    ]
M_HUMAN = 75
TEST_POSE = np.array([0,-10,0])

## Goal configuration

GOAL_POSES = [np.array([0,2,0.5]),np.array([0,4,0.5]),np.array([0,6,0.5]),np.array([0,8,0.5]),np.array([0,10,0.5])]
GOAL_THRESH = 0.1

## Controller configuartion ##

DEFAULT_ACTION = [0, 0]#, 0]

## Camera configuration ##

RESOLUTION = 10
CAM_DIST = 10.0
CAM_YAW = 0
CAM_PITCH = -30
CAM_WIDTH = 256##48*RESOLUTION#480
CAM_HEIGHT = 256#48*RESOLUTION#360
CAM_FPS = 60
CAM_FOV = 60
CAM_NEARVAL = 0.01
CAM_FARVAL = 20.0


if __name__=='__main__':
    print(PATH_ROOT)


    