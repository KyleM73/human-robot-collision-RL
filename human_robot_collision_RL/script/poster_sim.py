from os import path

import pybullet as p
from pybullet import getEulerFromQuaternion as Q2E
import pybullet_utils.bullet_client as bc
import pybullet_data

import numpy as np
import datetime
import time

from human_robot_collision_RL.data.man import Man

from human_robot_collision_RL.script.collision import Collision
from human_robot_collision_RL.script.constants import *
from human_robot_collision_RL.script.control import ctlrRobot
from human_robot_collision_RL.script.util import *

from human_robot_collision_RL.script.config.collision_params import *
from human_robot_collision_RL.script.config.rewards import *

client = bc.BulletClient(connection_mode=p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

client.resetSimulation()

client.setAdditionalSearchPath(PATH_DATA)

## Set up simulation
client.setTimeStep(TIME_STEP)
client.setPhysicsEngineParameter(numSolverIterations=int(30))
client.setPhysicsEngineParameter(enableConeFriction=0)
client.setGravity(GRAVITY[0],GRAVITY[1],GRAVITY[2])

PATH_SCRIPT = path.dirname(path.realpath(__file__))
PATH_ROOT   = path.dirname(PATH_SCRIPT)
PATH_DATA   = PATH_ROOT+"/data"
pth = PATH_DATA + ROBOT_MODEL
print(pth)


robotModel = p.loadURDF(ROBOT_MODEL)

#models = setupWorld(client,1)
