import os

import pybullet as p
from pybullet import getEulerFromQuaternion as Q2E
from pybullet import getQuaternionFromEuler as E2Q
import pybullet_data

import numpy as np

import shutil

from human_robot_collision_RL.data.man import Man

from human_robot_collision_RL.script.constants import *

def setupGoal(client,pose=[0,10,0.5]):
    c = client

    #sphere = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[0,0,0,0], radius=0.25 )
    idCollisionShape = None
    sphere = None
    goal = p.createMultiBody(
        #baseVisualShapeIndex=sphere, 
        basePosition=pose)
    return goal


def setupGoalDuck(client, pose):
    c = client
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  

    meshScale = [1, 1, 1]
    #the visual shapes and collision shapes can be re-used by all createMultiBody instances (instancing)
    idVisualShape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        halfExtents=[0, 0, 0],
        fileName="duck.obj",
        visualFrameOrientation=p.getQuaternionFromEuler([PI/2,0,PI/2]),
        meshScale=meshScale)

    idCollisionShape = None

    ##TODO: debug, doesn't always face the right direction, duck should point towards robot's initial starting loc
    #pose = [0,-1,0]
    #FIX THIS!!!!
    #way point behavior? floating red spheres
    goalPosition = [pose[0], pose[1], 0.025]
    goalPoseScaled = pose/np.linalg.norm(goalPosition)
    ref = np.array([1,0,0])
    th = np.arccos(np.dot(ref[0:2],goalPoseScaled[0:2]))
    if pose[1] >= 0:
        th -= PI/2
    else:
        th += PI/2
        if pose[0] > 0:
            th += PI/2
        elif pose[0] < 0:
            th -= PI/2
    #print(th)
    goalModel = p.createMultiBody(
        baseVisualShapeIndex=idVisualShape, 
        basePosition=goalPosition,
        baseOrientation=p.getQuaternionFromEuler([0,0,th])
        )

    return goalModel


def setupRobot(client, pose=ROBOT_POSE, ori=ROBOT_ORI):
    '''
    Args:
        client: pybullet client
        pose  : XYZ position of the robot [list]
        ori   : orientation of the robot in XYZ euler angles [list]

    Out:
        robotModel : pybullet model of the robot
    '''
    c = client 

    oriQ = p.getQuaternionFromEuler(ori)

    robotModel = p.loadURDF(
        PATH_DATA+ROBOT_MODEL,
        basePosition=pose,
        baseOrientation=oriQ,
        flags=p.URDF_USE_IMPLICIT_CYLINDER
        )

    p.changeDynamics(robotModel,-1,
        lateralFriction=0,
        spinningFriction=0, #TODO: are these necessary? need to test
        rollingFriction=0
        )

    return robotModel

def setupGround(client):
    c = client

    shapePlane = p.createCollisionShape(shapeType = p.GEOM_PLANE)
    terrainModel  = p.createMultiBody(0, shapePlane)
    p.changeDynamics(terrainModel, -1, lateralFriction=1.0) 

    return terrainModel

def setupWalls(client,oriZ=0):
    c = client

    walls = p.loadURDF("walls.urdf",basePosition=rotZ(oriZ)@np.array([0,-0.5,0.5]),baseOrientation=E2Q([0,0,oriZ]),useFixedBase=1)
    p.changeDynamics(walls, -1, lateralFriction=1.0)
    return walls

def setupHuman(client,pose=[0,4,1.112],oriZ=0):
    c = client 

    human = Man(client._client,
            partitioned=False,
            self_collisions=False,
            pose=pose,
            ori=[PI/2,0,oriZ],
            fixed=1,
            timestep=TIME_STEP,
            scaling=1)
    return human

def rotZ(th):
    return np.array([
            [np.cos(th), -np.sin(th), 0],
            [np.sin(th),  np.cos(th), 0],
            [0         ,  0         , 1]
            ])

def setupWorld(client,humans=None,humanPose=HUMAN_POSE):
    c = client

    shapePlane = p.createCollisionShape(shapeType = p.GEOM_PLANE)
    terrainModel  = p.createMultiBody(0, shapePlane)
    p.changeDynamics(terrainModel, -1, lateralFriction=1.0) 

    #sample goal poses such that the goal is at minimum 4m from the robot
    goalDist = 0
    while goalDist < MIN_TARGET_DIST:
        goalPose = np.random.uniform(low=-FIELD_RANGE, high=FIELD_RANGE, size=3)
        goalPose[2] = 0 #set z coord
        goalDist = np.linalg.norm(goalPose)
    goalPose = TEST_POSE #FOR DEBUGGING ONLY
    goalModel = setupGoal(c, goalPose)

    robotModel = setupRobot(c, ROBOT_POSE, ROBOT_ORI)

    modelsDict = {
        'robot': robotModel,
        'terrain': terrainModel,
        'goal': goalModel,
        }

    if isinstance(humans,int):
        ## make human here
        #TODO: add multiple human options
        #TODO: make humans non fixed
        #TODO: let humans walk (implimented but unused currently)
        #random angle
        rn = 2*np.random.random_sample()-1 #uniform random interval [-1,1)
        #random scale between human and target spawn location
        rnScale = 0.5*np.random.random_sample()+0.25 #uniform random interval [0.25,0.75)
        #random scale of simulated human
        rnHumanScale = 0.035*np.random.randn()+1 #normal random interval mu=1,sigma=0.035 (men=0.36, women=0.34)
        if rn < 0:
            #standing 
            humanPose = INIT_POSE_LIST[0][0] + rnScale*goalPose + (rnHumanScale-1)
            humanOri = np.array(INIT_POSE_LIST[0][1]) + PI*np.array([0,0,rn])
        else:
            #laying face up
            humanPose = INIT_POSE_LIST[1][0] + rnScale*goalPose
            humanOri = np.array(INIT_POSE_LIST[1][1]) + PI*np.array([0,0,rn])

        humanPose = POSE + goalPose/2
        humanModel = Man(c._client,
            partitioned=False,
            self_collisions=False,
            pose=humanPose,
            ori=humanOri,
            fixed=1,
            timestep=TIME_STEP,
            scaling=rnHumanScale)
        modelsDict['human'] = humanModel

    return modelsDict

def rgba2rgb( rgba, background=(255,255,255) ):
    #https://stackoverflow.com/a/58748986
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

def copy_reward_gains(path):
    reward_path = r'{}/{}'.format(PATH_CONFIG,'rewards.py')
    new_reward_path = r'{}/{}'.format(path,'rewards.py')

    collision_params_path = r'{}/{}'.format(PATH_CONFIG,'collision_params.py')
    new_collision_params_path = r'{}/{}'.format(path,'collision_params.py')

    constants_path = r'{}/{}'.format(PATH_SCRIPT,'constants.py')
    new_constants_path = r'{}/{}'.format(path,'constants.py')

    os.makedirs(os.path.dirname(new_reward_path), exist_ok=True)
    os.makedirs(os.path.dirname(new_collision_params_path), exist_ok=True)
    os.makedirs(os.path.dirname(new_constants_path), exist_ok=True)

    shutil.copyfile(reward_path, new_reward_path)
    shutil.copyfile(collision_params_path, new_collision_params_path)
    shutil.copyfile(constants_path, new_constants_path)


