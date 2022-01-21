import pybullet as p
from pybullet import getEulerFromQuaternion as Q2E
import pybullet_utils.bullet_client as bc
import pybullet_data

import numpy as np
import datetime
import time

from gym import spaces, Env
from stable_baselines3.common.env_checker import check_env
#import cv2

from constants import *
import control


def setupGoal(client, pose):
    c = client
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  

    #idVisualShape = p.createVisualShape(p.GEOM_CYLINDER, radius=1, length=1)

    meshScale = [1, 1, 1]
    #the visual shapes and collision shapes can be re-used by all createMultiBody instances (instancing)
    idVisualShape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        halfExtents=[0, 0, 0],
        fileName="duck.obj",
        visualFrameOrientation=p.getQuaternionFromEuler([PI/2,0,PI/2]),
        meshScale=meshScale)

    idCollisionShape = None

    goalPosition = [pose[0], pose[1], 0.025]
    goalModel = p.createMultiBody(
        baseVisualShapeIndex=idVisualShape, 
        basePosition=goalPosition
        )

    return goalModel


def setupRobot(client, pose=[0,0,0], ori=[0,0,0]):
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

    H = 0.635 #[m]
    R = 0.2794 #[m]
    M = 4.53592 #[kg]

    robotModel = p.loadURDF(
        PATH_DATA+"/trikey2.urdf",
        basePosition=pose,
        baseOrientation=oriQ
        )

    '''
    robotCollision = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=R,
        height=H
        )

    robotModel = p.createMultiBody(
        baseMass=M,
        baseCollisionShapeIndex=robotCollision,
        basePosition=pose,
        baseOrientation=oriQ
        )
    '''

    p.changeDynamics(robotModel,-1,lateralFriction=0)

    return robotModel


def setupWorld(client):
    c = client

    shapePlane = p.createCollisionShape(shapeType = p.GEOM_PLANE)
    terrainModel  = p.createMultiBody(0, shapePlane)
    p.changeDynamics(terrainModel, -1, lateralFriction=1.0) 

    goalModel = setupGoal(c, np.random.uniform(low=-FIELD_RANGE, high=FIELD_RANGE, size=2))

    robotModel = setupRobot(c, [0., 0., 0.5], [0, 0, 0])

    modelsDict = {'robot': robotModel, 'terrain': terrainModel, 'goal': goalModel}

    return modelsDict


class myEnv(Env):
    def __init__(self, training=True, reward={"Position":1}, maxSteps=2000):
        '''
        Observation Space -> X,Y,thZ,vX,vY,vthZ,gX,gY,gthZ,gvX,gvY,gvthZ [12]
        Action Space -> vX,vY,vthZ [3]
        '''
        if training:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.dictRewardCoeff = reward

        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(8,), 
                                            dtype=np.float32
                                            )

        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(3,),
                                       dtype=np.float32
                                       )
        self.maxSteps = maxSteps

    def reset(self):

        # Generate a new episode
        self._setup()

        # Observation initiation
        ob = self._getObs()

        return ob

    def _setup(self):

        ## Initiate simulation
        self.client.resetSimulation()

        self.client.setAdditionalSearchPath(PATH_DATA)

        ## Set up simulation
        self.client.setTimeStep(TIME_STEP)
        self.client.setPhysicsEngineParameter(numSolverIterations=int(30))
        self.client.setPhysicsEngineParameter(enableConeFriction=0)
 
        ## Set up playground
        self.client.setGravity(0, 0, -9.8)

        # creating environment
        self.models = setupWorld(self.client)
        self.robot = self.models['robot']

        self.goal = self.models['goal']

        self.control = control.ctlrRobot(self.robot)
        self.dictCmdParam = {"Offset": np.zeros(NUM_COMMANDS), 
                            "Scale":  np.array([1] * NUM_COMMANDS)}
        self.dictActParam = {"Offset": np.zeros(NUM_ACTIONS), 
                            "Scale":  np.array([1] * NUM_ACTIONS)}
        self.target = self._getTargets()

        self.cnt = 0

        for _ in range(REPEAT_INIT):
            self.control.holdRobot()
            self.client.stepSimulation()
            self._getObs()
        self._evaluate()



    def step(self, action):

        self._runSim(action)
        ob = self._getObs()
        reward, done, dictLog = self._evaluate()

        return ob, reward, done, dictLog


    def _runSim(self, action):

        while not self.control.updateAction(self.dictActParam["Scale"] * action + self.dictActParam["Offset"]):
            self.control.step()
            self.client.stepSimulation()


    def _evaluate(self):

        done = False
        dictLog = {}
        dictRew = {}
        dictState = {}
        reward = 0

        target = np.array(self._getTargets())
        bodyPose_, bodyOri_ = p.getBasePositionAndOrientation(self.robot)
        bodyPose = np.array(bodyPose_)
        bodyOri = np.array(Q2E(bodyOri_))

        ### YOUR ENVIRONMENT CONSTRAINTS HERE ###
        
        sqrErr = np.sum((target[0:2] - bodyPose[0:2])**2)
        
        dictState["Distance"] = sqrErr

        dictRew["Position"] = self.dictRewardCoeff["Position"] * sqrErr

        if sqrErr < 0.01:
            done = True
            dictRew["Goal"] = self.dictRewardCoeff["Goal"]
        elif self.cnt > self.maxSteps:
            dictRew["Fail"] = self.dictRewardCoeff["Fail"]
            done = True
        else:
            self.cnt+=1

        for rew in dictRew.values():
            reward += rew

        dictRew["Sum"] = reward

        if done:
            dictLog["Done"] = 1
        dictLog["Reward"] = dictRew
        dictLog["State"] = dictState

        return reward, done, dictLog

    def _getObs(self):

        bodyPose, obsBodyOri = p.getBasePositionAndOrientation(self.robot)
        obsBodyPose = [bodyPose[i]/FIELD_RANGE for i in range(3)]
        obsVel,obsAngularRate = p.getBaseVelocity(self.robot)
        obsTarget = [self._getTargets()[i] - obsBodyPose[i] for i in range(3)]
        
        obs = np.concatenate((obsBodyPose[0:2], Q2E(obsBodyOri)[2], obsVel[0:2], obsAngularRate[2], obsTarget[0:2]/FIELD_RANGE), axis=None)

        return obs


    def _getTargets(self):

        XYZ, _ = self.client.getBasePositionAndOrientation(self.goal)

        return XYZ
    
    def seed(self):
        pass

    def close(self):
        pass 

    def render(self):
        return      



env = myEnv(False)
check_env(env)




"""
E = myEnv(False)
ob = E.reset()
for i in range(1000):
    ob, reward, done, dictLog = E.step([0,1,0])
    time.sleep(1./240.)
print(ob)

client = p.connect(p.GUI) #DIRECT or GUI
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.resetSimulation()
p.setGravity(0,0,-9.81)
d = setupWorld(client)

p.resetBaseVelocity(d['robot'],[0,0,0])

for i in range(20000):
    position, orientation = p.getBasePositionAndOrientation(d['robot'])
    if i%100==0: print(orientation)
    p.stepSimulation()
    time.sleep(1./240.)

"""




