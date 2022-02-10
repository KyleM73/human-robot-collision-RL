import pybullet as p
from pybullet import getEulerFromQuaternion as Q2E

import numpy as np

from human_robot_collision_RL.script.constants import *

class ctlrRobot(object):
    def __init__(self, robot, actions=NUM_ACTIONS):

        self.defaultAction = np.array(DEFAULT_ACTION) # see constants
        self.minVel = np.array([-1] * actions)
        self.maxVel = np.array([1] * actions)
        #self.max_dV = 1

        self.robot = robot
        #self.constraintID = p.createConstraint(robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

        self._flagAction = True
        self.inCollision = False

        self.reset()
        self.rcvObservation()

        return

    def reset(self):

        ## Initiate internal variables

        self._cntStep = 0
        self._action = np.zeros(NUM_ACTIONS) #action space [vX vY vTh]

        self.rcvObservation()

        return


    def holdRobot(self):

        self.rcvObservation()
        self.applyAction(self._action)

    def rcvObservation(self):

        self._pose,self._ori = p.getBasePositionAndOrientation(self.robot)
        self._vel,self._angularRate = p.getBaseVelocity(self.robot)

        return self._pose,self._ori,self._vel,self._angularRate

    def applyAction(self, action):

        #dX = self._pose[0] + TIME_STEP*action[0]
        #dY = self._pose[1] + TIME_STEP*action[1]

        #dth = Q2E(self._ori)[2] + TIME_STEP*action[2]

        #pivot = [dX,dY,0]
        #ori = p.getQuaternionFromEuler([0, 0, dth])
        #p.changeConstraint(self.constraintID,jointChildPivot=pivot,jointChildFrameOrientation=ori,maxForce=20)

        # note: setting the velocity like this violates the simulation dynamics 
        # and will not yield accurate collisions after the initial contact
        #vZ = self._vel[2] - GRAVITY[2]*TIME_STEP
        vZ = 0
        #TODO: set max acceleration (change in velocity)
        #idea: get vel. if commanded vel has opposite sign as current vel, command zero vel
        #because we set vel instantly this is a non-issue, but is a big issue on the real robot
        #maybe policy -> smoother -> PD controller -> motor torques

        '''
        dVx = self._vel[0] - action[0]
        dVy = self._vel[1] - action[1]
        dVth = self._angularRate[2] - action[2]

        dV = [dVx,dVy,dVth]
        dVV = []
        for v in dV:
            #dont change velocity by more than 0.1 m/s at a time
            if v > 0.5:
                dVV.append(v - 0.5)
            elif v < 0.01:
                dVV.append(v + 0.5)
            else:
                dVV.append(v)



        p.resetBaseVelocity(self.robot,[dVV[0],dVV[1],vZ],[0,0,dVV[2]])
        '''
        if not self.inCollision:
            p.resetBaseVelocity(self.robot,[action[0],action[1],vZ],[0,0,action[2]])

        return
    
    def step(self):

        self.rcvObservation()
        self.applyAction(self._action)
        self._cntStep += 1

        return

    def getTimeSinceReset(self):
        return self._cntStep * TIME_STEP

    def getPoseAndOri(self):
        return self._pose,self._ori

    def getVelAndAngularRate(self):
        return self._vel,self._angularRate


    def updateAction(self, action):

        if self._flagAction:
            self._flagAction = False
            self._action = action
            return True
        else:
            if self._cntStep % REPEAT_ACTION == 0:
                self._flagAction = True
            return False

    def setCollision(self, flag):
        self.inCollision = flag




