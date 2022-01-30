import numpy as np

import pybullet as p
from pybullet import getEulerFromQuaternion as Q2E

from human_robot_collision_RL.script.constants import *

class ctlrRobot(object):
    def __init__(self, robot):

        self.defaultAction = np.array(DEFAULT_ACTION) # see constants
        self.minVel = np.array([-1] * NUM_ACTIONS)
        self.maxVel = np.array([1] * NUM_ACTIONS)
        self.max_dV = 1

        self.robot = robot
        #self.constraintID = p.createConstraint(robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

        self._flagAction = True

        self.reset()
        self.rcvObservation()

        return

    def reset(self):

        ## Initiate internal variables

        self._cntStep = 0
        self._action = np.zeros(3) #action space [vX vY vTh]

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
        if self._pose[2] > 0.1:
            vZ = -1
        else:
            vZ = 0

        #TODO: set max acceleration (change in velocity)
        #idea: get vel. if commanded vel has opposite sign as current vel, command zero vel
        #because we set vel instantly this is a non-issue, but is a big issue on the real robot
        #maybe policy -> smoother -> PD controller -> motor torques
        #dVx = abs(self._vel[0] - action[0])
        #dVy = abs(self._vel[1] - action[1])
        #dVth = abs(self._angularRate[2] - action[2])

        #dV = [dVx,dVy,dVth]

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




