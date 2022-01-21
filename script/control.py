import pybullet as p
from pybullet import getEulerFromQuaternion as Q2E
import numpy as np

from constants import *

class ctlrRobot(object):
        
    def __init__(self, robot):

        self.defaultAction = np.array(DEFAULT_ACTION)
        self.minVel = np.array([-1] * NUM_ACTIONS)
        self.maxVel = np.array([1] * NUM_ACTIONS)

        self.robot = robot
        self.constraintID = p.createConstraint(robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

        self._flagAction = True

        self.reset()
        self.rcvObservation()

        return

    def reset(self):

        ## Initiate internal variables

        self._cntStep = 0
        self._action = np.zeros(3)

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

        Xaction_clip = max(min(action[0],10),-10)
        Yaction_clip = max(min(action[1],10),-10)
        THaction_clip = max(min(action[2],10),-10)

        dX = self._pose[0] + TIME_STEP*Xaction_clip
        dY = self._pose[1] + TIME_STEP*Yaction_clip
        dth = Q2E(self._ori)[2] + TIME_STEP*THaction_clip


        pivot = [dX,dY,0]
        ori = p.getQuaternionFromEuler([0, 0, dth])
        p.changeConstraint(self.constraintID, jointChildPivot=pivot, jointChildFrameOrientation=ori, maxForce=50)

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




