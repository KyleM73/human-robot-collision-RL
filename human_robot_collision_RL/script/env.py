import os

import pybullet as p
from pybullet import getEulerFromQuaternion as Q2E
import pybullet_utils.bullet_client as bc
import pybullet_data

import numpy as np
import datetime
import time

from gym import spaces, Env
from stable_baselines3.common.env_checker import check_env
import cv2

from human_robot_collision_RL.data.man import Man

from human_robot_collision_RL.script.collision import Collision
from human_robot_collision_RL.script.constants import *
#from human_robot_collision_RL.script.control import ctlrRobot
from human_robot_collision_RL.script.util import *

from human_robot_collision_RL.script.config.collision_params import *
from human_robot_collision_RL.script.config.rewards import *

class safetyEnv(Env):
    def __init__(self, training=True, reward={}, maxSteps=MAX_STEPS, include_human=False):
        '''
        EGO-CENTRIC
        Observation Space -> vX,vY,vthZ,gX,gY,gthZ,hX,hY,hthZ,[bpX,bpY,bpZ]x20 joints [5+3h+3*20*h] (68 for h=1)
        Action Space -> dvX,dvY,dvthZ [3] (effectively accelerations, see control.py)
        '''
        self.training = training
        self.include_human = include_human
        if training:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            #p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.dictRewardCoeff = reward

        obs_shape = 6 + int(self.include_human)*3

        self.observation_space_vector = spaces.Box(low=-1,
                                            high=1,
                                            shape=(obs_shape,), 
                                            dtype=np.float32
                                            )
        self.observation_space_img = spaces.Box(low=0,
                                                high=1,
                                                shape=(1,CAM_WIDTH,CAM_HEIGHT),
                                                dtype=np.uint8)

        self.observation_space = spaces.Dict(
            spaces={
                "vec": self.observation_space_vector,
                "img": self.observation_space_img
                    }
                )

        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(NUM_ACTIONS,),
                                       dtype=np.float32
                                       )
        self.maxSteps = maxSteps

        self.record = False
        self.recorder = None

        self.projMatrix = p.computeProjectionMatrixFOV(
            fov=CAM_FOV,
            aspect=float(CAM_WIDTH) / CAM_HEIGHT,
            nearVal=CAM_NEARVAL,
            farVal=CAM_FARVAL)

        #self.lastAction = np.array([0,0,0])

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
        self.client.setGravity(GRAVITY[0],GRAVITY[1],GRAVITY[2])

        # setup environment
        self.frames = 0
        self.min_px = 0
        self.max_px = 0
        self.next_goal = False
        self.inCollision = False
        self.waypoints = GOAL_POSES
        self.waypt = 0
        self.goal_pose = self.waypoints[self.waypt]
        ## self.waypoints = [w1,w2,w3,..,wn]
        ## self.goal_pose = self.waypoints[0]
        ## if dist(self.goal_pose) < thresh:
        ##      self.goal = setupGoal(self.waypoints[n+1])
        self.human_pose = HUMAN_POSE
        self.human_ori = HUMAN_ORI
        self.randAng = 0#2*PI*np.random.rand()
        self.rot = np.array([
            [np.cos(self.randAng), -np.sin(self.randAng), 0],
            [np.sin(self.randAng),  np.cos(self.randAng), 0],
            [0                   ,  0                   , 1]
            ])
        
        # creating environment
        self.ground = setupGround(self.client)
        self.walls = setupWalls(self.client,self.randAng)
        self.robot = setupRobot(self.client) ##add ori when it matters
        self.goals = [setupGoal(self.client,self.rot@np.array(waypt)) for waypt in self.waypoints]
        self.goal = self.goals[self.waypt]
        #self.goal = setupGoal(self.client,self.rot@np.array(self.goal_pose))
        if self.include_human:
            self.human = setupHuman(self.client,self.rot@np.array(self.human_pose),self.randAng)

            #self.human.reset()

            #TODO: use to get person walking
            #self.human.fix()
            #self.human.resetGlobalTransformation() #can add args later, using defaults

            self.nJ = p.getNumJoints(self.human.body_id)
            self.linkList = [i for i in range(self.nJ)]

            self.collider = Collision(
                self.client._client,
                robot=self.robot,
                human=self.human
                )

        self.target = self._getTargets()
        self.collisions = []

        self.cnt = 0

        for _ in range(REPEAT_INIT):
            self.client.stepSimulation()
        self._getObs()
        self._evaluate()

    def step(self, action):

        ##TODO
        # update goal in real time
        # dont spawn human until robot makes it to waypoint X
        # include updated positional / velocity rewards
        # clean up code (remove unneccesary stuff)
        # put unused functions / files in <depreciated> folder
        # make experiment plan
        # new state definition?
        # simplify human state?
        # collision thresh. scaling with age / size?
            # find data

        self._runSim(action)
        img = self.render()
        ob = self._getObs(img)
        robot_pose = np.linalg.inv(self.rot) @ p.getBasePositionAndOrientation(self.robot)[0]
        goal_pose = np.linalg.inv(self.rot) @ p.getBasePositionAndOrientation(self.goals[self.waypt])[0]
        if robot_pose[1] >= goal_pose[1]:
            self.next_goal = True
            self.waypt += 1
            #self.goal_pose = self.waypoints[self.waypt]
            #if not self.training:
            #    p.removeBody(self.goal)
            if self.waypt != len(self.goals):
                self.goal = self.goals[self.waypt]

        reward, done, dictLog = self._evaluate(ob["vec"],action)

        if self.record:
            if self.recorder == None:
                self._startRecorder()

            self._writeRecorder(img)

            if done:
                self._closeRecorder()
        #if not self.training:
        #    self.client.getCameraImage(480,320)

        return ob, reward, done, dictLog

    def _runSim(self,action):
        for _ in range(REPEAT_ACTION):
            self._checkCollision(self.robot,self.walls)
            self._control(action)
            self.client.stepSimulation()

    def _checkCollision(self,A,B):
        collision = p.getContactPoints(A,B)
        if collision:
            self.collisions = [c[9] for c in collision if c[9] > 0]
            #self.inCollision = True
            return
        else:
            self.collisions = []
            #self.inCollision = False
            return 

    def _control(self,action):

        if self.inCollision:
            return

        ax = action[0] 
        ay = action[1]
        az = 0#-9.8
        ath = 0#action[2]

        vX = self.obsVel[0] + ax*TIME_STEP
        vY = self.obsVel[1] + ay*TIME_STEP

        if (vX**2 + vY**2)**0.5 <= 1:
            p.applyExternalForce(self.robot,-1,[M_ROBOT*ax,M_ROBOT*ay,M_ROBOT*az],[0,0,0],p.WORLD_FRAME)
        #p.resetBaseVelocity(self.robot,[vX,vY,vZ],[0,0,0])

    def _getTargets(self):

        XYZ, TH = p.getBasePositionAndOrientation(self.goal)

        return XYZ,TH

    def _getObs(self,img=None):

        self.bodyPose, self.obsBodyOri = p.getBasePositionAndOrientation(self.robot)
        self.obsVel,self.obsAngularRate = p.getBaseVelocity(self.robot)
        
        targetXYZ,targetTH = self._getTargets()
        self.obsTarget = [(targetXYZ[i] - self.bodyPose[i])/FIELD_RANGE for i in range(3)]
        self.obsTargetOriEgo = (Q2E(self.obsBodyOri)[2] - Q2E(targetTH)[2])/(2*PI) #angle between robot heading and target
        
        #humanPose,humanOri = p.getBasePositionAndOrientation(self.human.body_id)
        #obsHumanPose = [(humanPose[i] - bodyPose[i])/FIELD_RANGE for i in range(3)]
        #obsHumanOriEgo = (Q2E(obsBodyOri)[2] - Q2E(humanOri)[2])/(2*PI) #angle between robot heading and human
        
        #links = p.getLinkStates(self.human.body_id,self.linkList)
        #obsLinks = [l_/FIELD_RANGE for l in links for l_ in list(l[0])]

        obs = np.concatenate((
            self.obsVel[0:2], 
            self.obsAngularRate[2], 
            self.obsTarget[0:2],
            self.obsTargetOriEgo,
            #obsHumanPose[0:2],
            #obsHumanOriEgo
            ), axis=None)

        if img is None:
            img = self.render()
        self.img = (img[:,:,0]/255.).reshape((1,CAM_HEIGHT,CAM_WIDTH))

        return {"vec" : obs, "img" : self.img}

    def _evaluate(self,ob=None,action=None):

        done = False
        dictLog = {}
        dictRew = {}
        dictState = {}

        #maxCost = max_cost #see rewards.max_cost

        #Fdict = self._getCollision()

        if ob is None:
            obDict = self._getObs()
            ob = obDict["vec"]
            obimg = obDict["img"]

        #bodyPose = FIELD_RANGE*np.array(ob[0:2])
        #bodyOri = 2*PI*np.array(ob[2])
        bodyVel = np.array(ob[0:2])
        bodyAngularVel = np.array(ob[2])
        target = FIELD_RANGE*np.array(ob[3:5])
        targetOri = 2*PI*np.array(ob[5])

        sqrErrPose = np.sum((target)**2)
        sqrErrAng = targetOri**2 
        sqrErrVel = np.sum(bodyVel**2)
        sqrErrAngVel = bodyAngularVel**2

        dictState["Position"] = sqrErrPose
        dictState["Angle"] = sqrErrAng
        dictState["Velocity"] = sqrErrVel
        dictState["AngularVelocity"] = sqrErrAngVel

        #collisionReward = 0
        #if Fdict is not None:
        #    for F in Fdict.keys():
        #        (rew,maxCostFlag) = getCollisionReward(Fdict,F,collisionDictTransient,collisionDictQuasiStatic,maxCost)
        #        collisionReward += self.dictRewardCoeff["Collision"][F] * rew
        #        if maxCostFlag : done = True

        #smoothRew = 0
        #if action is not None:
        #    diffAction = abs(action - self.lastAction)
        #    for i in range(action.shape[0]):
        #        if diffAction[i] < MAX_ACTION_DIFF:
        #            smoothRew += self.dictRewardCoeff["ActionSmooth"]*diffAction[i]
        #        else:
        #            smoothRew += self.dictRewardCoeff["ActionNotSmooth"]*diffAction[i]
        #    self.lastAction = action
        #dictRew["Action"] = smoothRew


        dictRew["Position"] = self.dictRewardCoeff["Position"] * sqrErrPose
        #dictRew["Angle"] = self.dictRewardCoeff["Angle"] * sqrErrAng

        if self.collisions:
            dictRew["Collisions"] = -5

        
        #only penalize velocity near the target
        #if sqrErrPose < vel_penalty_radius: #see rewards.vel_penalty_radius
        #    dictRew["Velocity"] = self.dictRewardCoeff["Velocity"] * sqrErrVel   
        #else:
        #    dictRew["Velocity"] = 0
        
        #dictRew["AngularVelocity"] = self.dictRewardCoeff["AngularVelocity"] * sqrErrAngVel
        #dictRew["Collision"] = collisionReward

        #if p.getBasePositionAndOrientation(self.robot)[0][2] > self.initZ + MAX_HEIGHT_DEVIATION:
        #    print("warning: height deviation detected")

        if self.next_goal:
            dictRew["Goal"] = self.dictRewardCoeff["Goal"]
            self.next_goal = False
            if self.waypt == len(self.waypoints):
                dictRew["Final Goal"] = 10*self.dictRewardCoeff["Goal"]
                done = True
        elif self.cnt > self.maxSteps:
            dictRew["Fail"] = self.dictRewardCoeff["Fail"]
            done = True
        else:
            self.cnt+=1

        reward = 0
        for rew in dictRew.values():
            reward += rew

        dictRew["Sum"] = reward

        if done:
            dictLog["Done"] = 1
        dictLog["Reward"] = dictRew
        dictLog["State"] = dictState

        return reward, done, dictLog


    def setRecord(self,v=True,log_path=None,best=False):
        self.record = v
        self.best = best

        if self.record:
            self._cameraDist = CAM_DIST
            self._cameraYaw = CAM_YAW
            self._cameraPitch = CAM_PITCH
            self._renderWidth = CAM_WIDTH
            self._renderHeight = CAM_HEIGHT
            self._videoFormat = cv2.VideoWriter_fourcc(*'mp4v')
            if log_path is not None:
                self._videoSavePath = log_path #do not end in "/", see _startRecord()
            else:
                print("error: no video save path")
            self.recorder = None

    def _writeRecorder(self,img=None):
    
        if img is None:
            img = self.render()
        self.recorder.write(img)

    def _startRecorder(self,name=DT):
        if self.best:
            name = name + "_best"

        path = "{}/{}.mp4".format(self._videoSavePath, name)
        self.recorder = cv2.VideoWriter(path, self._videoFormat, CAM_FPS, (self._renderWidth, self._renderHeight))

    def _closeRecorder(self):

        self.recorder.release()
        self.recorder = None

    def render(self,*args):
        
        #if not self.record:
        #    return
        
        try:
            bodyPose, obsBodyOri = p.getBasePositionAndOrientation(self.robot)
        except:
            bodyPose = np.zeros(3)
            obsBodyOri = [0,0,0,1]

        targetPose = self.rot @ np.array([0,100,0.5])

        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[bodyPose[0], bodyPose[1], bodyPose[2]],
            cameraTargetPosition=targetPose,
            cameraUpVector=[0,0,1])
        #start = time.time()
        (_, _, _, dpth, _) = p.getCameraImage(
            width=CAM_WIDTH,
            height=CAM_HEIGHT,
            renderer=p.ER_TINY_RENDERER,#ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=viewMatrix,
            projectionMatrix=self.projMatrix,
            shadow=0)
        #stop = time.time()
        #fps = 1./(stop - start)
        #print(fps)
            
        #import matplotlib.pyplot as plt
        depth_buffer = np.asarray(dpth)#np.reshape(dpth, [w, h])
        #depth = CAM_FARVAL * CAM_NEARVAL / (CAM_FARVAL - (CAM_FARVAL - CAM_NEARVAL) * depth_buffer)
        self.frames += 1
        self.min_px += np.min(depth_buffer)
        self.max_px += np.max(depth_buffer)
        depth = (self.frames*depth_buffer - self.min_px) / (self.max_px - self.min_px) #running avg min/max normalization
        depth = 255*depth
        depth = np.where(depth<0,0,depth)
        depth = np.where(depth>255,255,depth)
        

        #plt.subplot(1, 1, 1)
        #plt.imshow(depth, cmap='gray', vmin=0, vmax=1)
        #plt.title('Tiny Renderer')

        #plt.show()
        #return rgba2rgb(rgbArray)
        #dpthArray = np.asarray(dpth, dtype='uint8')
        img = np.zeros((CAM_HEIGHT,CAM_WIDTH,3))
        img[:,:,0] = depth
        img[:,:,1] = depth
        img[:,:,2] = depth
        return img.astype('uint8')
        #cv2.imshow("img",img)
        #assert 1 == 0

        #return 255*cv2.cvtColor(np.asarray(dpth, dtype='uint8'),cv2.COLOR_GRAY2RGB)

if __name__ == "__main__": 
    #env = myEnv(False,reward=rewardDict)
    env = safetyEnv(True,reward=rewardDict)#,humans=1)
    obs = env.reset()
    #act = np.array([-np.sin(env.randAng),np.cos(env.randAng),0]) #[m/s]
    act = np.array([0,1,0])
    #env.setRecord(True,PATH_TMP)
    for _ in range(MAX_STEPS*100):
        ob, reward, done, dictLog = env.step(act)
        #env.render()
        #time.sleep(TIME_STEP)
        if done:
            break





