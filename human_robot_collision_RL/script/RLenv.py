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

from human_robot_collision_RL.script.constants import *
from human_robot_collision_RL.script.control import ctlrRobot
from human_robot_collision_RL.script.config.rewards import *
from human_robot_collision_RL.script.config.collision_params import *
from human_robot_collision_RL.data.man import Man
from human_robot_collision_RL.script.collision import Collision

def setupGoal(client, pose):
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


def setupRobot(client, pose=[0,-1,0.5], ori=[0,0,0]):
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
        PATH_DATA+"/trikey2.urdf",
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


def setupWorld(client,humans=None,humanPose=POSE):
    c = client

    shapePlane = p.createCollisionShape(shapeType = p.GEOM_PLANE)
    terrainModel  = p.createMultiBody(0, shapePlane)
    p.changeDynamics(terrainModel, -1, lateralFriction=1.0) 

    #sample goal poses such that the goal is at minimum 4m from the robot
    goalDist = 0
    while goalDist < 4:
        goalPose = np.random.uniform(low=-FIELD_RANGE, high=FIELD_RANGE, size=3)
        goalPose[2] = 0 #set z coord
        goalDist = np.linalg.norm(goalPose)
    #goalPose = TEST_POSE #FOR DEBUGGING ONLY
    goalModel = setupGoal(c, goalPose)

    robotModel = setupRobot(c, [0., 0., 0.5], [0, 0, 0])

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
        rn = 2*np.random.random_sample()-1 #uniform random interval [-1,1)
        if rn < 0: 
            humanPose = INIT_POSE_LIST[0][0] + goalPose/2
            humanOri = np.array(INIT_POSE_LIST[0][1]) + PI*np.array([0,0,rn])
        else:
            humanPose = INIT_POSE_LIST[1][0] + goalPose/2
            humanOri = np.array(INIT_POSE_LIST[1][1]) + PI*np.array([0,0,rn])

        #humanPose = POSE + goalPose/2
        humanModel = Man(c._client,partitioned=False,self_collisions=False,fixed=1,timestep=TIME_STEP,pose=humanPose,ori=humanOri)
        modelsDict['human'] = humanModel

    return modelsDict


class myEnv(Env):
    def __init__(self, training=True, reward={}, maxSteps=2000):
        '''
        Observation Space -> X,Y,thZ,vX,vY,vthZ,gX,gY,gthZ [8]
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

        self.record = False
        self.recorder = None

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
        self.client.setGravity(0, 0, -9.8)

        # creating environment
        self.models = setupWorld(self.client)
        self.robot = self.models['robot']

        self.goal = self.models['goal']

        # controller executes actions commanded from RL policy
        self.control = ctlrRobot(self.robot)

        self.dictActParam = {"Offset": np.zeros(NUM_ACTIONS), 
                            "Scale":  np.array([1] * NUM_ACTIONS)}
        
        self.target = self._getTargets()

        self.cnt = 0

        #initialize robot for REPEAT_INIT timesteps
        for _ in range(REPEAT_INIT):
            #self.control.holdRobot() #let robot fall
            self.client.stepSimulation()
            self._getObs()
        self._evaluate()


    def step(self, action):

        #advance simulation one timestep 
        self._runSim(action)

        #make observation of state/environment
        ob = self._getObs()

        #evaluate the action based on the recieved observation
        reward, done, dictLog = self._evaluate()

        if self.record:
            if self.recorder == None:
                self._startRecorder()

            self._writeRecorder()

            if done:
                self._closeRecorder()

        return ob, reward, done, dictLog


    def _runSim(self, action):

        #run simulation with current action REPEAT_ACTION times
        while not self.control.updateAction(self.dictActParam["Scale"] * action + self.dictActParam["Offset"]):
            self.control.step()
            self.client.stepSimulation()


    def _evaluate(self):

        #init tracking objects
        done = False
        dictLog = {}
        dictRew = {}
        dictState = {}
        reward = 0

        #get observation
        target = np.array(self._getTargets())
        bodyPose_, bodyOri_ = p.getBasePositionAndOrientation(self.robot)
        bodyPose = np.array(bodyPose_)
        bodyOri = np.array(Q2E(bodyOri_))
        vel_,angularRate_ = p.getBaseVelocity(self.robot)
        vel = np.array(vel_)
        angularRate = np.array(angularRate_)

        #get rewards
        sqrErrPose = np.sum((target[0:2] - bodyPose[0:2])**2) + bodyPose[2]**2 
        sqrErrVel = np.sum(vel[0:2]**2) + angularRate[2]**2
        
        dictState["Distance"] = sqrErrPose

        dictRew["Position"] = self.dictRewardCoeff["Position"] * sqrErrPose
        dictRew["Velocity"] = self.dictRewardCoeff["Velocity"] * sqrErrVel

        #check done conditions
        if sqrErrPose < 0.1 and sqrErrVel < 0.1:
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
        obsBodyPose = [bodyPose[i]/FIELD_RANGE for i in range(3)] #normalize to [-1,1]
        obsVel,obsAngularRate = p.getBaseVelocity(self.robot)
        obsTarget = [self._getTargets()[i]/FIELD_RANGE for i in range(3)] #normalize to [-1,1]
        
        obs = np.concatenate((obsBodyPose[0:2], Q2E(obsBodyOri)[2]/(2*PI), obsVel[0:2], obsAngularRate[2], obsTarget[0:2]), axis=None)

        return obs


    def _getTargets(self):

        XYZ, _ = self.client.getBasePositionAndOrientation(self.goal)

        return XYZ
    
    def seed(self):
        pass #only set if we want to produce repeatable goal positions

    def close(self):
        pass 

    def setRecord(self,v=True,path='/Experiment_1/videos'):
        self.record = v

        if self.record:
            self._cameraDist = 10.0
            self._cameraYaw = 0
            self._cameraPitch = -30
            self._renderWidth = 480
            self._renderHeight = 360
            self._videoFormat = cv2.VideoWriter_fourcc(*'mp4v')
            self._videoSavePath = PATH_SAVE+path #do not end in "/", see _startRecord()
            self.recorder = None


    def _writeRecorder(self):
        
        img = self.render()
        self.recorder.write(img)


    def _startRecorder(self):

        path = "{}/{}.mp4".format(self._videoSavePath, datetime.datetime.now().strftime("%m%d_%H%M"))
        self.recorder = cv2.VideoWriter(path, self._videoFormat, 30, (self._renderWidth, self._renderHeight))


    def _closeRecorder(self):

        self.recorder.release()
        self.recorder = None

    def render(self,*args):
        
        if not self.record:
            return
        
        try:
            bodyPose, obsBodyOri = p.getBasePositionAndOrientation(self.robot)
        except:
            bodyPose = np.zeros(3)

        viewMatrix = self.client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=bodyPose,
            distance=self._cameraDist,
            yaw=self._cameraYaw,
            pitch=self._cameraPitch,
            roll=0,
            upAxisIndex=2)
        projMatrix = self.client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._renderWidth) / self._renderHeight,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = self.client.getCameraImage(
            width=self._renderWidth,
            height=self._renderHeight,
            renderer=self.client.ER_BULLET_HARDWARE_OPENGL,# ER_TINY_RENDERER,
            viewMatrix=viewMatrix,
            projectionMatrix=projMatrix)
        rgbArray = np.array(px)
        preImage = rgbArray[:, :, :3]
        image = preImage[:]
        #RGB to BGR
        image[:,:,0] = preImage[:,:,2]
        image[:,:,1] = preImage[:,:,1]
        image[:,:,2] = preImage[:,:,0]
        #TODO colors are still weird
        
        return image  

    def stable_sigmoid(self,x):
        #currently unused, may use in reward function in the future
        sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
        return sig   


class humanEnv(myEnv):
    def __init__(self, training=True, reward={}, maxSteps=2000, humans=1):
        '''
        Observation Space -> X,Y,thZ,vX,vY,vthZ,gX,gY,gthZ,hX,hY,hthZ [8+3h]
        Action Space -> vX,vY,vthZ [3]
        '''
        self.training = training
        if training:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.dictRewardCoeff = reward

        self.humans = int(humans)
        obs_shape = 8 + 3*self.humans

        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(obs_shape,), 
                                            dtype=np.float32
                                            )

        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(3,),
                                       dtype=np.float32
                                       )
        self.maxSteps = maxSteps

        self.record = False
        self.recorder = None

        self.inCollision = False

    def _setup(self):

        ## Initiate simulation
        self.client.resetSimulation()

        self.client.setAdditionalSearchPath(PATH_DATA)

        ## Set up simulation
        self.client.setTimeStep(TIME_STEP)
        self.client.setPhysicsEngineParameter(numSolverIterations=int(30))
        self.client.setPhysicsEngineParameter(enableConeFriction=0)
        self.client.setGravity(0, 0, -9.8)

        # creating environment
        self.models = setupWorld(self.client,self.humans) ## use function including human
        self.robot = self.models['robot']

        self.goal = self.models['goal']

        self.human = self.models['human']
        self.human.reset()
        #TODO: use to get person walking
        #self.human.fix()
        #self.human.resetGlobalTransformation() #can add args later, using defaults

        self.collider = Collision(
            self.client._client,
            robot=self.robot,
            human=self.human
        )

        self.control = ctlrRobot(self.robot)
        
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

        if self.record:
            if self.recorder == None:
                self._startRecorder()

            self._writeRecorder()

            if done:
                self._closeRecorder()

        return ob, reward, done, dictLog

    def _runSim(self, action):

        while not self.control.updateAction(self.dictActParam["Scale"] * action + self.dictActParam["Offset"]):
            self.control.setCollision(self.inCollision)
            self.control.step()
            #self.human.advance(POSE,p.getQuaternionFromEuler([0,0,0])) #used to advance human gait
            self.client.stepSimulation()

    def _getObs(self):

        bodyPose, obsBodyOri = p.getBasePositionAndOrientation(self.robot)
        obsBodyPose = [bodyPose[i]/FIELD_RANGE for i in range(3)]
        obsVel,obsAngularRate = p.getBaseVelocity(self.robot)
        obsTarget = [self._getTargets()[i]/FIELD_RANGE for i in range(3)]
        humanPose,humanOri = p.getBasePositionAndOrientation(self.human.body_id)
        obsHumanPose = [humanPose[i]/FIELD_RANGE for i in range(3)] #TODO: add body parts? velocity?
        
        obs = np.concatenate((obsBodyPose[0:2], Q2E(obsBodyOri)[2]/(2*PI), obsVel[0:2], obsAngularRate[2], obsTarget[0:2],obsHumanPose[0:2],Q2E(humanOri)[2]/(2*PI)), axis=None)

        return obs

    def _getCollision(self):

        vel,_ = p.getBaseVelocity(self.robot)

        Fdict = self.collider.get_collision_force(vel)
        
        if Fdict is not None:
            # ---- Collision Detected ----
            self.inCollision = True
            
            return Fdict # [N]
        else:
            if self.inCollision:
                self.inCollision = False
            return None

    def _evaluate(self):

        done = False
        dictLog = {}
        dictRew = {}
        dictState = {}
        reward = 0
        maxP = 100 #TODO: set maxP per part (need dict)
        maxCost = 150 #good value??

        Fdict = self._getCollision()

        target = np.array(self._getTargets())
        bodyPose_, bodyOri_ = p.getBasePositionAndOrientation(self.robot)
        bodyPose = np.array(bodyPose_)
        bodyOri = np.array(Q2E(bodyOri_))
        vel_,angularRate_ = p.getBaseVelocity(self.robot)
        vel = np.array(vel_)
        angularRate = np.array(angularRate_)

        #Currently structured as addative rewards -> what about 1/costs?        
        sqrErrPose = np.sum((target[0:2] - bodyPose[0:2])**2) + bodyPose[2]**2 
        sqrErrVel = np.sum(vel[0:2]**2) + angularRate[2]**2

        fReward = 0
        if Fdict is not None:
            for F in Fdict.keys():
                #pReward += self.dictRewardCoeff["Collision"][P]*getPressureReward(dictPressure[P],maxP,maxCost)
                fReward += -15.0
                #TODO: 
                #define dictPressureThreshold for each body part
                #if dictPressure[P] > self.dictPressureThreshold[P]: done = True
                #done = True
                #if dictPressure[P] > maxP: done = True

        dictRew["Collision"] = fReward

        
        dictState["Distance"] = sqrErrPose

        dictRew["Position"] = self.dictRewardCoeff["Position"] * sqrErrPose
        dictRew["Velocity"] = self.dictRewardCoeff["Velocity"] * sqrErrVel

        if sqrErrPose < 1 and sqrErrVel < 0.2:
            done = True
            dictRew["Goal"] = self.dictRewardCoeff["Goal"]
        elif self.cnt > self.maxSteps:
            dictRew["Fail"] = self.dictRewardCoeff["Fail"]
            done = True
        elif self.training and self.inCollision:
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

    def setRecord(self,v=True,path='/Experiment_2/videos'):
        super().setRecord(v,path) #calls method from myEnv with the given args


class safetyEnv(humanEnv):
    def __init__(self, training=True, reward={}, maxSteps=2000, humans=1):
        '''
        Observation Space -> X,Y,thZ,vX,vY,vthZ,gX,gY,gthZ,hX,hY,hthZ,[bpX,bpY,bpZ]x20 joints [8+3h+3*20*h] (71 for h=1)
        Action Space -> vX,vY,vthZ [3]
        '''
        self.training = training
        if training:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.dictRewardCoeff = reward

        self.humans = int(humans)
        assert self.humans == 1, 'Error: incorrect number of humans provided'

        obs_shape = 8 + 3*self.humans + 3*20*self.humans

        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(obs_shape,), 
                                            dtype=np.float32
                                            )

        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(3,),
                                       dtype=np.float32
                                       )
        self.maxSteps = maxSteps

        self.record = False
        self.recorder = None

        self.inCollision = False

    def _setup(self):

        ## Initiate simulation
        self.client.resetSimulation()

        self.client.setAdditionalSearchPath(PATH_DATA)

        ## Set up simulation
        self.client.setTimeStep(TIME_STEP)
        self.client.setPhysicsEngineParameter(numSolverIterations=int(30))
        self.client.setPhysicsEngineParameter(enableConeFriction=0)
        self.client.setGravity(0, 0, -9.8)

        # creating environment
        self.models = setupWorld(self.client,self.humans) ## use function including human
        self.robot = self.models['robot']

        self.goal = self.models['goal']

        self.human = self.models['human']
        self.human.reset()

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

        self.control = ctlrRobot(self.robot)
        
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
        reward, done, dictLog = self._evaluate(ob)

        if self.record:
            if self.recorder == None:
                self._startRecorder()

            self._writeRecorder()

            if done:
                self._closeRecorder()

        return ob, reward, done, dictLog

    def _getObs(self):

        bodyPose, obsBodyOri = p.getBasePositionAndOrientation(self.robot)
        obsBodyPose = [bodyPose[i]/FIELD_RANGE for i in range(3)]
        obsVel,obsAngularRate = p.getBaseVelocity(self.robot)
        obsTarget = [self._getTargets()[i]/FIELD_RANGE for i in range(3)]
        humanPose,humanOri = p.getBasePositionAndOrientation(self.human.body_id)
        obsHumanPose = [humanPose[i]/FIELD_RANGE for i in range(3)] #TODO: add body parts? velocity?
        links = p.getLinkStates(self.human.body_id,self.linkList)
        obsLinks = [l_/FIELD_RANGE for l in links for l_ in list(l[0])]

        obs = np.concatenate((
            obsBodyPose[0:2], 
            Q2E(obsBodyOri)[2]/(2*PI), 
            obsVel[0:2], 
            obsAngularRate[2], 
            obsTarget[0:2],
            obsHumanPose[0:2],
            Q2E(humanOri)[2]/(2*PI),
            obsLinks
            ), axis=None)

        return obs

    def _evaluate(self,ob=None):

        done = False
        dictLog = {}
        dictRew = {}
        dictState = {}

        maxCost = 300 #good value??

        Fdict = self._getCollision()

        if ob is None:
            ob = self._getObs()

        bodyPose = FIELD_RANGE*np.array(ob[0:2])
        bodyOri = 2*PI*np.array(ob[2])
        bodyVel = np.array(ob[3:5])
        bodyAngularVel = np.array(ob[5])
        target = FIELD_RANGE*np.array(ob[6:8])

       
        sqrErrPose = np.sum((target - bodyPose)**2)
        sqrErrAng = bodyOri**2 
        sqrErrVel = np.sum(bodyVel**2)
        sqrErrAngVel = bodyAngularVel**2

        dictState["Position"] = sqrErrPose
        dictState["Angle"] = sqrErrAng
        dictState["Velocity"] = sqrErrVel
        dictState["AngularVelocity"] = sqrErrAngVel

        collisionReward = 0
        if Fdict is not None:
            for F in Fdict.keys():
                (rew,maxCostFlag) = getCollisionReward(Fdict,F,collisionDictTransient,collisionDictQuasiStatic,maxCost)
                collisionReward += self.dictRewardCoeff["Collision"][F] * rew
                if maxCostFlag : done = True

        dictRew["Position"] = self.dictRewardCoeff["Position"] * sqrErrPose
        dictRew["Angle"] = self.dictRewardCoeff["Angle"] * sqrErrAng
        dictRew["Velocity"] = self.dictRewardCoeff["Velocity"] * sqrErrVel
        dictRew["AngularVelocity"] = self.dictRewardCoeff["AngularVelocity"] * sqrErrAngVel
        dictRew["Collision"] = collisionReward

        if sqrErrPose < 0.5 and sqrErrVel < 0.2:
            done = True
            dictRew["Goal"] = self.dictRewardCoeff["Goal"]
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

    def setRecord(self,v=True,path='/Experiment_3/videos'):
        super().setRecord(v,path) #calls method from myEnv with the given args


if __name__ == "__main__": 
    env = safetyEnv(False,reward=rewardDict)
    obs = env.reset()
    sim_time = 2 # [s]
    steps = int(sim_time/TIME_STEP)
    act = [0,-1,0]
    env.setRecord(True)
    for _ in range(steps):
        ob, reward, done, dictLog = env.step(act) #[m/s]
        time.sleep(TIME_STEP/REPEAT_ACTION)





