import pybullet as p
from pybullet import getQuaternionFromEuler as E2Q
import pybullet_utils.bullet_client as bc
import pybullet_data

import time

from human_robot_collision_RL.script.util import *

from man import Man

PI = 3.14159265359

## Initiate simulation
client = bc.BulletClient(connection_mode=p.GUI)
#p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
client.resetSimulation()

## Set up simulation
client.setTimeStep(1/240.)
client.setPhysicsEngineParameter(numSolverIterations=int(30))
client.setPhysicsEngineParameter(enableConeFriction=0)
client.setGravity(0,0,-9.8)

## Make ground
plane = p.createCollisionShape(shapeType = p.GEOM_PLANE)
ground  = p.createMultiBody(0, plane)
p.changeDynamics(ground, -1, lateralFriction=1.0)

## Make walls
walls = p.loadURDF("walls.urdf",basePosition=[0,-0.5,0.5],baseOrientation=E2Q([0,0,0]),useFixedBase=1)

## Make robot
robot = p.loadURDF("trikey2.urdf",basePosition=[0,0,0.5],baseOrientation=E2Q([0,0,0]))

## Make man
sf = 1
human = Man(client._client,
            partitioned=False,
            self_collisions=False,
            pose=[0,4,sf*1.112],
            ori=[PI/2,0,0],
            fixed=1,
            timestep=1/240.,
            scaling=sf)


def render(*args):
    try:
        bodyPose, obsBodyOri = p.getBasePositionAndOrientation(robot)
    except:
        bodyPose = np.zeros(3)
        obsBodyOri = [0,0,0,1]
    ## Camera configuration ##

    CAM_DIST = 10.0
    CAM_YAW = p.getEulerFromQuaternion(obsBodyOri)[-1]
    CAM_PITCH = -30
    CAM_WIDTH = 480
    CAM_HEIGHT = 360
    CAM_FPS = 60
    CAM_FOV = 60
    CAM_NEARVAL = 0.1
    CAM_FARVAL = 100.0
    

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[bodyPose[0], bodyPose[1], bodyPose[2]],
        cameraTargetPosition=[0,10,0.5],
        cameraUpVector=[0,0,1])
    projMatrix = p.computeProjectionMatrixFOV(
        fov=CAM_FOV,
        aspect=float(CAM_WIDTH) / CAM_HEIGHT,
        nearVal=CAM_NEARVAL,
        farVal=CAM_FARVAL)
    (_, _, px, _, _) = p.getCameraImage(
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        renderer=p.ER_TINY_RENDERER,#ER_BULLET_HARDWARE_OPENGL,
        viewMatrix=viewMatrix,
        projectionMatrix=projMatrix)
    rgbArray = np.array(px)
        #preImage = rgbArray[:, :, :3]
        #image = preImage[:]
        #RGB to BGR
        #image[:,:,0] = preImage[:,:,2]
        #image[:,:,1] = preImage[:,:,1]
        #image[:,:,2] = preImage[:,:,0]
        #TODO colors are still weird
        
    return rgba2rgb(rgbArray)


for _ in range(240*1000):
    client.stepSimulation()
    render()
    ctx = p.getContactPoints(ground,human.body_id)
    #print(ctx)
    #if ctx:
        #print(ctx)
        #print(p.getBasePositionAndOrientation(human.body_id))
        #break

    time.sleep(1/240.)
