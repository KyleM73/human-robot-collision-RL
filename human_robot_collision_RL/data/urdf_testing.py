import pybullet as p
from pybullet import getQuaternionFromEuler as E2Q
import pybullet_utils.bullet_client as bc
import pybullet_data

import time

from man import Man

PI = 3.14159265359

## Initiate simulation
client = bc.BulletClient(connection_mode=p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
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
#walls = p.loadURDF("walls.urdf",basePosition=[0,-0.5,0.5],baseOrientation=E2Q([0,0,0]),useFixedBase=1)

## Make robot
#robot = p.loadURDF("trikey2.urdf",basePosition=[0,0,0.5],baseOrientation=E2Q([0,0,0]))

## Make man
sf = 1
human = Man(client._client,
            partitioned=False,
            self_collisions=False,
            pose=[0,0,sf*1.112],
            ori=[PI/2,0,0],
            fixed=1,
            timestep=1/240.,
            scaling=sf)


for _ in range(240*1000):
    client.stepSimulation()
    ctx = p.getContactPoints(ground,human.body_id)
    #print(ctx)
    #if ctx:
        #print(ctx)
        #print(p.getBasePositionAndOrientation(human.body_id))
        #break

    time.sleep(1/240.)
