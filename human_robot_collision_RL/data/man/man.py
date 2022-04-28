import os

import pybullet as p

from human_robot_collision_RL.script.constants import *
from human_robot_collision_RL.data.human import Human


class Man(Human):
    def __init__(self,
                 pybtPhysicsClient,
                 partitioned=False,
                 self_collisions=False,
                 pose=HUMAN_POSE,
                 ori=HUMAN_ORI,
                 fixed=0,
                 timestep=0.01,
                 scaling=1.0):
        oriQ = p.getQuaternionFromEuler(ori)
        if partitioned:
            self.body_id = p.loadURDF(
                os.path.join(os.path.dirname(__file__),
                             "man_partitioned.urdf"),
                flags=p.URDF_MAINTAIN_LINK_ORDER,
                physicsClientId=pybtPhysicsClient,
                globalScaling=scaling,
                basePosition=pose,
                baseOrientation=oriQ,
                useFixedBase=fixed,
            )
        else:
            urdf_load_flags = p.URDF_MAINTAIN_LINK_ORDER
            if self_collisions:
                urdf_load_flags = p.URDF_MAINTAIN_LINK_ORDER | p.URDF_USE_SELF_COLLISION
            self.body_id = p.loadURDF(
                os.path.join(os.path.dirname(__file__),
                             "man.urdf"),
                flags=urdf_load_flags,
                physicsClientId=pybtPhysicsClient,
                globalScaling=scaling,
                basePosition=pose,
                baseOrientation=oriQ,
                useFixedBase=fixed,
            )

        super().__init__(
            pybtPhysicsClient,
            folder=os.path.dirname(__file__),
            timestep=timestep,
            scaling=scaling,
            translation_scaling=0.95,   #0.95 this is a calibration/scaling of the mocap velocities
        )
