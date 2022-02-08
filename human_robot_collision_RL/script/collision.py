import pybullet as p

import numpy as np

from human_robot_collision_RL.script.constants import *

from human_robot_collision_RL.script.config.collision_params import *

class Collision:
    def __init__(
        self,
        client,
        robot,
        human,
        human_mass=M_HUMAN,
        robot_mass=M_ROBOT,
        height=H_ROBOT,
    ):
        self.client = client #pybullet client
        self.robot = robot #robot model
        self.human = human #human model
        self.human_mass = human_mass
        self.robot_mass = robot_mass
        self.height = height
        self.foot_list = [
        'right_foot_to_right_sole',
        'left_foot_to_left_sole',
        'right_sole_to_right_toes',
        'left_sole_to_left_toes',
        'right_shin_to_right_foot',
        'left_shin_to_left_foot'
        ]

    def get_collision_force(self,vel):
        """Get collision force in case of collision

        Returns
        -------
        None or ndarray
            In case of no collision, returns `None` otherwise returns an array containing contact forces and moments.
            Order of force and moments is [Fx, Fy, Fz, Mx, My, Mz]
        """
        contact_points = p.getContactPoints(
            self.human.body_id,
            self.robot
            )

        FmagDict = {}
        z = np.array([0,0,1])
        for contact_point in contact_points:
            if contact_point[8] <= 0:
                # Penetration or Contact

                human_part_id = contact_point[3]
                robot_part_id = contact_point[4]

                try:
                    human_part_id_name = p.getJointInfo(self.human.body_id,human_part_id)[1].decode('UTF-8')
                except:
                    print("Error: getJointInfo could not find id name")
                    continue
                
                F = self.get_force(robot_part_id,human_part_id,vel)

                n = -np.array(contact_point[7]) #normal FROM human TO robot
                th = np.arccos(np.dot(n,z)) - PI/2 #angle from normal to z axis in world coords
                if abs(th) < PI/6: #PI/4? 
                    #clamping down force
                    collision_type = 'quasi-static'
                else:
                    #head-on collision
                    collision_type = 'transient'


                FmagDict[human_part_id_name] = (F,collision_type)

        if FmagDict:
            return FmagDict
        return None

    def get_force(self,robot_part_id,human_part_id,vel):
        #https://ieeexplore-ieee-org.ezproxy.lib.utexas.edu/stamp/stamp.jsp?tp=&arnumber=8868390
        # quasi static contact model
        eff_mass_robot = self.__get_eff_mass_robot(robot_part_id)
        eff_mass_human = self.__get_eff_mass_human(human_part_id)
        mu = 1 / (1/eff_mass_robot + 1/eff_mass_human)

        k_robot = self.__get_eff_spring_const_human(robot_part_id)
        k_human = self.__get_eff_spring_const_human(human_part_id)

        eff_spring_const = 1 / (1/k_robot + 1/k_human)

        v = np.linalg.norm(vel)

        F = (eff_spring_const * mu * (v**2) )**0.5 #1/2 mu v^2 = 1/2 F^2 / k

        #https://www.nist.gov/system/files/documents/2019/11/22/2019-11-04%20IROS%202019%20-%20HRC%20Workshop%20-%20Behrens.pdf
        
        #quasi-static to transient contact multiplier ~ 2

        F_trans = 2*F

        return F_trans

    def __get_eff_mass_human(self, part_id):
        """Get effective human mass based on colliding part

        Parameters
        ----------
        part_id : int
            Part ID of colliding part

        Returns
        -------
        float
            Effective mass of colliding part of the human
        """
        return eff_mass_human[part_id] * self.human_mass

    def __get_eff_spring_const_human(self, part_id):
        """Get effective spring constant of the human based on colliding part

        Parameters
        ----------
        part_id : int
            Part ID of colliding part

        Returns
        -------
        float
            Effective spring constant of colliding part of the human
        """
        return eff_spring_const_human[part_id]

    def __get_eff_spring_const_robot(self, part_id):
        """Get effective spring constant of the robot based on colliding part

        Parameters
        ----------
        part_id : int
            Part ID of colliding part

        Returns
        -------
        float
            Effective spring constant of colliding part of the robot
        """
        return eff_spring_const_robot[part_id]

    def __get_eff_mass_robot(self, part_id):
        """Get effective robot mass based on colliding part

        Parameters
        ----------
        part_id : int
            Part ID of colliding part

        Returns
        -------
        float
            Effective mass of colliding part of the robot
        """
        return eff_mass_robot[part_id] * self.robot_mass













