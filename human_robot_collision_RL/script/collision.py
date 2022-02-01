from human_robot_collision_RL.script.constants import *

import numpy as np
import pybullet as p
import time

# Based on ISO/TS 15066 for 75 kg
eff_mass_human = np.array([
    40,        # chest
    40,        # belly
    40,        # pelvis
    75, 75,    # upper legs    <-- Previous implementation was different
    75, 75,    # shins         <-- Previous implementation was different
    75, 75,    # ankles/feet (Same as shin)    <-- Previous implementation was different
    3, 3,      # upper arms
    2, 2,      # forearms
    0.6, 0.6,  # hands
    1.2,       # neck
    8.8,       # head + face (?)
    75, 75,    # soles (Same as shin)  <-- Previous implementation was different
    75, 75,    # toes (Same as shin)   <-- Previous implementation was different
    40,        # chest (back)
    40,        # belly (back)
    40,        # pelvis (back)
    1.2,       # neck (back)
    4.4,       # head (back)
]) / M_HUMAN


eff_spring_const_human = np.array([
    25,      # chest
    10,      # belly
    25,      # pelvis
    50, 50,  # upper legs
    60, 60,  # shins
    75, 75,  # ankles/feet (Same as shin)    <-- From Previous implementation
    30, 30,  # upper arms    <-- Previous implementation was different
    40, 40,  # forearms      <-- Previous implementation was different
    75, 75,  # hands
    50,      # neck
    150,     # head (face ?)
    75, 75,  # soles (Same as shin)  <-- From Previous implementation
    75, 75,  # toes (Same as shin)   <-- From Previous implementation
    35,      # chest (back)
    35,      # belly (back)
    35,      # pelvis (back)
    50,      # neck (back)
    150,     # head (back)
]) * 1e3


# eff_mass_robot = np.array([
#     50,   # Main Body
# ]) / M_ROBOT

## TODO: validate this spring constant for the robot exterior

eff_mass_robot = np.array([1])


eff_spring_const_robot = np.array([
    10  # Main Body
    ]) * 1e3

class Collision:
    def __init__(
        self,
        pybtPhysicsClient,
        robot,
        human,
        human_mass=M_HUMAN,
        robot_mass=M_ROBOT,
        height=H_ROBOT,
        ftsensor_loc=[0.0, 0.0],#0.035,0 (not sure why)
        timestep=TIME_STEP,
    ):
        self.pybtPhysicsClient = pybtPhysicsClient #pybullet client
        self.robot = robot #robot model
        self.human = human #human model
        self.human_mass = human_mass
        self.robot_mass = robot_mass
        self.height = height
        self.ftsensor_loc = ftsensor_loc #force-torque sensor location
        self.timestep = timestep

    def get_collision_force(self):
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

        z = np.array([0,0,1])
        FmagDict = {}
        for contact_point in contact_points:
            if contact_point[8] <= 0:
                # Penetration or Contact
                human_part_id = contact_point[3]
                robot_part_id = contact_point[4]
                pos_on_robot = contact_point[6] #this is sometimes greater than max height of robot? #TODO figure out why
                if pos_on_robot[2] > H_ROBOT : pos_on_robot = np.array(pos_on_robot) - np.array([0,0,H_ROBOT])
                n = -np.array(contact_point[7]) #normal FROM body TO robot
                area = self._getArea(pos_on_robot,n,self.human.body_id,human_part_id)
                if area < 0.0001 or contact_point[9] < 0.0001: #omit sufficently small areas that would create unboundedly large pressures
                    continue
                
                #EPFL-LASA collision method - assumes spring model
                #self.__collide(robot_part_id, human_part_id)
                #Fmag = self.__get_contact_force(contact_point[8])
                #(h, theta) = self.__get_loc_of_collision(pos_on_robot)
                #self.delta_v, self.delta_omega = self.collision_dynamics(pos_on_robot, Fmag, theta)
                #return (
                #    Fmag * np.sin(theta),
                #    Fmag * np.cos(theta),
                #    0,
                #    -Fmag * np.cos(theta) * h,
                #    Fmag * np.sin(theta) * h,
                #    0,
                #)

                human_part_id_name = p.getJointInfo(self.human.body_id,human_part_id)[1].decode('UTF-8')
                FmagDict[human_part_id_name] = (contact_point[9],area)

        if FmagDict:
            return FmagDict
        return None


    def _getArea(self,pos,normal,human_id,link_id,ref=np.array([0,0,1]),delta=0.00001,stepWidth=0.001):
        #see notes
        #stepWidth 0.05 -> 170 hits vs. 0.01 -> 3500+ hits

        #get orthonormal basis vectors of n on the plane to which n is normal
        z = ref
        b1 = z - np.dot(z,normal) # "up"
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(b1,normal) # "right"
        b2 /= np.linalg.norm(b2)
        #basis = [n,b1,b2] #orthonormal basis of n

        #check angle between z and b1
        th = np.arccos(np.dot(z,b1))
        if abs(th) > PI/4:
            print("err: bad normal")
            return 0

        #take a delta-sized step in the direction opposite the normal vector from the point of collision
        collisionPointWithTolerance = pos - delta*normal

        zz = self.height - collisionPointWithTolerance[2]
        zDotb1 = np.dot(z,b1) # projection of z onto b1
        #thZ = np.arccos(zDotb1) #probably redundant
        #dz = float(zz/np.cos(thZ)) 
        dz = float(zz/zDotb1) #distance "up" the plane

        #dx = R_ROBOT#*2 #edge case if collision is exactly on side (omited *2 due to ray max batch size constraint)
        #dx = float((xx**2 - delta**2)**0.5) #almost equivalent to above because  0 < delta << 1
        #dx = 0.1 #10 cm max horiz collision area with curved surface? see below
        dx = (R_ROBOT**2 - (R_ROBOT-delta)**2)**0.5 #~0.07 for delta=0.01

        #split range into stepWidth-sized steps
        zRange = int(np.ceil(dz/stepWidth))
        xRange = int(np.ceil(dx/stepWidth))

        #sweep through range top-to-bottom
        #mark points for ray testing if they are above ground
        #TODO: do not include points that could not be hit by either body
        #   if successful, decrease stepWidth to 0.01 
        pt = collisionPointWithTolerance
        toPos = []
        fromPos = []
        toPosReverse = []
        for xStep in range(-xRange,xRange):
            for zStep in range(-zRange,zRange):
                testPt = pt - stepWidth*(zStep*b1+xStep*b2) #switch the order: top to bottom
                #testPt = pt - stepWidth*(zStep*z+xStep*np.array([1,0,0])) #plane is up and out
                testPtReverse = testPt+delta*normal
                if testPtReverse[2] < 0 or testPt[2] < 0: #top to bottom means we can stop searching when we hit zero
                    break
                fromPos.append(pos)
                toPos.append(testPt)
                toPosReverse.append(testPtReverse)
        
        #omit rays from collision point to collision plane
        try:
            hits = p.rayTestBatch(fromPos,toPos)
        except:
            print("no hits")
            return 0
        area = 0
        for h in hits:
            if h[0] == human_id and h[1] == link_id: 
                area += stepWidth**2
        
        '''
        try:
            #mark rays between delta-advanced collision plane and collision plane
            hitsReverse = p.rayTestBatch(toPos,toPosReverse)
        except:
            return 0

        areaR = 0
        for hr in hitsReverse:
            #if hr[0] == self.robot: # and hr[1] == robot_link_id: 
            if h[0] == human_id and h[1] == link_id: 
                areaR += stepWidth**2
        '''

        #print(4*zRange*xRange*stepWidth**2) #max possible area

        #draw collision plane (for debugging with RLenv.py)
        #self.markArea(dx,delta,dz,normal,pos)

        return area

    def markArea(self,l,w,h,n,pos):
        #TODO: despawn multibodies after X timesteps
        idVisualShape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[l/2, w, h/2],
        visualFrameOrientation=p.getQuaternionFromEuler(n)
        )

        idCollisionShape = None

        boxModel = p.createMultiBody(
        baseVisualShapeIndex=idVisualShape, 
        basePosition=pos
        )

        return boxModel

    ## NONE OF THESE FUNCTIONS ARE CURRENTLY USED BUT MAY BE IN THE FUTURE
    #   |   |   | 
    #   v   v   v 


    def collision_dynamics(self, pos, Fmag, theta):
        Vx = Fmag * np.sin(theta) * self.timestep / self.robot_mass
        Vy = Fmag * np.cos(theta) * self.timestep / self.robot_mass
        return self.__cartesian_to_differential(pos, Vx, Vy)

    def __collide(self, robot_part_id, human_part_id):
        #TODO: return eff spring const instead of setting as class var since multiple collisions may occur on the same timestep
        self.eff_mass_robot = self.__get_eff_mass_robot(robot_part_id)
        self.eff_mass_human = self.__get_eff_mass_human(human_part_id)

        k_robot = self.__get_eff_spring_const_human(robot_part_id)
        k_human = self.__get_eff_spring_const_human(human_part_id)

        self.eff_spring_const = 1 / (1/k_robot + 1/k_human)

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

    def __get_contact_force(self, penetration):
        """Get contact force based on penetration

        Parameters
        ----------
        penetration : float
            Penetration of robot into human

        Returns
        -------
        float
            Effective Contact Force
        """
        if penetration > 0:
            # No Contact
            return 0
        else:
            return (-self.eff_spring_const * penetration)

    def __get_loc_of_collision(self, pos):
        # why switch indices? x-y?
        #theta = np.arctan2(pos[0] - self.ftsensor_loc[1], - pos[1] - self.ftsensor_loc[0])
        theta = np.arctan2(pos[0], - pos[1])
        h = self.height - pos[2]
        return (h, theta)

    def __cartesian_to_differential(self, pos, vx, vy):
        pt = (-pos[1], pos[0])
        self.inv_jacobian = np.array([
            [pt[1]/pt[0], 1.],
            [-1./pt[0], 0.],
        ])
        return (self.inv_jacobian @ np.array([vx, vy]))










