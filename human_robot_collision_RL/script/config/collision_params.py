import numpy as np

from human_robot_collision_RL.script.constants import *

# Based on ISO/TS 15066 for 75 kg (165 lbs)
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
# EDIT : it does not appear to matter, results are constant over 3 orders of magnitude of spring constants for robot

eff_mass_robot = np.array([1])


eff_spring_const_robot = np.array([
    10  # Main Body
    ]) * 1e3

## TODO: update with ISO/TS 15066 values (couldn't find the paper online for free)
#EDIT: found semi legit numbers from Luis, still would be good to cross reference

#maximum force [N] before pain
# IMPACT force, not squeeze/clamp
collisionDictTransient = {
    "chest_to_belly"             :      160, #210 for chest
    "belly_to_pelvis"            :      250,
    "pelvis_to_right_leg"        :      250,
    "pelvis_to_left_leg"         :      250,
    "right_leg_to_right_shin"    :      170,
    "left_leg_to_left_shin"      :      170,
    "right_shin_to_right_foot"   :      160,
    "left_shin_to_left_foot"     :      160,
    "chest_to_right_arm"         :      190,
    "chest_to_left_arm"          :      190,
    "right_arm_to_right_forearm" :      220,
    "left_arm_to_left_forearm"   :      220,
    "right_forearm_to_right_hand":      180,
    "left_forearm_to_left_hand"  :      180,
    "chest_to_neck"              :       35, #190 for back of neck
    "neck_to_head"               :        0, #collision with the head is not allowed
    "right_foot_to_right_sole"   :      160,
    "left_foot_to_left_sole"     :      160,
    "right_sole_to_right_toes"   :      160,
    "left_sole_to_left_toes"     :      160
}

collisionDictQuasiStatic = {
    "chest_to_belly"             :      110, 
    "belly_to_pelvis"            :      180,
    "pelvis_to_right_leg"        :      180,
    "pelvis_to_left_leg"         :      180,
    "right_leg_to_right_shin"    :      140,
    "left_leg_to_left_shin"      :      140,
    "right_shin_to_right_foot"   :      125,
    "left_shin_to_left_foot"     :      125,
    "chest_to_right_arm"         :      150,
    "chest_to_left_arm"          :      150,
    "right_arm_to_right_forearm" :      160,
    "left_arm_to_left_forearm"   :      160,
    "right_forearm_to_right_hand":      135,
    "left_forearm_to_left_hand"  :      135,
    "chest_to_neck"              :       35, #145 for back of neck
    "neck_to_head"               :        0, #collision with the head is not allowed
    "right_foot_to_right_sole"   :      125,
    "left_foot_to_left_sole"     :      125,
    "right_sole_to_right_toes"   :      125,
    "left_sole_to_left_toes"     :      125
}