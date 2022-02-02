colDict = {
    "chest_to_belly"             :     -0.1,
    "belly_to_pelvis"            :     -0.1,
    "pelvis_to_right_leg"        :     -0.1,
    "pelvis_to_left_leg"         :     -0.1,
    "right_leg_to_right_shin"    :     -0.1,
    "left_leg_to_left_shin"      :     -0.1,
    "right_shin_to_right_foot"   :     -0.1,
    "left_shin_to_left_foot"     :     -0.1,
    "chest_to_right_arm"         :     -0.1,
    "chest_to_left_arm"          :     -0.1,
    "right_arm_to_right_forearm" :     -0.1,
    "left_arm_to_left_forearm"   :     -0.1,
    "right_forearm_to_right_hand":     -0.1,
    "left_forearm_to_left_hand"  :     -0.1,
    "chest_to_neck"              :     -0.1,
    "neck_to_head"               :     -0.1,
    "right_foot_to_right_sole"   :     -0.1,
    "left_foot_to_left_sole"     :     -0.1,
    "right_sole_to_right_toes"   :     -0.1,
    "left_sole_to_left_toes"     :     -0.1
}

rewardDict = {
    "Position"        :  -0.04,
    "Angle"           :  -0.01,
    "Velocity"        : -0.001,
    "AngularVelocity" :  -0.01,
    "Fail"            :    -10,
    "Goal"            :     10,
    "Collision"       : colDict
}

def getPressureReward(P,maxP=100,maxCost=150):
    return maxCost
    #if P < maxP:
    #    return P
    #return maxCost