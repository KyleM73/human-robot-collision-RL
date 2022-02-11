colDict = {
    "chest_to_belly"             :     -0.05,
    "belly_to_pelvis"            :     -0.05,
    "pelvis_to_right_leg"        :     -0.05,
    "pelvis_to_left_leg"         :     -0.05,
    "right_leg_to_right_shin"    :     -0.05,
    "left_leg_to_left_shin"      :     -0.05,
    "right_shin_to_right_foot"   :     -0.05,
    "left_shin_to_left_foot"     :     -0.05,
    "chest_to_right_arm"         :     -0.05,
    "chest_to_left_arm"          :     -0.05,
    "right_arm_to_right_forearm" :     -0.05,
    "left_arm_to_left_forearm"   :     -0.05,
    "right_forearm_to_right_hand":     -0.05,
    "left_forearm_to_left_hand"  :     -0.05,
    "chest_to_neck"              :     -0.05,
    "neck_to_head"               :     -0.05,
    "right_foot_to_right_sole"   :     -0.05,
    "left_foot_to_left_sole"     :     -0.05,
    "right_sole_to_right_toes"   :     -0.05,
    "left_sole_to_left_toes"     :     -0.05,
}

rewardDict = {
    "Position"        :     10,#-0.02
    "Angle"           :  -0.01,
    "Velocity"        : -0.001,
    "AngularVelocity" :  -0.01,
    "Fail"            :    -10,
    "Goal"            :     50,
    "ActionSmooth"    :      0,
    "ActionNotSmooth" :      0,
    "Collision"       : colDict
}

max_cost = 300 #good value??
vel_penalty_radius = 2
pose_radius = 1
vel_radius = 0.5

def getCollisionReward(F,partName,transDict,quasiStaticDict,maxCost):
    if F[partName][1] == "quasi-static":
        if F[partName][0]/2 > quasiStaticDict[partName]: #F was multiplied by 2 to get transient force, see collision.get_force()
            return (maxCost,True)
        else:
            return (F[partName][0]/2,False)
    elif F[partName][1] == "transient":
        if F[partName][0] > transDict[partName]:
            return (maxCost,True)
        else:
            return (F[partName][0],False)
    else:
        print("Error: bad collision type")
        return (maxCost,True)









