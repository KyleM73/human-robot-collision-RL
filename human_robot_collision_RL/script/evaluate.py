import sys

import pybullet as p

from stable_baselines3 import PPO

from human_robot_collision_RL.script.constants import *
from human_robot_collision_RL.script.RLenv import myEnv, humanEnv, safetyEnv
from human_robot_collision_RL.script.util import *

from human_robot_collision_RL.script.config.rewards import rewardDict

def evaluate(model_path,exp_num=EXP_NUM):

    print()
    print('EVALUATING MODEL...')
    print()

    envs = [myEnv,humanEnv,safetyEnv]
    envTest = envs[exp_num-1](True,rewardDict)
    modelTest = PPO.load(model_path, env=envTest)

    obs = envTest.reset()

    envTest.setRecord(True,exp_num)
    
    for i in range(MAX_STEPS):
        action, _states = modelTest.predict(obs)
        obs, rewards, dones, info = envTest.step(action)
        #if dones:
        #    break
        envTest.render()
        if i % 200 == 0:
            print("progress...    ",100*i/MAX_STEPS,"%")

    print()
    print("pose:   ",[FIELD_RANGE*obs[i] for i in range(3)])
    print("heading:",[2*PI*obs[2]])
    print("vel:    ",obs[3:6])
    print("target: ",[FIELD_RANGE*obs[6+i] for i in range(2)])

def evaluateMain(experiment_num=EXP_NUM):
    model_path = PATH_SAVE+'/Experiment_'+str(experiment_num)+'/models/'
    args = sys.argv
    if len(args) == 1:
        print('not enough arguments passed')
        model = str(input('enter the name of the model you would like to run:\n'+model_path))
    elif len(args) > 2:
        print('too many arguments passed')
        model = srtr(input('enter the name of the model you would like to run:\n'+model_path))
    else:
        model = str(args[1])
    try:
        model_path += model
        print(model_path)
        evaluate(model_path,experiment_num)
    except:
        print('no valid model provided')
    

if __name__=="__main__":
    evaluateMain(EXP_NUM)

    

