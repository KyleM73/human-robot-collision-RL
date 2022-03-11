import sys

import pybullet as p

from stable_baselines3 import PPO

from human_robot_collision_RL.script.constants import *
from human_robot_collision_RL.script.RLenv import myEnv, humanEnv, safetyEnv
from human_robot_collision_RL.script.util import *

from human_robot_collision_RL.script.config.rewards import rewardDict

def evaluate(model_name,log_save_path=None,exp_num=EXP_NUM):
    model_path = PATH_LOG+'/Experiment_'+str(exp_num)+'/'+model_name+'_1/'
    path = model_path+model_name

    print()
    print('EVALUATING MODEL...')
    print()

    envs = [myEnv,humanEnv,safetyEnv]
    envTest = envs[exp_num-1](True,rewardDict,MAX_STEPS,NUM_HUMANS)
    modelTest = PPO.load(path, env=envTest)

    obs = envTest.reset()

    if log_save_path is not None:
        envTest.setRecord(True,log_save_path,exp_num)
    else:
        envTest.setRecord(True,model_path,exp_num)
    
    for i in range(MAX_STEPS):
        action, _states = modelTest.predict(obs)
        obs, rewards, dones, info = envTest.step(action)
        envTest.render()
        if dones:
            print("'done' condition hit")
            break
        if i % 200 == 0:
            print("progress...    ",100*i/MAX_STEPS,"%")
        elif i == MAX_STEPS-1:
            print("progress...     100 %")

    print()
    '''
    print("pose:   ",[FIELD_RANGE*obs[i] for i in range(3)])
    print("heading:",[2*PI*obs[2]])
    print("vel:    ",obs[3:6])
    print("target: ",[FIELD_RANGE*obs[6+i] for i in range(2)])
    '''

def evaluateLive(model_name,exp_num=EXP_NUM):
    model_path = PATH_LOG+'/Experiment_'+str(exp_num)+'/'+model_name+'_1/'
    path = model_path+model_name

    print()
    print('EVALUATING MODEL...')
    print()

    envs = [myEnv,humanEnv,safetyEnv]
    envTest = envs[exp_num-1](False,rewardDict)
    modelTest = PPO.load(path, env=envTest)

    obs = envTest.reset()
    
    for i in range(MAX_STEPS):
        action, _states = modelTest.predict(obs)
        obs, rewards, dones, info = envTest.step(action)
        envTest.render()
        if dones:
            print("'done' condition hit")
            break

    '''
    print()
    print("pose:   ",[FIELD_RANGE*obs[i] for i in range(3)])
    print("heading:",[2*PI*obs[2]])
    print("vel:    ",obs[3:6])
    print("target: ",[FIELD_RANGE*obs[6+i] for i in range(2)])
    '''


def evaluateMain(experiment_num=EXP_NUM):
    live = False
    args = sys.argv
    if len(args) == 1:
        print('not enough arguments passed')
        model = str(input('enter the name of the model you would like to run:\n'+model_path))
    elif len(args) > 3:
        print('too many arguments passed')
        model = srtr(input('enter the name of the model you would like to run:\n'+model_path))
    else:
        model = str(args[1])
        if len(args) == 3:
            if str(args[2]) == 'live':
                live = True
    if live:
        evaluateLive(model,exp_num=experiment_num)
    else:
        evaluate(model,exp_num=experiment_num)

if __name__=="__main__":
    evaluateMain(EXP_NUM)


    

