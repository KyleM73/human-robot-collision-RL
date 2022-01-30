import sys

from stable_baselines3 import PPO

from human_robot_collision_RL.script.RLenv import myEnv, humanEnv
from human_robot_collision_RL.script.config.rewards import rewardDict
from human_robot_collision_RL.script.constants import *

def evaluate(model_path,exp_num=1):
    envs = [myEnv,humanEnv]
    envTest = envs[exp_num-1](True,rewardDict)
    modelTest = PPO.load(model_path, env=envTest)

    obs = envTest.reset()
    envTest.setRecord(True)
    
    for i in range(2000):
        action, _states = modelTest.predict(obs)
        obs, rewards, dones, info = envTest.step(action)
        envTest.render()

    print()
    print("pose:   ",[FIELD_RANGE*obs[i] for i in range(3)])
    print("heading:",[2*PI*obs[2]])
    print("vel:    ",obs[3:6])
    print("target: ",[FIELD_RANGE*obs[6+i] for i in range(2)])

def evaluateMain():
    model_path = './human_robot_collision_RL/save/Experiment_2/models/'
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
        evaluate(model_path)
    except:
        print('no valid model provided')

if __name__=="__main__":
    evaluateMain()

    

