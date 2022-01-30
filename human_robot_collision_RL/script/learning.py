import pybullet as p
import numpy as np
import datetime
import time
import gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from human_robot_collision_RL.script.RLenv import myEnv, humanEnv
from human_robot_collision_RL.script.evaluate import evaluate
from human_robot_collision_RL.script.config.rewards import rewardDict
from human_robot_collision_RL.script.constants import *


def makeEnvs(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

env_ids = ['simple-v0','human-v0'] # env ids are registered in outermost __init__.py

if __name__ == "__main__":

    experiment_num = 1

    ## define path for saved model
    save_folder = PATH_SAVE+'/Experiment_'+str(experiment_num)+'/models'
    save_path = '{}/{}'.format(save_folder,datetime.datetime.now().strftime('%m%d_%H%M'))

    ## define path for tensorboard logging
    tb_log_path = PATH_LOG+'/Experiment_'+str(experiment_num)
    tb_log_subpath = '{}'.format(datetime.datetime.now().strftime('%m%d_%H%M'))

    ## set the environment params
    env_id = env_ids[experiment_num-1]
    num_cpu = 6 #machine dependant 

    ## make parallel environments
    env = SubprocVecEnv([makeEnvs(env_id) for i in range(num_cpu)]) #env.env_method(method_name='setRecord')

    ## train model
    model = PPO("MlpPolicy", env, verbose=1,tensorboard_log=tb_log_path) #TODO: add custom policy args
    model.learn(total_timesteps=500_000,tb_log_name=tb_log_subpath)

    ## save the model
    model.save(save_path)
    del model  # delete trained model to demonstrate loading

    
    ## evaluate the model
    evaluate(save_path,experiment_num)

    
    



