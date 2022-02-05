import os
import pybullet as p
import numpy as np
import datetime
import time
import gym
import shutil

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from human_robot_collision_RL.script.RLenv import myEnv, humanEnv, safetyEnv
from human_robot_collision_RL.script.evaluate import evaluate
from human_robot_collision_RL.script.config.rewards import rewardDict
from human_robot_collision_RL.script.constants import *


def makeEnvs(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

def copy_reward_gains(path):
    reward_path = r'{}/{}'.format(PATH_CONFIG,'rewards.py')
    new_reward_path = r'{}/{}'.format(path,'rewards.py')

    collision_params_path = r'{}/{}'.format(PATH_CONFIG,'collision_params.py')
    new_collision_params_path = r'{}/{}'.format(path,'collision_params.py')

    os.makedirs(os.path.dirname(new_reward_path), exist_ok=True)
    os.makedirs(os.path.dirname(new_collision_params_path), exist_ok=True)

    shutil.copyfile(reward_path, new_reward_path)
    shutil.copyfile(collision_params_path, new_collision_params_path)


env_ids = ['simple-v0','human-v0','safety-v0'] # env ids are registered in outermost __init__.py

if __name__ == "__main__":

    dt = datetime.datetime.now().strftime('%m%d_%H%M')

    experiment_num = 3

    ## define path for saved model
    save_folder = PATH_SAVE+'/Experiment_'+str(experiment_num)+'/models'
    save_path = '{}/{}'.format(save_folder,dt)

    ## define path for tensorboard logging
    tb_log_path = PATH_LOG+'/Experiment_'+str(experiment_num)
    tb_log_subpath = '{}'.format(dt)
    log_path_full = '{}/{}_1'.format(tb_log_path,tb_log_subpath)

    ## set the environment params
    env_id = env_ids[experiment_num-1]
    num_cpu = 6 #machine dependant 

    ## make parallel environments
    env = SubprocVecEnv([makeEnvs(env_id) for i in range(num_cpu)]) #env.env_method(method_name='setRecord')

    ## train model
    model = PPO("MlpPolicy", env, verbose=1,tensorboard_log=tb_log_path) #TODO: add custom policy args
    model.learn(total_timesteps=TRAIN_STEPS,tb_log_name=tb_log_subpath)

    ## save the model
    model.save(save_path)
    del model  # delete trained model to demonstrate loading
    copy_reward_gains(log_path_full)
    
    ## evaluate the model
    evaluate(save_path,experiment_num)

    
    



