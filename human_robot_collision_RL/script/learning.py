import os

import pybullet as p

import numpy as np
import datetime
import time

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from human_robot_collision_RL.script.constants import *
from human_robot_collision_RL.script.evaluate import evaluate
from human_robot_collision_RL.script.RLenv import myEnv, humanEnv, safetyEnv
from human_robot_collision_RL.script.util import *

from human_robot_collision_RL.script.config.rewards import rewardDict


def makeEnvs(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return _init


if __name__ == "__main__":

    ## define path for saved model
    #save_folder = PATH_SAVE+'/Experiment_'+str(EXP_NUM)+'/models'
    #save_path = '{}/{}'.format(save_folder,DT)

    ## define path for tensorboard logging
    tb_log_path = PATH_LOG+'/'+str(EXP_NAME)
    tb_log_subpath = '{}'.format(DT)
    log_path_full = '{}/{}_1'.format(tb_log_path,tb_log_subpath)

    ## set the environment params
    env_id = 'safety-v0'#ENV_IDS[EXP_NUM-1]
    num_cpu = CPU_NUM #machine dependent 

    ## evaluation env
    eval_env = gym.make(env_id)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path_full,log_path=log_path_full)

    ## make parallel environments
    env = SubprocVecEnv([makeEnvs(env_id) for i in range(num_cpu)],start_method='fork') #env.env_method(method_name='setRecord')

    ## train model
    model = PPO("MlpPolicy", env, policy_kwargs=POLICY_KWARGS,verbose=1,tensorboard_log=tb_log_path)
    startTime = time.time()
    model.learn(total_timesteps=TRAIN_STEPS,tb_log_name=tb_log_subpath,callback=eval_callback)
    endTime = time.time()
    print("Model train time: "+str(datetime.timedelta(seconds=endTime-startTime)))

    ## save the model
    save_path = '{}/{}'.format(log_path_full,DT)
    model.save(save_path)
    del model  # delete trained model to demonstrate loading

    ## save relevant files with hyperparams
    copy_reward_gains(log_path_full) #be careful if training is stopped
    
    
    ## evaluate the model
    #evaluate(DT,log_path_full,EXP_NUM)

    
    



