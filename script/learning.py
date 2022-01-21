import pybullet as p
import numpy as np
import datetime
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from RLenv import myEnv
from constants import *

if __name__ == "__main__":
    env = myEnv(True)
    #num_cpu = 1
    #env.reset()

    #envVec = make_vec_env(env, n_envs=1)
    #env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200000)

    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if i%100==0: print(rewards)



