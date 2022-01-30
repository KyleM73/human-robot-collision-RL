if __name__=="__main__":
    try:
        import gym
        import numpy
        import cv2
        import pybullet
        import pybullet_data
        import pybullet_utils
        import time
        import datetime
        import sys
        import os
        import stable_baselines3
        import human_robot_collision_RL.script.RLenv
        import human_robot_collision_RL.data.man
        print("SETUP PASSED")
    except:
        print("ERROR: SETUP FAILED")
