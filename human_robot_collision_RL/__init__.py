from gym.envs.registration import register
from human_robot_collision_RL.script.config.rewards import rewardDict
from human_robot_collision_RL.script.constants import MAX_STEPS

register(
    id='simple-v0',
    entry_point='human_robot_collision_RL.script:myEnv',
    kwargs={
        'training':True,
        'reward':rewardDict,
        'maxSteps':MAX_STEPS
        },
    )

register(
    id='human-v0',
    entry_point='human_robot_collision_RL.script:humanEnv',
    kwargs={
        'training':True,
        'reward':rewardDict,
        'maxSteps':MAX_STEPS
        },
    )

register(
    id='safety-v0',
    entry_point='human_robot_collision_RL.script:safetyEnv',
    kwargs={
        'training':True,
        'reward':rewardDict,
        'maxSteps':MAX_STEPS
        },
    )


"""
THE HUMAN MODELS USED IN THIS SIMULATION WERE PRODUCED BY EPFL
SEE https://github.com/epfl-lasa/human-robot-collider

THEIR CODE IS PROTECTED UNDER THE FOLLOWING LICENSE

MIT License

Copyright (c) 2019 LASA Laboratory, EPFL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""