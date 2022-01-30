# USER GUIDE

## SECTIONS
#### SETUP
#### RUNNING THE CODE
#### DIRECTORY STRUCTURE

## SETUP

**this module was tested with python 3.9.7**

- if using a virtul env (conda, venv, etc.), create an environment with python>=3.8. to check python version, in the terminal run

    `python -V`
    and
    `python3 -V`

    if the result is not greater than 3.8.x, then everywhere where this guide says 'python,' replace with 'python3.x' for x = 8 or x = 9.

- from outermost folder (/human_robot_collision_RL), run

    `pip install -e .`

this will build the module. make sure that the libraries in setup.py are installed in the terminal. to verify, run

`python setup_test.py`
the result should be 'SETUP PASSED'

## RUNNING THE CODE

to test the simulation environment  : see script/RLenv.py
to train the model                  : see script/learning.py
to evaluate the model               : see script/evaluate.py
to view model evaluation online     : see save/Experiment_*/videos
to change reward parameters         : see script/config/rewards.py
to change reward function           : see script/RLenv.($CLASS)._evaluate()


after calling script/learning.py, you will see the following line:
'Logging to ./human_robot_collision_RL/log/Experiment_*/****_****_*'
after learning is complete (or simultaneously in a separate terminal),
RUN 'tensorboard --logdir ./human_robot_collision_RL/log/Experiment_*/****_****_*'
this command will give you a host to connect to via a browser
example: 'localhost:6006/'
COPY and PASTE this address into a web browser to access tensorboard logging


## DIRECTORY STRUCTURE

    /human_robot_collision_RL
        /setup.py
        /setup_test.py
        /README.md
        /RL_TODO.txt
        /human_robot_collision_RL.egg-info
            /dependency_links.txt
            /PKG-INFO
            /requires.txt
            /SOURCES.txt
            /top_level.txt
        /human_robot_collision_RL
            /__init__.py
            /data
                /__init__.py
                /trikey2.urdf
                /table.png
                /constrainTest.py
                /human.py
                /man
                    /__init__.py
                    /man.urdf
                    /man_partitioned.urdf
                    /man.py
                    /stl/*
                    /walk/*
                    /xarco/*
                /child
                    /__init__.py
                    /child.urdf
                    /child.py
                    /stl/*
                    /walk/*
                    /xarco/*
            /log
                /Experiment_1/*
                /Experiment_2/*             
            /save
                /__init__.py
                /Experiment_1
                    /models
                    /videos
                /Experiment_2
                    /models
                    /videos
            /script
                /__init__.py
                /RLenv.py
                /learning.py
                /evaluate.py
                /control.py
                /collision.py
                /constants.py
                /config
                    /__init__.py
                    /rewards.py

