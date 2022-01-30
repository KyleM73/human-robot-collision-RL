from setuptools import setup

#RUN pip install -e .  

setup(name='human_robot_collision_RL',
      version='0.1.0',
      install_requires=[
            'gym',
            'numpy',
            'opencv-python',
            'pybullet',
            'datetime',
            'stable_baselines3',
            'tensorflow'
            ] #And any other dependencies required
)