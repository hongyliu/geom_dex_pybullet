import pybullet as p
import pybullet_data
import os
import numpy as np
from train_classification import parse_args, makeEnv
from numpy.random import RandomState
import gym
from shadowhand_gym.envs.config import *

args = parse_args()
envs = [gym.make('ShadowHandBlock-v1', object=name, render=True) for name in ['YcbTomatoSoupCan']]
for env in envs:
    env.reset()
    done = False
    # env.get_point_cloud('object', 1024, RandomState(42))
    while True:
        action = np.random.uniform(-1., 1., (1, 20))
        obs, reward, done, info = env.step(action)
        print(env.robot.get_obs())
        p.stepSimulation()
p.disconnect()





# cheetahDIR = os.path.dirname(__file__)
#
# physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
# p.setGravity(0, 0, -9.81)
#
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# planeId = p.loadURDF("plane.urdf")
#
# file = 'shadowhand_gym/envs/assets/toilet/102621/102621.urdf'
# filename= '/mnt/data/gibson_challenge_data_2021/ig_dataset/objects/sink/sink_6/sink_6.urdf'
# # filename = os.path.join(igibson.ig_dataset_path, "objects", "bathtub", 'e34833b19879020d54d7082b34825ef0', 'shape', 'collision', "e34833b19879020d54d7082b34825ef0_cm.obj")
# p.loadURDF(filename)
#
#
# while True:
#     p.stepSimulation()
# p.disconnect()