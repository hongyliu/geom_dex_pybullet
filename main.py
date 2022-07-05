import time
import os
import json
import numpy as np
import gym
from numpy.random import RandomState
from rl_modules.utils import makeEnv
import shadowhand_gym

env = gym.make('ShadowHandBlock-v1', object='YcbGelatinBox', render=True)

obs = env.reset()
done = False
# while not done:
    # Random action
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
env.get_point_cloud('object', 512, RandomState(42))
env.get_point_cloud('target', 512, RandomState(42))
while True:
    time.sleep(1)

env.close()

# import matplotlib.pyplot as plt
# import numpy as np
# import pybullet as p
# import time
# import pybullet_data
# import cv2
#
# direct = p.connect(p.GUI)  #, options="--window_backend=2 --render_device=0")
# #egl = p.loadPlugin("eglRendererPlugin")
#
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.loadURDF('plane.urdf')
# p.loadURDF("r2d2.urdf", [0, 0, 1])
# p.loadURDF('cube_small.urdf', basePosition=[0.0, 0.0, 0.025])
# cube_trans = p.loadURDF('cube_small.urdf', basePosition=[0.0, 0.1, 0.025])
# p.changeVisualShape(cube_trans, -1, rgbaColor=[1, 1, 1, 0.1])
# width = 128
# height = 128
#
# fov = 60
# aspect = width / height
# near = 0.02
# far = 1
#
# view_matrix = p.computeViewMatrix([0.2, 0.2, 1.5], [0, 0, 0], [1, 0, 0])
# projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
#
# # Get depth values using the OpenGL renderer
# images = p.getCameraImage(width,
#                           height,
#                           view_matrix,
#                           projection_matrix,
#                           shadow=True,
#                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
# # NOTE: the ordering of height and width change based on the conversion
# rgb_opengl = np.reshape(images[2], (height, width, 4))
# depth_buffer_opengl = np.reshape(images[3], [width, height])
# depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
# seg_opengl = np.reshape(images[4], [width, height]) * 1. / 255.
#
# cv2.imwrite('./color_r2d2.jpg', rgb_opengl)
# cv2.imwrite('./depth_r2d2.jpg', depth_opengl)
#
# for i in range(10000):
#     p.stepSimulation()
#     time.sleep(1. / 240.)
# p.disconnect()



# import pybullet as p
# import pybullet_data
# import cv2
# import time
# from shadowhand_gym.envs.camera import Camera, CameraArray
#
# physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
# p.configureDebugVisualizer(lightPosition=[0, 0, 9])
#
# Id = p.loadURDF("r2d2.urdf", (0, 0, -0.6))
#
# camera = Camera((0.2, 0, 0.7), (0.704, 0.697, -0.114, -0.072))
# rgb, depth = camera.get_image()
# cv2.imwrite('./color_r2d2.jpg', rgb)
# cv2.imwrite('./depth_r2d2.jpg', depth)
# cv2.imshow('depth', depth)
# for i in range(10000):
#     p.stepSimulation()
#     time.sleep(1. / 240.)
# p.disconnect()

# import numpy as np
# import open3d as o3d
# import glob
# import cv2
#
# if __name__ == "__main__":
#     pcds = []
#     trajectory = o3d.io.read_pinhole_camera_trajectory(
#         './shadowhand_gym/envs/camera.json')
#     # o3d.io.write_pinhole_camera_trajectory("test.json", trajectory)
#     images = glob.glob('./color_*.png')
#     depths = glob.glob('./depth_*.png')
#     for i, (img, depth) in enumerate(zip(images, depths)):
#         im1 = cv2.imread(img)
#         im2 = cv2.imread(depth)
#         tmp = np.array(im2)[:, :, :1]
#         print(tmp[tmp == 1].shape)
        # im1 = o3d.io.read_image(redwood_rgbd.depth_paths[i])
        # im2 = o3d.io.read_image(redwood_rgbd.color_paths[i])
    #     im = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         im2, im1, 1000.0, 5.0, False)
    #     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #         im, trajectory.parameters[i].intrinsic,
    #         trajectory.parameters[i].extrinsic)
    #     pcds.append(pcd)
    # o3d.visualization.draw_geometries(pcds)
