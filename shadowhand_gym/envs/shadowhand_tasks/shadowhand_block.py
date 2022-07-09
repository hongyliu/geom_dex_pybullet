import json
import math
from abc import ABC
import os
from shadowhand_gym.envs.core import RobotTaskEnv
from shadowhand_gym.pybullet_shadow_hand import PyBullet
from shadowhand_gym.envs.robots import ShadowHand
from shadowhand_gym.envs.tasks import Block
import pybullet as p
import numpy as np
from numpy.random import RandomState
from shadowhand_gym.envs.camera import CameraArray, Camera, CalibratedCamera
import cv2
import yaml
import open3d as o3d


class ShadowHandBlockEnv(RobotTaskEnv, ABC):
    def __init__(self, object: str='YcbPear', classify: bool=False, render: bool = False, reward_type: str = "sparse") -> None:
        """Block manipulation task with Shadow Dexterous Hand robot.

        Args:
            render (bool, optional): Activate rendering. Defaults to False.
            reward_type (str, optional): 'sparse' or 'dense'. Defaults to 'sparse'.
        """
        self.sim = PyBullet(render=render)
        self.robot = ShadowHand(
            sim=self.sim,
            base_position=[-0.3, 0.0, 0.0],
            base_orientation=[0.5, -0.5, 0.5, -0.5],
        )
        self.object_name = object
        self.task = Block(sim=self.sim, robot=self.robot, object=object, classify=classify, reward_type=reward_type)
        self.physicId = self.sim.physics_client._client
        self._set_cameras()
        RobotTaskEnv.__init__(self)


    def _set_cameras(self):
        # self._path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        camera_list = []
        target_camera_list = []
        object_up_vector = [0, -1, 0]
        target_up_vector = [1, 0, 0]
        camera_list.append(Camera((0.2, 0.2, 0.7), (0.707, 0.691, -0.147, -0.043),
                                  (0.25, 0.0, 0.0), object_up_vector, pybullet_client_id=self.physicId))
        camera_list.append(Camera((0.2, 0.1, 0.7), (0.707, 0.691, -0.147, -0.043),
                                  (0.25, 0.0, 0.0), object_up_vector, pybullet_client_id=self.physicId))
        camera_list.append(Camera((0.2, 0, 0.7), (0.755, 0.642, -0.119, -0.063),
                                  (0.25, 0.0, 0.0), object_up_vector, pybullet_client_id=self.physicId))
        camera_list.append(Camera((0.2, -0.1, 0.7), (0.791, 0.593, -0.057, -0.136),
                                  (0.25, 0.0, 0.0), object_up_vector, pybullet_client_id=self.physicId))
        camera_list.append(Camera((0.2, -0.2, 0.7), (0.791, 0.593, -0.057, -0.136),
                                  (0.25, 0.0, 0.0), object_up_vector, pybullet_client_id=self.physicId))

        target_camera_list.append(Camera((-4.8, -4.8, 0.55), (0.707, 0.691, -0.147, -0.043), (-5, -5, 0.4), target_up_vector, pybullet_client_id=self.physicId))
        target_camera_list.append(Camera((-4.8, -4.9, 0.55), (0.707, 0.691, -0.147, -0.043), (-5, -5, 0.4), target_up_vector, pybullet_client_id=self.physicId))
        target_camera_list.append(Camera((-4.8, -5, 0.55), (0.755, 0.642, -0.119, -0.063), (-5, -5, 0.4), target_up_vector, pybullet_client_id=self.physicId))
        target_camera_list.append(Camera((-4.8, -5.1, 0.55), (0.791, 0.593, -0.057, -0.136), (-5, -5, 0.4), target_up_vector, pybullet_client_id=self.physicId))
        target_camera_list.append(Camera((-4.8, -5.2, 0.55), (0.791, 0.593, -0.057, -0.136), (-5, -5, 0.4), target_up_vector, pybullet_client_id=self.physicId))

        self._object_cameras = CameraArray(camera_list)
        self._target_cameras = CameraArray(target_camera_list)
        # self._trajectory = o3d.io.read_pinhole_camera_trajectory(os.path.join(self._path, 'camera.json'))

    def get_point_cloud(self, camera_name: str, num_points, rand: RandomState):
        if 'object' == camera_name:
            rgbs, depths = self._object_cameras.get_images()
            exs, ins = self._object_cameras.get_matrices()
        elif 'target' == camera_name:
            rgbs, depths = self._target_cameras.get_images()
            exs, ins = self._target_cameras.get_matrices()
        pcds = o3d.geometry.PointCloud()
        for i, (img, depth) in enumerate(zip(rgbs, depths)):
            # cv2.imwrite('./{}_color_{}.jpg'.format(camera_name, self.object_name), img)
            # cv2.imwrite('./{}_depth_{}.jpg'.format(camera_name, i), depth)

            im = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(img), o3d.geometry.Image(depth), 5.0, 3.5, False)
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(*ins[i])
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    im, intrinsic, exs[i])
            # o3d.visualization.draw_geometries([pcd])
            # o3d.io.write_point_cloud(os.path.join(self._path, 'object_{}.ply'.format(i)), pcd)
            pcds = pcd
            if pcds.has_points():
                break
        # o3d.visualization.draw_geometries(pcds)
        # o3d.io.write_point_cloud('./points_{}.ply'.format(camera_name), pcds)
        object_points = np.asarray(pcds.points)
        selected = rand.randint(
            low=0, high=object_points.shape[0], size=num_points)
        sampled_points = object_points[selected].copy()
        assert sampled_points.shape[0] == num_points and sampled_points.shape[1] == 3
        sample_pc = o3d.geometry.PointCloud()
        sample_pc.points = o3d.utility.Vector3dVector(sampled_points)
        # o3d.io.write_point_cloud('./sample_points_{}_{}.ply'.format(camera_name, self.object_name), sample_pc)
        if not sample_pc.has_normals():
            sample_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        sampled_normals = np.asarray(sample_pc.normals)
        assert sampled_normals.shape[0] == num_points and sampled_normals.shape[1] == 3

        return sampled_points, sampled_normals
