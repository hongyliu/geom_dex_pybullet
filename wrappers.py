import copy
import numpy as np
from gym import Wrapper
from numpy.random import RandomState
import open3d as o3d


class PointCloudWrapper(Wrapper):
    def __init__(self, env, args):
        super(PointCloudWrapper, self).__init__(env)
        self.env_name = env.spec.id[:-3]
        self._max_episode_steps = args.max_episode_steps
        self.observation_space = copy.deepcopy(self.env.observation_space)
        self.rand = RandomState(args.seed)
        self.args = args
        self._target = dict()
        # rename the original obs to minimal_obs
        self.observation_space.spaces['minimal_obs'] = self.observation_space.spaces.pop(
            'observation')
        if args.point_cloud:
            self.observation_space.spaces['pc_obs'] = copy.deepcopy(
                self.observation_space.spaces['minimal_obs'])
            pc_dim = self.args.num_points * 6
            # shadowhands env also has target body, which also has point cloud as obs
            pc_dim *= 2
            self.observation_space.spaces['pc_obs'].shape = (
                self.observation_space.spaces['minimal_obs'].shape[0] + pc_dim,)

    def _normalize_points(self, point_set):
        """zero-center and scale to unit sphere"""
        point_set = point_set - \
            np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        return point_set

    def _save_point_cloud(self, filename, xyz):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud(filename, pcd)

    def _get_target(self, num_points, rand):
        if len(self._target.items()) == 0:
            target_points, target_normals = self.env.get_point_cloud('target', self.args.num_points, self.rand)
            self._target = {'points': target_points, 'normals': target_normals}
        return self._target['points'], self._target['normals']

    def observation(self, observation):
        assert isinstance(observation, dict)
        observation['minimal_obs'] = observation.pop('observation')
        if self.args.point_cloud:
            object_points, object_normals = self.env.get_point_cloud('object', self.args.num_points, self.rand)
            target_points, target_normals = self._get_target(self.args.num_points, self.rand)

            # concat all obs
            observation['pc_obs'] = np.concatenate([observation['minimal_obs'],
                                                    object_points.flatten(),
                                                    object_normals.flatten(),
                                                    target_points.flatten(),
                                                    target_normals.flatten()])

        return observation

    def flat2dict(self, obs):
        """convert flat obs to dict"""
        state_dim = 53
        goal_dim = 7
        if obs.shape[-1] == state_dim:  # without goal
            return {'minimal_obs': obs}
        elif obs.shape[-1] == state_dim + goal_dim:  # with goal
            return {'minimal_obs': obs[..., :-goal_dim], 'desired_goal': obs[..., -goal_dim:]}
            # with normals, without goal
        elif obs.shape[-1] == state_dim + self.args.num_points * 12:
            return {'minimal_obs': obs[..., :state_dim],
                    'object_points': obs[..., state_dim:state_dim + self.args.num_points * 3],
                    'object_normals': obs[...,
                                      state_dim + self.args.num_points * 3:state_dim + self.args.num_points * 6],
                    'target_points': obs[...,
                                     state_dim + self.args.num_points * 6:state_dim + self.args.num_points * 9],
                    'target_normals': obs[..., state_dim + self.args.num_points * 9:]}
            # with normals, with goal
        elif obs.shape[-1] == state_dim + self.args.num_points * 12 + goal_dim:
            return {'minimal_obs': obs[..., :state_dim],
                    'object_points': obs[..., state_dim:state_dim + self.args.num_points * 3],
                    'object_normals': obs[..., state_dim + self.args.num_points * 3:state_dim + self.args.num_points * 6],
                    'target_points': obs[..., state_dim + self.args.num_points * 6:state_dim + self.args.num_points * 9],
                    'target_normals': obs[..., state_dim + self.args.num_points * 9:state_dim + self.args.num_points * 12],
                    'desired_goal': obs[..., -goal_dim:]}
        else:
            print(obs.shape)
            raise NotImplementedError

    def reset(self):
        observation = self.env.reset()
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info
