"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from shadowhand_gym.envs.config import *
from torch.utils.data import Dataset, DataLoader
from numpy.random import RandomState
from train_pointnet import init_wandb, log_callback
from scipy.spatial.transform import Rotation as R
import gym

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--train_names', nargs='*', default=[], type=str, help='the environment name')
    parser.add_argument('--test_names', nargs='*', default=[], type=str, help='the environment name')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--std_data_aug', type=float, default=0.00384, help="data augmentation noise std")
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    argus = parser.parse_args()
    # default to use train/test split specified in dex_envs/configs
    argus.train_names = argus.train_names if argus.train_names else ALL_CLS_TRAIN
    argus.test_names = argus.test_names if argus.test_names else ALL_CLS_TEST
    assert len(list(set(argus.train_names) & set(argus.test_names))
               ) == 0, 'cannot have overlapping train/test envs'
    return argus


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def makeEnv(env_name, idx, render, args):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""
    def helper():
        e = gym.make('ShadowHandBlock-v1', object=env_name, classify=True, render=render)
        e.seed(args.seed + idx)
        return e
    return helper


class TwoStreamDataset(Dataset):
    def __init__(self, env_names, num_points=2500, data_aug=True, std_data_aug=0.02, num_data=100000, seed=123):
        self.envs = [makeEnv(env_name, 0, False, args)() for env_name in env_names]
        self.num_classes = len(env_names)
        for env in self.envs:
            env.reset()

        self.rand = RandomState(seed)
        # self.rand = np.random
        self.num_points = num_points
        self.data_aug = data_aug
        self.num_data = num_data
        self.std_data_aug = std_data_aug
        self.cache = {}

    def _get_points(self, env):
        object_points, object_normals = env.get_point_cloud('target', self.num_points, self.rand)
        return object_points, object_normals

    def _normalize(self, point_set):
        """zero-center and scale to unit sphere"""
        point_set = point_set - \
            np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        return point_set

    def _augment(self, point_set):
        # random jitter
        point_set += self.rand.normal(0, self.std_data_aug, size=point_set.shape)
        return point_set

    def __getitem__(self, index):
        target = index % self.num_classes
        if target in self.cache.keys():
            point_set = self.cache[target][:, :3]
            sampled_normals = self.cache[target][:, 3:]
        else:
            sampled_points, sampled_normals = self._get_points(self.envs[target])
            # zero-center and scale to unit sphere
            point_set = self._normalize(sampled_points)
            # data augmentation
            if self.data_aug:
                point_set = self._augment(point_set)

        rotation = R.random()
        point_set = np.matmul(point_set, rotation.as_matrix().T)
        # apply same rotations to normals
        normal_set = np.matmul(sampled_normals, rotation.as_matrix().T)

        return_set = np.concatenate(
            [point_set, normal_set], axis=-1).astype(np.float32)
        if target not in self.cache.keys():
            self.cache[target] = return_set
        return return_set, target

    def __len__(self):
        return int(self.num_data)


def get_dataloaders(args):

    train_dataset = TwoStreamDataset(args.train_names, num_points=args.num_point, data_aug=True,
                                     std_data_aug=args.std_data_aug, seed=args.seed)
    val_dataset = TwoStreamDataset(args.train_names, num_points=args.num_point,
                                   data_aug=False, seed=args.seed + 1)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True)

    return train_dataloader, val_dataloader


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    init_wandb(args, 'pointnet_cls')
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./dex_logs/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    train_dataloader, val_dataloader = get_dataloaders(args)
    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    device = torch.device('cpu') if args.use_cpu else torch.device('cuda')
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model_category_10.pth', map_location=device)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except Exception as e:
        log_string(e)
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9):
            loss_dict = dict()
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            loss_dict['train_loss'] = loss
            correct = pred_choice.eq(target.long().data).cpu().sum()
            acc = correct.item() / float(points.size()[0])
            mean_correct.append(acc)
            loss_dict['train_accuracy'] = acc
            loss.backward()
            optimizer.step()
            global_step += 1
            log_callback(loss_dict)

        acc_dict = dict()
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        acc_dict['train_epoch_accuracy'] = train_instance_acc


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), val_dataloader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            acc_dict['test_instance_accuracy'] = instance_acc
            acc_dict['test_class_accuracy'] = class_acc

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model_category_10.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
        log_callback(acc_dict)
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
