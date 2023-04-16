# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from pytorch3d.renderer import PerspectiveCameras,look_at_rotation
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from numpy import genfromtxt
import yaml


DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "data_lego_white"
)

DEFAULT_URL_ROOT = "https://dl.fbaipublicfiles.com/pytorch3d_nerf_data"

ALL_DATASETS = ("lego", "fern", "pt3logo", "legow")


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    return batch


class ListDataset(Dataset):
    """
    A simple dataset made of a list of entries.
    """

    def __init__(self, entries: List) -> None:
        """
        Args:
            entries: The list of dataset entries.
        """
        self._entries = entries

    def __len__(
        self,
    ) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]



def get_synthetic_datasets(
    data_path: str = DEFAULT_DATA_ROOT,
    )-> Tuple[Dataset, Dataset, Dataset]:
        # Load eye and look at positions
    eye_file = os.path.join(data_path+"/config", "eye.csv")
    eye = genfromtxt(eye_file, delimiter=',')
    look_at_file = os.path.join(data_path+"/config", "look_at.csv")
    look_at = genfromtxt(look_at_file, delimiter=',')
    
    eye = np.concatenate([eye[:, 2, None], eye[:, 1, None], -eye[:, 0, None]], axis=1)
    look_at = np.concatenate([look_at[:, 2, None], look_at[:, 1, None], -look_at[:, 0, None]], axis=1)

    R_c_w = look_at_rotation(eye, look_at, device="cpu") # left multiplication

    t_c_w = torch.from_numpy(eye).float().to("cpu")
    t_w_c = -torch.bmm(R_c_w.transpose(1,2), torch.unsqueeze(t_c_w, dim=-1))
    t_w_c = torch.squeeze(t_w_c, dim=-1)

    # Load parameters
    with open(os.path.join(data_path+"/config", "config.yaml"), 'r') as stream:
        data_config = yaml.safe_load(stream)
    K_mat = np.array(data_config["cam_intrinics"])
    n_poses = np.array(data_config["number_poses"])
    light_c = torch.from_numpy(np.array(data_config["light_positions_XYZ"])[0]).float()
    light_w = torch.matmul(R_c_w, light_c)+t_c_w
    light_dir = R_c_w[:, :, -1]

    focal_length = np.array(K_mat[0, 0])
    focal_lengths = torch.from_numpy(focal_length).expand(n_poses, 1).float()
    principal_point = K_mat[0:2, 2]
    principal_points = torch.from_numpy(principal_point).expand(n_poses, 2).float()
    image_size = [data_config["image_height"], data_config["image_width"]]
    image_sizes = torch.Tensor(image_size).expand(n_poses, 2)
    cameras = PerspectiveCameras(
        focal_length=focal_lengths, 
        principal_point=principal_points, 
        R = R_c_w, 
        T = t_w_c, 
        in_ndc=False, 
        image_size=image_sizes, 
        device = "cpu")

    file_lst = os.listdir(data_path+"/images")
    transform = transforms.Compose([transforms.PILToTensor()])

    imgs = []
    for file in sorted(file_lst):
        if file.startswith("full"):
            image = Image.open(os.path.join(data_path+"/images", file))
            image = transform(image).permute(1,2,0)
            imgs.append(torch.unsqueeze(image, dim=0))

    images = torch.cat(imgs).float()/255

    # train1 = range(40, 120)
    train1 = range(0, 80)

    # train  = train1+train2
    train_idx =train1
    val_idx =train1

    test_idx = train1
    train_dataset, val_dataset, test_dataset = [
        ListDataset(
            [
                # {"image": images[i], "camera": cameras[i], "camera_idx": int(i)}
                {"image": images[i], "camera": cameras[i], "camera_idx": int(i), "light_xyz": light_w[i], "light_dir": light_dir[i]}
                for i in idx
            ]
        )
        for idx in [train_idx, val_idx, test_idx]
    ]

    return train_dataset, val_dataset, test_dataset


def get_colmap_datasets(
    data_path: str = DEFAULT_DATA_ROOT,  # 'lego | fern'
    )-> Tuple[Dataset, Dataset, Dataset]:
        # Load eye and look at positions
    optical_down_scale = 20
    pose_down_scale = 16

    poses_arr = np.load(os.path.join(data_path, 'config/poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    # print("Bounds: "+str(bds))
    # print(poses_arr[:, -3:])
    # import sys
    # sys.exit()
    bounds = torch.tensor(bds/pose_down_scale).cuda()
    imgdir = os.path.join(data_path, 'images')
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    R_c_w = torch.tensor(poses[:, 0:3, :])
    R_c_w = torch.cat([-R_c_w[:, 1:2, :], -R_c_w[:, 0:1, :], -R_c_w[:, 2:3, :]], dim=1)
    # R_c_w = torch.cat([-R_c_w[1:2, :, :], -R_c_w[0:1, :, :], -R_c_w[2:3, :, :]], dim=0)
    R_c_w = R_c_w.permute(2,0,1).float()

    t_w_c = torch.tensor(poses[:, 3, :]).transpose(0,1).float()
    # t_w_c = torch.concat([t_w_c[:, 0, None], t_w_c[:, 2, None],t_w_c[:, 1, None]], dim=1)
    # print(t_w_c.shape)
    # import sys
    # sys.exit()
    # t_w_c = -torch.bmm(R_c_w, torch.unsqueeze(t_w_c, dim=-1))
    t_w_c = -torch.bmm(R_c_w.transpose(1,2), torch.unsqueeze(t_w_c, dim=-1))
    t_w_c = torch.squeeze(t_w_c, dim=-1)/pose_down_scale

    # Load parameters
    with open(os.path.join(data_path+"/config", "config.yaml"), 'r') as stream:
        data_config = yaml.safe_load(stream)
    K_mat = np.array(data_config["cam_intrinics"])
    n_poses = np.array(data_config["number_poses"])
    light_c = torch.from_numpy(np.array(data_config["light_positions_XYZ"])[0]).float()
    light_w = torch.matmul(R_c_w, light_c)
    light_dir = R_c_w[:, :, -1]

    focal_length = np.array(K_mat[0, 0])
    focal_lengths = torch.from_numpy(focal_length).expand(n_poses, 1).float()/optical_down_scale
    principal_point = K_mat[0:2, 2]
    principal_points = torch.from_numpy(principal_point).expand(n_poses, 2).float()/optical_down_scale
    # image_size = [data_config["image_width"], data_config["image_height"]]
    image_size = [data_config["image_height"], data_config["image_width"]]
    image_sizes = torch.Tensor(image_size).expand(n_poses, 2)/optical_down_scale

    R_c_w[:, 0:2, :]  = -R_c_w[:, 0:2, :]
    cameras = PerspectiveCameras(
        focal_length=focal_lengths, 
        principal_point=principal_points, 
        R = R_c_w, 
        T = t_w_c, 
        in_ndc=False, 
        image_size=image_sizes, 
        device = "cpu")

    file_lst = os.listdir(data_path+"/images")

    # transform = transforms.Compose([transforms.PILToTensor(),transforms.Resize([200, 300])])
    # transform = transforms.Compose([transforms.Resize([200, 300])])
    transform = transforms.Compose([transforms.PILToTensor()])

    imgs = []
    for file in sorted(file_lst):
        if file.startswith("DSC"):
            image = Image.open(os.path.join(data_path+"/images", file))
            # image = tifffile.imread(/os.path.join(data_path+"/images", file))

            image = transform(image).permute(1,2,0)
            imgs.append(torch.unsqueeze(image, dim=0))

    images = torch.cat(imgs).float()/255.0
    # images = torch.cat(imgs).float()

    # train1 = range(0,81,3)
    train1 = range(0,74,3)
    # train1 = range(0,14)

    # print(train1)
    # print(len(train1))
    # train1 = range(0,107,8)
    # print(len(train1))

    # import sys
    # sys.exit()
    train_idx =train1
    val_idx =train1

    test_idx = train1
    train_dataset, val_dataset, test_dataset = [
        ListDataset(
            [
                {"image": images[i], "camera": cameras[i], "camera_idx": int(i),
                "light_xyz": light_w[i], "light_dir": light_dir[i]}
                # "light_xyz": light_w[i], "light_dir": light_dir[i], "bounds": bounds[:, i]}
                for i in idx
            ]
        )
        for idx in [train_idx, val_idx, test_idx]
    ]

    return train_dataset, val_dataset, test_dataset