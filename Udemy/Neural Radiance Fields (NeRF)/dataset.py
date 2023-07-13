import torch
import numpy as np

import os
import imageio.v3 as iio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def get_rays(datapath, mode):

    pose_files_names = [f for f in os.listdir(
        f'{datapath}/{mode}/pose') if f.endswith('.txt')]
    intrinsics_file_names = [f for f in os.listdir(
        f'{datapath}/{mode}/intrinsics') if f.endswith('.txt')]
    img_file_names = [f for f in os.listdir(datapath + '/imgs') if mode in f]

    assert len(pose_files_names) == len(intrinsics_file_names)
    assert len(img_file_names) == len(pose_files_names)

    # read
    N = len(pose_files_names)
    poses = np.zeros((N, 4, 4))
    intrinsics = np.zeros((N, 4, 4))
    images = []  # shape of images in unknown

    for i in range(N):

        # read poses
        name = pose_files_names[i]
        pose = open(f'{datapath}/{mode}/pose/{name}').read().split()
        poses[i] = np.array(pose, dtype=float).reshape(4, 4)

        # read intrinsics
        name = intrinsics_file_names[i]
        intrinsic = open(f'{datapath}/{mode}/intrinsics/{name}').read().split()
        intrinsics[i] = np.array(intrinsic, dtype=float).reshape(4, 4)

        # read images
        name = img_file_names[i]
        img = iio.imread(datapath + '/imgs/' + name) / \
            255.  # read and normalize the data (img.max() == 255.)

        images.append(img[None, ...])  # append to list the unsqueezed image

    # concatenate all the images (shape == (90, 400, 400, 4) 90 images, size 400 x 400, 4 channels)
    images = np.concatenate(images)

    H = images.shape[1]
    W = images.shape[2]

    if images.shape[3] == 4:  # RGBA -> RGB
        images = images[..., :3] * images[..., -1:] + (1 - images[..., -1:])

    ray_origins = np.zeros((N, H*W, 3))  # ray origins
    ray_directions = np.zeros((N, H*W, 3))  # ray directions
    target_pixel_values = images.reshape((N, H*W, 3))

    for i in range(N):  # for each image

        c2w = poses[i]
        f = intrinsics[i, 0, 0]

        u = np.arange(W)  # pixel horizontal coordinate
        v = np.arange(H)  # pixel vertical coordinate
        u, v = np.meshgrid(u, v)

        # direction vectors ()
        directions = np.stack((u - W / 2,  # u to X axis
                               # v to Y axis (v goes on the opposite direction)
                               -(v - H / 2),
                               -np.ones_like(u) * f),  # Z axis (goes on the opposite direction)
                              axis=-1)  # inverts coordinates order => dirs.shape -> (400, 400, 3)
        directions = (c2w[:3, :3] @ directions[..., None]).squeeze(-1)
        # normalized direction vectors
        directions = directions / \
            np.linalg.norm(directions, axis=-1, keepdims=True)

        # reshape from (440, 400, 3) to (160000, 3)
        ray_directions[i] = directions.reshape(-1, 3)
        ray_origins[i] += c2w[:3, 3]

    return ray_origins, ray_directions, target_pixel_values
