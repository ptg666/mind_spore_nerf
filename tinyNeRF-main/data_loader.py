import os
import json
import imageio

import mindspore
# import torch

import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_blender_data(BASE_DIR: str, dev_res: int = 4, skip: int = 4, dtype=np.float32, device="cuda"):
    # Read json files for 'train', 'val' and 'test' dataset, details covered above.
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(BASE_DIR, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    # 获取所有图片和对应位姿下 c2w 矩阵
    # Get all the images and transform matrix from the folder
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []

        # Skip some images to make the dataset small enough to train on Google
        # Colab, the default skip value is set to 4, which downsize the total
        # number of images from 400 to 100 (for LEGO)
        if s is not 'train':
            for frame in meta['frames'][::skip]:
                fname = os.path.join(BASE_DIR, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname)) # (800, 800, 4)
                poses.append(np.array(frame['transform_matrix'])) # append 4*4 matrix
        else:
            for frame in meta['frames']: # load all training images
                  fname = os.path.join(BASE_DIR, frame['file_path'] + '.png')
                  imgs.append(imageio.imread(fname)) # (800, 800, 4)
                  poses.append(np.array(frame['transform_matrix'])) # append 4*4 matrix

        # Transfer RGBA values from 0-255 to 0-1, notice that we have 4 channels
        imgs = (np.array(imgs) / 255.).astype(dtype=dtype)
        poses = np.array(poses).astype(dtype=dtype)

        # Indicators of seperation between train/val/test dataset
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    # Calculate the focal length of the camera
    H, W = imgs[0].shape[:2] # 800, 800
    camera_angle_x = float(meta['camera_angle_x']) # train/val/test use the same camera
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # [train_index, val_index, test_index]
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    # Concatenate train/val/test set to one large set, and we'll use
    # i_split to access different parts later
    imgs = np.concatenate(all_imgs, 0) # (N, 800, 800, 4)
    poses = np.concatenate(all_poses, 0) # (N, 4, 4)

    # Reduce image resolution so that we could train on Colab
    if dev_res > 1:
        if H % dev_res != 0:
            raise ValueError(
                f"""The value H is not dividable by dev_res. Please select an
                    appropriate value.""")
        H = int(H // dev_res)
        W = int(W // dev_res)
        focal = focal / float(dev_res)
        imgs_reduce = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_reduce[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_reduce

    # Print data shape
    print("Images shape: ", imgs.shape)
    print("Poses shape: ", poses.shape)

    # Convert useful variables to tensors
    imgs = mindspore.Tensor.from_numpy(np.asarray(imgs))
    poses = mindspore.Tensor.from_numpy(np.asarray(poses))
    focal = mindspore.Tensor.from_numpy(np.asarray(focal))
    # imgs.astype(mindspore.float32)
    # poses.astype(mindspore.float32)
    # focal.astype(mindspore.float32)
    # imgs[:,:,:] /=
    return imgs, poses, [H, W, focal], i_split
