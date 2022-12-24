import os
import sys
import os
import time
import matplotlib.pyplot as plt
# import torch
import mindspore
# import torchvision
from data_loader import load_blender_data
from NeRF import get_rays
from NeRF import sample_points_from_rays
from NeRF import positional_encoding
from NeRF import TinyNeRF
from NeRF import volume_rendering
from NeRF import train

# from NeRF import tinynerf_step_forward
import numpy as np
# from NeRF import tinynerf_step_forward
from NeRF import get_minibatches
from tqdm import tqdm_notebook as tqdm
import imageio
from IPython.display import HTML
from base64 import b64encode


# --mindspore 会根据设备自动设置上下文环境

# if torch.cuda.is_available():
#     print("GPU")
#     DEVICE = torch.device("cuda")
# else:
#     print("Using CPU!!")
#     DEVICE = torch.device("cpu")
DTYPE = mindspore.float32

# 加载 NeRF 所需的数据
dataset_path = "./"
LEGO_DIR = os.path.join(dataset_path, "data", "nerf_synthetic", "lego")
print("Base_dir is :",LEGO_DIR)

dev_res = 4
skip = 8

# 加载数据集
images, poses, hwf, i_split = load_blender_data(LEGO_DIR, dev_res, skip)
H, W, focal_length = hwf
# ----- 11.29 22：26
ray_origins, ray_directions = get_rays(H, W, focal_length, poses[1])
print(ray_origins.shape, ray_directions.shape)
print(ray_origins, ray_origins.dtype)
print(ray_directions[0][0])

num_samples = 4
near_point = 2
far_point = 6

sampled_points, depth_values = sample_points_from_rays(
                                 ray_origins,
                                 ray_directions,
                                 near_point=near_point,
                                 far_point=far_point,
                                 num_samples=num_samples,
                                 random=True
                               )
print(depth_values.shape)
print(sampled_points.shape)
print(sampled_points[0][0])
# 11.30 10:36
flattened_sampled_points = sampled_points.reshape((-1, 3))
pos_out = positional_encoding(flattened_sampled_points, 3)

pos_dim = pos_out.shape[-1]
# 11.30 15.14
model = TinyNeRF(pos_dim, fc_dim=128)
# 由于 mindspore 自动分析当前环境，所以不用转换数据
# model.cuda()

# 将所有的位姿经过一次 MLP，得到计算出的 RGB 和体密度，相当于做了一次初始化 (160000,21) -> (160000,4)

# # 11.30 21:13
# rgb_map, depth_map, acc_map = volume_rendering(radiance_field, ray_origins, depth_values)
#

# Near and far clipping distance
near_point = 2
far_point = 6

# Encoder definition, we use positional encoding
freq = 6
include_input = True
encoder = lambda x: positional_encoding(x, include_input=include_input, freq=freq)
enc_dim = (include_input + 2 *freq) * 3

# Define a tinynerf model, feel free to change the `fc_dim`
model = TinyNeRF(enc_dim, fc_dim=128)

# 训练
# Number of depth samples along each ray.
num_depth_samples_per_ray = 32

num_iters = 140


# 尝试 12.1 10：57
# 构建训练函数

# Instantiate loss function and optimizer
# loss_fn = mindspore.nn.CrossEntropyLoss()
# lr = 5e-3
# optimizer = mindspore.nn.Adam(params=model.trainable_params(),learning_rate=lr)
images = mindspore.Tensor(images,dtype=mindspore.float32)
poses = mindspore.Tensor(poses,dtype=mindspore.float32)


train(images, poses, hwf, i_split, near_point,
      far_point, num_depth_samples_per_ray,
      num_iters, model)