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
skip = 4

# 加载数据集
images, poses, hwf, i_split = load_blender_data(LEGO_DIR, dev_res, skip)
H, W, focal_length = hwf
# ----- 11.29 22：26
ray_origins, ray_directions = get_rays(H, W, focal_length, poses[1])
print(ray_origins.shape, ray_directions.shape)
# print(ray_origins, ray_origins.dtype)
# print(ray_directions[0][0])

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
pos_out = pos_out.astype(mindspore.float32)
radiance_field = model(pos_out)
radiance_field = radiance_field.view(H, W, num_samples, 4)
print(radiance_field[0][0])
# 11.30 21:13
rgb_map, depth_map, acc_map = volume_rendering(radiance_field, ray_origins, depth_values)


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
num_depth_samples_per_ray = 64

num_iters = 2000

images = images.astype("float")
poses = poses.astype("float")

# 尝试 12.1 10：57
# 构建训练函数

# Instantiate loss function and optimizer
# loss_fn = mindspore.nn.CrossEntropyLoss()
# lr = 5e-3
# optimizer = mindspore.nn.Adam(params=model.trainable_params(),learning_rate=lr)



train(images, poses, hwf, i_split, near_point,
      far_point, num_depth_samples_per_ray, encoder,
      num_iters, model)

# 体渲染
trans_t = lambda t : torch.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=torch.float32)

rot_phi = lambda phi : torch.tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=torch.float32)

rot_theta = lambda th : torch.tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=torch.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    tmp = torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]).to(dtype=torch.float32)
    c2w = tmp @ c2w
    return c2w

frames = []
for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    c2w = pose_spherical(th, -30., 4.).to(device="cuda")
    rays_o, rays_d = get_rays(H, W, focal_length, c2w[:3,:4])
    rgb = tinynerf_step_forward(H, W, focal_length,
                                            c2w, 2, 6, 64, encoder,
                                            get_minibatches, model)
    rgb = rgb.detach().cpu().numpy()
    frames.append((255*np.clip(rgb,0,1)).astype(np.uint8))

f = 'video.mp4'
imageio.mimwrite(f, frames, fps=30, quality=7)

mp4 = open('video.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls autoplay loop>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

# clean cache
torch.cuda.empty_cache()
