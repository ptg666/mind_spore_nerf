import math
from typing import Optional, Tuple

# import torch
import mindspore

# import torchvision
# from torch import nn
from mindspore import nn
# from torch.nn import functional as F
# from torchvision.models import feature_extraction
import numpy as np
import matplotlib.pyplot as plt

#                                                   torch.Tensor
def get_rays(H: int, W: int, F: float, cam2world: mindspore.Tensor):
    r"""Compute rays passing through all pixels of an image

    Args:
      H: Height of an image
      W: Width of an image
      G: Focal length of camera
      cam2world: A 6-DoF rigid-body transform (4x4 matrix) that transforms a 3D point 
      from the camera frame to the world frame (in homogeneous coordinate)
    
    Returns:
      ray_origins: A tensor of shape (H, W, 3) denoting the centers of each ray. 
        ray_origins[i][j] denotes the origin of the ray passing through pixel at
        row index `i` and column index `j`.
      ray_directions: A tensor of shape (H, W, 3) denoting the direction of each 
        ray. ray_directions[i][j] denotes the direction (x, y, z) of the ray 
        passing through the pixel at row index `i` and column index `j`.
    
    Note:
      We use homogeneous coordinate for transformation. Each point in the 3D space
      could be represent as [x, y, z, 1]. With cam2world @ [x, y, z, 1].T, we
      are transfomring that point from camera fram to the world frame.
      Besides, the origin point of the camera frame is [0, 0, 0, 1]. we could 
      get world-frame coordinate of that origin with cam2world @ [0, 0, 0, 1].T,
      which turns out to be cam2world[:3, -1].
    """
    ray_origins, ray_directions = None, None

    # 'i' is the x axis of points, all columns are identical
    # 'j' is the y axis of points, all rows are identical 
    # i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')

    i, j = mindspore.numpy.meshgrid(mindspore.numpy.linspace(0, W - 1, W), mindspore.numpy.linspace(0, H - 1, H),indexing="xy")


    # 变换了类型，注释掉了下面两行
    # i = i.to(cam2world) # (H, W)
    # j = j.to(cam2world) # (H, W)

    # Calculate the direction w.r.t the camera pinhole, whose x, y coordinates are
    # the same as the center of image (W/2, H/2), and z coordinate is negative 
    # focal length. You can image constructing a 3D coordinate, whose origin is
    # at the center of the image.
    # We simply normalize the direction with focal length here, so that coordinates 
    # value lie in [-1, 1], which is beneficial for numerical stability.
    # directions = torch.stack([(i - W * .5) / F,
    #                           -(j - H * .5) / F,
    #                           -torch.ones_like(i)] , dim=-1) # (H, W, 3)

    directions = mindspore.numpy.stack([(i - W * .5) / F,
                              -(j - H * .5) / F,
                              -mindspore.numpy.ones_like(i)],axis=2)  # (H, W, 3)

    # Apply transformation to the direction, f(d) = Ad = dA^(T)
    # 转置矩阵
    # ray_directions = directions @ cam2world[:3, :3].t() # (H, W, 3)
    np_transpose = mindspore.numpy.transpose(cam2world[:3, :3])
    ray_directions = directions.asnumpy() @ np_transpose.asnumpy()  # (H, W, 3)
    ray_directions = mindspore.Tensor(ray_directions)
    # All the rays share the same origin
    # ray_origins = cam2world[:3, -1].expand(ray_directions.shape) # (H, W, 3)
    ray_origins = cam2world[:3, -1].expand_as(ray_directions) # (H, W, 3)
    return ray_origins, ray_directions

#
def sample_points_from_rays(
    ray_origins: mindspore.Tensor,
    ray_directions: mindspore.Tensor,
    near_point: float,
    far_point: float,
    num_samples: int,
    random: Optional[bool] = True
) -> Tuple[mindspore.Tensor]:
    H, W, _ = ray_origins.shape

    depth_values = mindspore.numpy.linspace(near_point, far_point, num_samples)
    temp_tensor = mindspore.numpy.ones([H,W,num_samples])
    depth_values = depth_values.expand_as(temp_tensor)

    if random:

      noise = mindspore.numpy.rand([H, W, num_samples])
      # Normalize the noise by uniform distance among samples
      noise = noise * (far_point - near_point) / num_samples

      # Add noise to depth value, get shape (H, W, num_samples)
      depth_values = depth_values + noise

    # Note: ray_directions all have different lengths, but are all close to 1,
    #       we don't transfer them to unit vector for simplicity (?)
    # example not code (H, W, num_samples, 3) = (H, W, 1, 3) + (H, W, 1, 3) * (H, W, num_samples, 1)
    sampled_points = ray_origins[..., None, :] + ray_directions[..., None, :] \
                      * depth_values[..., :, None]
    return sampled_points, depth_values


def positional_encoding(
    pos_in, freq=32, include_input=True, log_sampling=True
) -> mindspore.Tensor:
    # torch.Tensor

    # Whether or not include the input in positional encoding
    pos_out = [pos_in] if include_input else []

    # Shape of freq_bands: (freq)
    if log_sampling:
        # freq_bands = 2.0 ** torch.linspace(0.0, freq - 1, freq).to(pos_in)
        freq_bands = 2.0 ** mindspore.numpy.linspace(0.0, freq - 1, freq)
    else:
        freq_bands = mindspore.numpy.linspace(2.0 ** 0.0, 2.0 ** (freq - 1), freq)
        # freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (freq - 1), freq).to(pos_in)

    # TODO: why reduce \pi when calculating sin and cos
    for freq in freq_bands:
        for func in [mindspore.numpy.sin, mindspore.numpy.cos]:
        # for func in [torch.sin, torch.cos]:
            pos_out.append(func(pos_in * freq))

    # pos_out = torch.cat(pos_out, dim=-1)
    concat_op = mindspore.ops.Concat(axis=1)
    pos_out = concat_op(pos_out)
    return pos_out


def cumprod_exclusive(tensor: mindspore.Tensor) -> mindspore.Tensor:

    # cumprod = (tensor[0], tensor[0]*tensor[1], tensor[0]*tensor[1]*tensor[2], ...)
    # cumprod = mindspore.numpy.cumprod(tensor)
    cumprod = tensor
    # Roll down the elements along dimension 'dim' by 1 element.
    cumprod = mindspore.numpy.roll(cumprod, 1, -1)

    # cumprod = (1, tensor[0], tensor[0]*tensor[1], ...)
    cumprod[..., 0] = 1.

    return cumprod


def volume_rendering(
    radiance_field: mindspore.Tensor,
    ray_origins: mindspore.Tensor,
    depth_values: mindspore.Tensor
) -> Tuple[mindspore.Tensor]:

    rgb_map, depth_map, acc_map = None, None, None

    # Concatenate the H and W dimension, so that the first dimension represents
    # the number of rays
    H, W, num_samples, _ = radiance_field.shape
    radianceField = radiance_field.view(H*W, num_samples, 4) # (num_rays, num_samples, 4)
    rayOrigins = ray_origins.view(H*W, -1) # (num_rays, 3)
    depthValues = depth_values.view(H*W, -1) # (num_rays, num_samples)

    # radianceField = radiance_field.clone().contiguous().view(H * W, num_samples, 4)  # (num_rays, num_samples, 4)
    # rayOrigins = ray_origins.clone().contiguous().view(H * W, -1)  # (num_rays, 3)
    # depthValues = depth_values.clone().contiguous().view(H * W, -1)  # (num_rays, num_samples)

    # Apply relu to the predicted volume density to make sure that all the values
    # are larger or equal than zero
    # radianceField
    relu = mindspore.ops.ReLU()
    sigma = relu(radianceField[..., 3]) # (num_rays, num_samples)
    # sigma = F.relu(radianceField[..., 3]) # (num_rays, num_samples)

    # Apply sigmoid to predicted RGB color (which is a logit), so that all the values
    # lie between -1 and 1
    sigmoid = mindspore.ops.Sigmoid() # (num_rays, num_samples, 3)
    rgb=sigmoid(radianceField[..., :3])
    # rgb = torch.sigmoid(radianceField[..., :3]) # (num_rays, num_samples, )
    # Redundant vector
    one_e_10 = mindspore.Tensor([1e10])
    one_e_10 = one_e_10.expand_as(depthValues[..., :1]) # (num_rays, 1)

    # We get the distance between sample points, but notice that the last sample
    # points of each ray would not have corresponding distance, we set it to be
    # a large value (1e10), so that it's approximately zero when `exp(-1e10 * sigma)`
    delta = depthValues[..., 1:] - depthValues[..., :-1] # (num_rays, num_samples-1)
    # 定义无穷远点,并放到采样序列中，这里为了节省资源，只设置4个采样
    op = mindspore.ops.Concat(axis=-1)
    delta = op((delta,one_e_10))
    # delta = torch.cat((delta, one_e_10), dim=-1) # (num_rays, num_samples)

    # ！！！！！
    # 注意这里的sigma，差距过大可能要改进
    # ！！！！！
    # Calculating `alpha = 1−exp(−𝜎𝛿)`
    op = mindspore.ops.Exp()
    alpha = 1. - op(-sigma * delta)  # (num_rays, num_samples)

    # Calculate transmittance value, notice that T_1 = 1
    # It's possible that we get alpha=1 (sigma=0) for point A, which could make
    # transmittance of all the points after point A to be 0, we also want to take
    # their information into consideration, therefore we add a small value (1e-10)
    # to avoid vanishing transmittance
    trans = cumprod_exclusive(1. - alpha + 1e-10) # (num_rays, num_samples)
    weights = alpha * trans # (num_rays, num_samples)

    # (num_rays, num_samples, 1) * (num_rays, num_samples, 3) -> (num_rays, num_samples, 3)
    rgb_map = (weights[..., None] * rgb).sum(axis=-2) # (num_rays, 3)
    rgb_map = rgb_map.view(H, W, 3) # (H, W, 3)

    # (num_rays, num_samples) * (num_rays, num_samples) -> (num_rays, num_samples)
    depth_map = (weights * depthValues).sum(axis=-1) # (num_rays)
    depth_map = depth_map.view(H, W) # (H, W)

    # accumulated transmittance map
    acc_map = weights.sum(-1) # (num_rays)
    acc_map = acc_map.view(H, W) # (H, W)

    return rgb_map, depth_map, acc_map

#
def get_minibatches(inputs: mindspore.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    inputs.astype(mindspore.float32)
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

from mindspore.common.initializer import Normal
class TinyNeRF(nn.Cell):
    def __init__(self, pos_dim, fc_dim=128):
      super().__init__()
      self.nerf = nn.SequentialCell(
                  nn.Dense(pos_dim, fc_dim, Normal(0.04), Normal(0.04)),
                  nn.ReLU(),
                  nn.Dense(fc_dim, fc_dim, Normal(0.04), Normal(0.04)),
                  nn.ReLU(),
                  nn.Dense(fc_dim, fc_dim, Normal(0.04), Normal(0.04)),
                  nn.ReLU(),
                  nn.Dense(fc_dim, 4, Normal(0.04), Normal(0.04))
                  )
    def construct(self, x):
      r"""Output volume density and RGB color (4 dimensions), given a set of
      positional encoded points sampled from the rays
      """
      x = self.nerf(x)
      return x

def tinynerf_step_forward(height, width, focal_length, trans_matrix,
                             near_point, far_point, num_depth_samples_per_ray,
                             encoder, get_minibatches_function, model):

    # Get the "bundle" of rays through all image pixels.
    # (H, W, 3) & (H, W, 3)
    ray_origins, ray_directions = get_rays(height, width, focal_length, trans_matrix)

    # Sample points along each ray
    # (H, W, num_samples, 3) & (H, W, num_samples)
    sampled_points, depth_values = sample_points_from_rays(
        ray_origins, ray_directions, near_point, far_point, num_depth_samples_per_ray
    )

    # "Flatten" the sampled points, (H * W * num_samples, 3)
    flattened_sampled_points = sampled_points.reshape((-1, 3))

    # Encode the sampled points (default: positional encoding). (H * W * num_samples, encode_dim)
    encoded_points = encoder(flattened_sampled_points)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=16384)
    predictions = []
    for batch in batches:
        batch = batch.astype("float32")
        predictions.append(model(batch))
    # radiance_field_flattened = torch.cat(predictions, dim=0) # (H*W*num_samples, 4)
    op = mindspore.ops.Concat(axis=0)
    radiance_field_flattened = op(predictions)  # (H*W*num_samples, 4)
    # "Unflatten" the radiance field.
    unflattened_shape = list(sampled_points.shape[:-1]) + [4] # (H, W, num_samples, 4)
    # radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape) # (H, W, num_samples, 4)
    radiance_field = mindspore.ops.reshape(radiance_field_flattened, tuple(unflattened_shape))
    # Perform differentiable volume rendering to re-synthesize the RGB image. # (H, W, 3)
    rgb_predicted, _, _ = volume_rendering(radiance_field, ray_origins, depth_values)

    return rgb_predicted
#
#


def train(images, poses, hwf, i_split, near_point,
          far_point, num_depth_samples_per_ray, encoder,
          num_iters, model, DEVICE="cuda"):
    # Image information
    H, W, focal_length = hwf
    H = int(H)
    W = int(W)
    i_train, i_val, i_test = i_split

    # Optimizer parameters
    lr = 5e-3

    # Misc parameters
    display_every = 100  # Number of iters after which stats are displayed

    # Define Adam optimizer
    optimizer = mindspore.nn.Adam(params=model.trainable_params(),learning_rate=lr)

    # Seed RNG, for repeatability
    seed = 9458
    mindspore.set_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    # Use the first test images for visualization
    test_idx = len(i_train)
    test_img_rgb = images[test_idx, ..., :3]
    test_pose = poses[test_idx]

    def model_forward_func(H, W, focal_length, train_pose, near_point,
                           far_point, num_depth_samples_per_ray,
                            train_img_rgb):
        encoder = lambda x: positional_encoding(x, include_input=True, freq=6)
        rgb_predicted = tinynerf_step_forward(H, W, focal_length,
                                              train_pose, near_point,
                                              far_point, num_depth_samples_per_ray,
                                              encoder, get_minibatches, model)
        lossfunction = mindspore.nn.MSELoss(reduction='none')
        loss = lossfunction(rgb_predicted, train_img_rgb)
        return loss, rgb_predicted

    for i in range(num_iters):
      # Randomly pick a training image as the target, get rgb value and camera pose
      train_idx = np.random.randint(len(i_train))
      train_img_rgb = images[train_idx, ..., :3]
      train_pose = poses[train_idx]
      # 定义梯度函数
      grad_fn = mindspore.ops.value_and_grad(model_forward_func,None,optimizer.parameters,has_aux=True)
      train_pose = train_pose.astype(mindspore.float32)
      train_img_rgb = train_img_rgb.astype(mindspore.float32)
      focal_length = focal_length.astype(mindspore.float32)

      (loss,_),grads = grad_fn(H, W, focal_length,train_pose, near_point,
                                far_point, num_depth_samples_per_ray,
                                train_img_rgb)
      loss = mindspore.ops.depend(loss,optimizer(grads))

      # Run one iteration of TinyNeRF and get the rendered RGB image.
      # rgb_predicted = tinynerf_step_forward(H, W, focal_length,
      #                                         train_pose, near_point,
      #                                         far_point, num_depth_samples_per_ray,
      #                                         encoder, get_minibatches, model)

      # Compute mean-squared error between the predicted and target images
      # lossfunction = mindspore.nn.MSELoss(reduction='none')
      # loss = torch.nn.functional.mse_loss(rgb_predicted, train_img_rgb)

      # loss = lossfunction(rgb_predicted,train_img_rgb)
      # grads = model.grad(model.weights)
      # loss = mindspore.ops.depend(loss,model.optimizer(grads))

      # optimizer.step()
      # optimizer.zero_grad()

      # Display rendered test images and corresponding loss
      if i % display_every == 0:
        # Render test image
        rgb_predicted = tinynerf_step_forward(H, W, focal_length,
                                              test_pose, near_point,
                                              far_point, num_depth_samples_per_ray,
                                                encoder, get_minibatches, model)

        # Calculate loss and Peak Signal-to-Noise Ratio (PSNR)
        # loss = F.mse_loss(rgb_predicted, test_img_rgb)
        test_loss_f = mindspore.nn.MSELoss(reduction='none')
        test_loss = test_loss_f(rgb_predicted,test_img_rgb)
        print("Loss:", loss.item())
        psnr = -10. * mindspore.ops.log10(loss)

        psnrs.append(psnr.item())
        iternums.append(i)

        # Visualizing PSNR
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(rgb_predicted.detach().cpu().numpy())
        plt.title(f"Iteration {i}")
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title("PSNR")
        plt.show()
#
#     print('Finish training')
