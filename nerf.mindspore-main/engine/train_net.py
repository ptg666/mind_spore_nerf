# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train step for nerf"""

import mindspore as md
from mindspore import nn
from mindspore.ops import operations as P
from engine import metrics

__all__ = ["train_net", "RendererWithCriterion"]


def train_net(config, iter_, train_renderer, optimizer, rays, gt):
    """
    Train a network.

    Args:
        config (Config): configuration.
        iter_ (int): current iterations.
        renderer (Callable): a volume renderer.
        optimizer (Optimizer): a network optimizer.
        rays (Tensor): a batch of rays for training. (#rays * #samples, 6)
        gt (Tensor): the ground truth.

    Returns:
        Tuple of 2 float, float], recorded metrics.
        - **loss** (float), loss to be recorded.
        - **psnr** (float), psnr to be recorded.
    """
    # [depth,depth_batchsize,rgb,rgb_batch]
    loss = train_renderer(rays, gt)
    # Update learning rate
    decay_rate = 0.1
    decay_steps = config.l_rate_decay * 1000
    new_l_rate = config.l_rate * (decay_rate**(iter_ / decay_steps))
    optimizer.learning_rate = md.Parameter(new_l_rate)

    return float(loss), float(metrics.psnr_from_mse(loss))


# class DepthSmoothnessLoss(nn.loss.LossBase):
#     def __init__(self):
#         super(DepthSmoothnessLoss, self).__init__()
#         self.diff_op = P.Sub()
#         self.abs_op = P.Abs()
#         self.sum_op = P.ReduceSum()
#         self.mean_op = P.ReduceMean()
#     def construct(self, depth):
#         depth = depth.reshape((1, 1, depth.shape[0], depth.shape[1]))
#         depth_dx = self.diff_op(depth[:, :, :, :-1], depth[:, :, :, 1:])
#         depth_dy = self.diff_op(depth[:, :, :-1, :], depth[:, :, 1:, :])
#         depth_smoothness = self.abs_op(depth_dx) + self.abs_op(depth_dy)
#         loss = self.mean_op(self.sum_op(depth_smoothness, (1, 2, 3)))
#         return loss


class RendererWithCriterion(nn.Cell):
    """Renderer with criterion.

    Args:
        renderer (nn.Cell): renderer.
        loss_fn (nn.Cell, optional): loss function. Defaults to nn.MSELoss().

    Inputs:
        rays (Tensor): rays tensor.
        gt (Tensor): ground truth tensor.

    Outputs:
        Tensor, loss.
    """

    def __init__(self, renderer, loss_fn=nn.MSELoss()):
        """Renderer with criterion."""
        super().__init__()
        self.renderer = renderer
        self.loss_fn = loss_fn
        # self.smooth_depth = DepthSmoothnessLoss()
    def construct(self, rays, gt):
        """Renderer Trainer construct."""
        use_ds = True
        gt_rgb = gt["rgb_gt"]
        rgb_batchsize = gt["rgb_batchsize"]
        rgb_map_fine, rgb_map_coarse, depth = self.renderer(rays)
        rgb_map_fine = rgb_map_fine[:rgb_batchsize]
        rgb_map_coarse = rgb_map_coarse[:rgb_batchsize]
        rgb_loss = self.loss_fn(rgb_map_fine, gt_rgb) + self.loss_fn(rgb_map_coarse, gt_rgb)
        if use_ds:
            gt_depth = gt["depth_gt"]
            depth_batchsize = gt["depth_batchsize"]
            depth = depth[rgb_batchsize:rgb_batchsize+depth_batchsize]
            depth_loss = self.loss_fn(depth, gt_depth) * 0.1
            total_loss = depth_loss + rgb_loss
            print(f"rgb_loss is {rgb_loss}")
            print(f"depth_loss is {depth_loss}")
        else:
            total_loss = rgb_loss
            print(f"rgb_loss is {rgb_loss}")
        return total_loss

    def backbone_network(self):
        """Return renderer."""
        return self.renderer
