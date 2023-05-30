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
"""nerf mlp"""

import mindspore.ops.operations as P
from mindspore import nn

__all__ = ["NeRFMLP"]


class NeRFMLP(nn.Cell):
    """
    NeRF MLP architecture. Implementation according to the official code release
    <https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105>.

    Args:
        cap_d (int, optional): model depth. Defaults to 8.
        cap_w (int, optional): model width. Defaults to 256.
        input_ch (int, optional): input channel. Defaults to 3.
        input_ch_views (int, optional): input view channel. Defaults to 3.
        output_ch (int, optional): output channel. Defaults to 4.
        skips (tuple, optional): skip connection layer index. Defaults to (4).
        use_view_dirs (bool, optional): use view directions or not. Defaults to False.

    Inputs:
        x (Tensor): query tensors: points and view directions (..., 6).

    Outputs:
        Tensor, query features (..., feature_dims).
    """

    def __init__(
            self,
            cap_d=8,
            cap_w=256,
            input_ch=3,
            input_ch_views=3,
            output_ch=4,
            skips=(4),
            use_view_dirs=False,
    ):
        super().__init__()
        self.cap_d = cap_d
        self.cap_w = cap_w
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_view_dirs = use_view_dirs

        self.pts_linears = nn.CellList([nn.Dense(in_channels=input_ch, out_channels=cap_w)] + [
            nn.Dense(in_channels=cap_w, out_channels=cap_w) if i not in
            self.skips else nn.Dense(in_channels=cap_w + input_ch, out_channels=cap_w) for i in range(cap_d - 1)
        ])

        self.views_linears = nn.CellList([nn.Dense(in_channels=input_ch_views + cap_w, out_channels=cap_w // 2)])

        if use_view_dirs:
            self.feature_linear = nn.Dense(in_channels=cap_w, out_channels=cap_w)
            self.alpha_linear = nn.Dense(in_channels=cap_w, out_channels=1)
            self.rgb_linear = nn.Dense(in_channels=cap_w // 2, out_channels=3)
        else:
            self.output_linear = nn.Dense(in_channels=cap_w, out_channels=output_ch)

    def construct(self, x):
        """NeRF MLP construct"""
        input_pts, input_views = x[..., :self.input_ch], x[..., self.input_ch:]
        h = input_pts
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = P.ReLU()(h)
            if i in self.skips:
                h = P.Concat(-1)([input_pts, h])

        if self.use_view_dirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = P.Concat(-1)([feature, input_views])

            for i, _ in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = P.ReLU()(h)

            rgb = self.rgb_linear(h)
            outputs = P.Concat(-1)([rgb, alpha])
        else:
            outputs = self.output_linear(h)

        return outputs
