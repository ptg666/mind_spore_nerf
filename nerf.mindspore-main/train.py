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
# ==============================================================================
"""Buile and train model."""

import os
import time
import cv2
import mindspore as md
import numpy as np
from engine import RendererWithCriterion, test_net, train_net
from tqdm import tqdm
import mindspore.ops.operations as P
from data.load_llff import load_llff_data
from models import VolumeRenderer
from utils.config import get_config
from utils.engine_utils import context_setup, create_nerf
from utils.ray import generate_rays
from utils.results_handler import save_image, save_video
from utils.sampler import sample_grid_2d

import os, imageio
from pathlib import Path
from colmapUtils.read_write_model import *
from colmapUtils.read_write_dense import *
import json


def train_pipeline(config, out_dir):
    """Train nerf model: data preparation, model and optimizer preparation, and model training."""
    md.set_seed(1)

    print(">>> Loading dataset")

    if config.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(config.data_dir, config.half_res,
                                                                      config.test_skip)
        print("Loaded blender", images.shape, render_poses.shape, hwf, config.data_dir)
        i_train, i_val, i_test = i_split
        near = 2.0
        far = 6.0

        if config.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif config.dataset_type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            config.data_dir,
            config.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=config.spherify,
        )
        # ---------------blur----------------------
        images_blur = []
        for img in images:
            img = (img * 255).astype(np.uint8)
            # 对图像进行高斯模糊处理，其中(9,9)参数表示核的大小，20表示标准差
            image_blur = cv2.GaussianBlur(img, (9,9), 20)
            # 将图像从8位整数类型转换为浮点数类型，并将像素值归一化到[0,1]范围内
            image_blur = image_blur.astype(np.float32) / 255.0
            images_blur.append(image_blur)
        images_blur = np.array(images_blur)
        # -----------------------------------------
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print("Loaded llff", images.shape, render_poses.shape, hwf, config.data_dir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if config.llff_hold > 0:
            print("Auto LLFF holdout,", config.llff_hold)
            i_test = np.arange(images.shape[0])[::config.llff_hold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])

        print("DEFINING BOUNDS")
        config.no_ndc = True
        if config.no_ndc:
            near = float(np.min(bds)) * 0.9
            far = float(np.max(bds)) * 1.0
        else:
            near = 0.0
            far = 1.0
        print("NEAR FAR", near, far)

    else:
        print("Unknown dataset type", config.dataset_type, "exiting")
        return

    if config.render_test:
        render_poses = poses[i_test.tolist()]

    print(f"TRAIN views: {i_train}\nTEST views: {i_test}\nVAL views: {i_val}")

    # Cast intrinsics to right types
    cap_h, cap_w, focal = hwf
    cap_h, cap_w = int(cap_h), int(cap_w)

    hwf = [cap_h, cap_w, focal]
    # Setup logging and directory for results
    print(">>> Saving checkpoints and results in", out_dir)
    # Create output directory if not existing

    os.makedirs(out_dir, exist_ok=True)
    # Record current configuration
    with open(os.path.join(out_dir, "configs.txt"), "w+", encoding="utf-8") as config_f:
        attrs = vars(config)
        for k in attrs:
            config_f.write(f"{k} = {attrs[k]}\n")

    # Create network models, optimizer and renderer
    print(">>> Creating models")

    # Create nerf model
    (
        start_iter,
        optimizer,
        model_coarse,
        model_fine,
        embed_fn,
        embed_dirs_fn,
    ) = create_nerf(config, out_dir)
    # Training steps
    global_steps = start_iter
    # Create volume renderer
    renderer = VolumeRenderer(
        config.chunk,
        config.cap_n_samples,
        config.cap_n_importance,
        config.net_chunk,
        config.white_bkgd,
        model_coarse,
        model_fine,
        embed_fn,
        embed_dirs_fn,
        near,
        far,
    )

    renderer_with_criterion = RendererWithCriterion(renderer)
    optimizer = md.nn.Adam(
        params=renderer.trainable_params(),
        learning_rate=config.l_rate,
        beta1=0.9,
        beta2=0.999,
    )

    
    train_renderer = md.nn.TrainOneStepCell(renderer_with_criterion, optimizer)
    train_renderer.set_train()

    # Start training
    print(">>> Start training")

    cap_n_rand = config.cap_n_rand
    im_l = [images,images_blur]
    # Move training data to GPU
    # images = md.Tensor(images)
    im_l = md.Tensor(im_l)
    poses = md.Tensor(poses)
    # images_blur = md.Tensor(images_blur)
    # Maximum training iterations
    cap_n_iters = config.cap_n_iters
    if start_iter >= cap_n_iters:
        return
    train_model(config, out_dir, im_l, poses, i_train, i_test, cap_h, cap_w, focal, start_iter, optimizer,
                global_steps, renderer, train_renderer, cap_n_rand, cap_n_iters)



# -----------------------------------------load colmap_depth-----------------------------------------------------------------

def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

def load_colmap_depth(basedir, factor=8, bd_factor=.75):
    data_file = Path(basedir) / 'colmap_depth.npy'
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')
    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    poses = get_poses(images)
    _, bds_raw, _ = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)
    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D/factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "error":np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    return data_list

def get_ds(depth_gt_dict,index,rate):
    depths = depth_gt_dict[index]["depth"]
    coords = depth_gt_dict[index]["coord"]
    weights = depth_gt_dict[index]["error"]
    rand_indices = np.random.choice(len(depths), size=int(len(depths)*rate), replace=False)
    depths = depths[rand_indices]
    coords = coords[rand_indices]
    weights = weights[rand_indices]
    return depths,coords,weights

def get_rays_by_coord_np(H, W, focal, c2w, coords):
    i, j = (coords[:,0]-W*0.5)/focal, -(coords[:,1]-H*0.5)/focal
    dirs = np.stack([i,j,-np.ones_like(i)],-1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d
# --------------------------------------------------------------------------------------------------------------

def train_model(config, out_dir, images, poses, i_train, i_test, cap_h, cap_w, focal, start_iter, optimizer,
                global_steps, renderer, train_renderer, cap_n_rand, cap_n_iters):
    # -----------------------------blur images--------------------------------
    images_blur = images[1]
    images = images[0]
    # --------------------------load colmap_depth-----------------------------
    use_ds = True
    if use_ds:
        colmap_gt = load_colmap_depth(config.data_dir, factor=8, bd_factor=.75)
    # ------------------------------------------------------------------------
    """Training model iteratively"""
    with tqdm(range(1, cap_n_iters + 1)) as p_bar:
        p_bar.n = start_iter
        for _ in p_bar:
            # Show progress
            p_bar.set_description(f"Iter {global_steps + 1:d}")
            p_bar.update()

            # Start time of the current iteration
            time_0 = time.time()

            img_i = int(np.random.choice(i_train))
            # --------------blur------------------
            blur_iter = 2000
            if global_steps < blur_iter:
                target = images_blur[img_i]
                print("blur")
            else:
                target = images[img_i]
            # --------------blur------------------
            pose = poses[img_i, :3, :4]

            if cap_n_rand is not None:
                gt_map = {}
                # -----------------------load colmap_depth---------------------
                if use_ds:
                    depths,coords,weights = get_ds(colmap_gt, img_i,0.5)
                    pose_np = pose.asnumpy()
                    rays_o_ds,rays_d_ds = get_rays_by_coord_np(cap_h,cap_w,focal,pose_np,coords)
                    depth_batch = len(depths)
                    rays_o_ds = md.Tensor(rays_o_ds,dtype=md.float32)
                    rays_d_ds = md.Tensor(rays_d_ds,dtype=md.float32)
                    # batch_rays_depth = md.ops.Stack(axis=0)([rays_o_ds, rays_d_ds])
                    gt_map["depth_gt"] = md.Tensor(depths,dtype=md.float32)
                    gt_map["depth_batchsize"] = depth_batch
                # -------------------------------------------------------------
                rgb_batch = cap_n_rand
                rays_o, rays_d = generate_rays(cap_h, cap_w, focal,
                                               md.Tensor(pose))  # (cap_h, cap_w, 3), (cap_h, cap_w, 3)
                sampled_rows, sampled_cols = sample_grid_2d(cap_h, cap_w, cap_n_rand)
                rays_o = rays_o[sampled_rows, sampled_cols]  # (cap_n_rand, 3)
                rays_d = rays_d[sampled_rows, sampled_cols]  # (cap_n_rand, 3)
                target_s = target[sampled_rows, sampled_cols]  # (cap_n_rand, 3)
                gt_map["rgb_gt"] = target_s
                gt_map["rgb_batchsize"] = rgb_batch
                if use_ds:
                    # cat opt
                    rays_o = P.Concat(0)([rays_o,rays_o_ds])
                    rays_d = P.Concat(0)([rays_d, rays_d_ds])
                # stack opt
                batch_rays = md.ops.Stack(axis=0)([rays_o, rays_d])
            loss, psnr = train_net(config, global_steps, train_renderer, optimizer, batch_rays, gt_map)
            p_bar.set_postfix(time=time.time() - time_0, loss=loss, psnr=psnr)

            # Logging
            # Save training states
            if (global_steps + 1) % config.i_ckpt == 0:
                path = os.path.join(out_dir, f"{global_steps + 1:06d}.tar")

                md.save_checkpoint(
                    save_obj=renderer,
                    ckpt_file_name=path,
                    append_dict={"global_steps": global_steps},
                    async_save=True,
                )
                p_bar.write(f"Saved checkpoints at {path}")

            # Save testing results
            if (global_steps + 1) % config.i_testset == 0:
                test_save_dir = os.path.join(out_dir, f"test_{global_steps + 1:06d}")
                os.makedirs(test_save_dir, exist_ok=True)

                p_bar.write(f"Testing (iter={global_steps + 1}):")

                test_time, test_loss, test_psnr = test_net(
                    cap_h,
                    cap_w,
                    focal,
                    renderer,
                    md.Tensor(poses[i_test.tolist()]),
                    images[i_test.tolist()],
                    on_progress=lambda j, img: save_image(j, img, test_save_dir),  # pylint: disable=cell-var-from-loop
                    on_complete=lambda imgs: save_video(global_steps + 1, imgs, test_save_dir),  # pylint: disable=cell-var-from-loop
                )

                p_bar.write(
                    f"Testing results: [ Mean Time: {test_time:.4f}s, Loss: {test_loss:.4f}, PSNR: {test_psnr:.4f} ]")

            global_steps += 1


def main():
    """main function, set up config."""
    config = get_config()

    # Cuda device
    context_setup(config.gpu, config.device, getattr(md.context, config.mode))

    # Output directory
    base_dir = config.base_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Experiment name
    exp_name = config.dataset_type + "_" + config.name
    # get the experiment number
    exp_num = max([int(fn.split("_")[-1]) for fn in os.listdir(base_dir) if fn.find(exp_name) >= 0] + [0])
    if config.no_reload:
        exp_num += 1

    # Output directory
    out_dir = os.path.join(base_dir, exp_name + "_" + str(exp_num))

    # Start training pipeline
    train_pipeline(config, out_dir)


if __name__ == "__main__":
    main()
