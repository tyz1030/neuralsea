#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings

import hydra
import numpy as np
import torch
from nerf.dataset import get_colmap_datasets, trivial_collate, get_synthetic_datasets
from nerf.ref_renderer import WaterReflectanceFieldRenderer
from nerf.stats import Stats
from omegaconf import DictConfig
from PIL import Image


CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="lego")
def main(cfg: DictConfig):

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the testing is unlikely to finish in reasonable time."
        )
        device = "cpu"

    # Initialize the Radiance Field model.
    model = WaterReflectanceFieldRenderer(
        image_size=cfg.data.image_size,
        n_pts_per_ray=cfg.raysampler.n_pts_per_ray,
        n_rays_per_image=cfg.raysampler.n_rays_per_image,
        near_bounding_in_z=cfg.raysampler.near_bounding_in_z,
        near_to_far_range_in_z=cfg.raysampler.near_to_far_range_in_z,
        stratified=cfg.raysampler.stratified,
        stratified_test=cfg.raysampler.stratified_test,
        chunk_size_test=cfg.raysampler.chunk_size_test,
        n_hidden_neurons_xyz=cfg.implicit_function.n_hidden_neurons_xyz,
        n_layers_xyz=cfg.implicit_function.n_layers_xyz,
        density_noise_std=cfg.implicit_function.density_noise_std,
        append_xyz = (cfg.implicit_function.skip,)
    )

    # Move the model to the relevant device.
    model.to(device)

    # Resume from the checkpoint.
    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Model checkpoint {checkpoint_path} does not exist!")

    print(f"Loading checkpoint {checkpoint_path}.")
    loaded_data = torch.load(checkpoint_path)
    # Do not load the cached xy grid.
    # - this allows setting an arbitrary evaluation image size.
    state_dict = {
        k: v
        for k, v in loaded_data["model"].items()
        if "_grid_raysampler._xy_grid" not in k
    }
    model.load_state_dict(state_dict, strict=True)

    # Load the test data.
    # if cfg.test.mode == "evaluation":
    #     _, _, test_dataset = get_colmap_datasets()
    if cfg.test.mode == "color_correction":
        if cfg.data.dataset_type == "synthetic":
            train_dataset, _, _ = get_synthetic_datasets(cfg.data.dataset_name)
        if cfg.data.dataset_type == "real":
            train_dataset, _, _ = get_colmap_datasets(cfg.data.dataset_name, cfg.data.image_down_scale, cfg.data.pose_down_scale)

        test_dataset = train_dataset
        # store the video in directory (checkpoint_file - extension + '_video')
        export_dir = os.path.splitext(checkpoint_path)[0] + "_video"
        os.makedirs(export_dir, exist_ok=True)
    else:
        raise ValueError(f"Unknown test mode {cfg.test_mode}.")

    # Init the test dataloader.
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    if cfg.test.mode == "evaluation":
        # Init the test stats object.
        # eval_stats = ["mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "sec/it"]
        eval_stats = ["mse_coarse", "psnr_coarse", "sec/it"]
        stats = Stats(eval_stats)
        stats.new_epoch()
    elif cfg.test.mode == "color_correction":
        # Init the frame buffer.
        frame_paths = []

    # Set the model to the eval mode.
    model.eval()
    # Run the main testing loop.
    for batch_idx, test_batch in enumerate(test_dataloader):
        test_image, test_camera, camera_idx, light_xyz, light_dir = test_batch[0].values()
        if test_image is not None:
            test_image = test_image.to(device)
        test_camera = test_camera.to(device)
        light_xyz = light_xyz.to(device)
        light_dir = light_dir.to(device)

        # Activate eval mode of the model (lets us do a full rendering pass).
        model.eval()
        with torch.no_grad():
            test_nerf_out, test_metrics = model(
                None,  # we do not use pre-cached cameras
                test_camera,
                test_image,
                light_xyz,
                light_dir, cfg.data.light_falloff
            )

        if cfg.test.mode == "evaluation":
            # Update stats with the validation metrics.
            stats.update(test_metrics, stat_set="test")
            stats.print(stat_set="test")

        elif cfg.test.mode == "color_correction":
            if cfg.data.dataset_type == "real":
                frame = (test_nerf_out["rgb_corrected"][0]**0.45).detach().cpu().clamp(0.0, 1.0)
            elif cfg.data.dataset_type == "synthetic":
                frame = (test_nerf_out["rgb_corrected"][0]).detach().cpu().clamp(0.0, 1.0)
            
            frame = np.flip(frame.numpy(), axis=[0,1])
            frame_path = os.path.join(export_dir, f"frame_{batch_idx:05d}.png")
            print(f"Writing {frame_path}.")
            Image.fromarray((frame * 255.0).astype(np.uint8)).save(frame_path)
            frame_paths.append(frame_path)

    if cfg.test.mode == "evaluation":
        print(f"Final evaluation metrics on '{cfg.data.dataset_name}':")
        for stat in eval_stats:
            stat_value = stats.stats["test"][stat].get_epoch_averages()[0]
            print(f"{stat:15s}: {stat_value:1.4f}")



if __name__ == "__main__":
    main()
