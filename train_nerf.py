#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import os
import pickle
import warnings

import hydra
import numpy as np
import torch
from nerf.dataset import get_colmap_datasets, trivial_collate, get_synthetic_datasets
from nerf.ref_renderer import  visualize_nerf_outputs
from nerf.stats import Stats
from omegaconf import DictConfig
from visdom import Visdom

from nerf.ref_renderer import WaterReflectanceFieldRenderer


CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="lego")
def main(cfg: DictConfig):

    # Set the relevant seeds for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the training is unlikely to finish in reasonable time."
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
        # n_hidden_neurons_dir=cfg.implicit_function.n_hidden_neurons_dir,
        n_layers_xyz=cfg.implicit_function.n_layers_xyz,
        density_noise_std=cfg.implicit_function.density_noise_std,
        visualization=cfg.visualization.visdom,
        append_xyz = (cfg.implicit_function.skip,),
        dataset_type = cfg.data.dataset_type
    )

    # Move the model to the relevant device.
    model.to(device)

    # Init stats to None before loading.
    stats = None
    optimizer_state_dict = None
    start_epoch = 0

    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_path)
    if len(cfg.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.resume and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}.")
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            stats = pickle.loads(loaded_data["stats"])
            print(f"   => resuming from epoch {stats.epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]
            start_epoch = stats.epoch

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        # weight_decay=1e-4
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # Init the stats object.
    if stats is None:
        stats = Stats(
            ["loss", "mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "sec/it"],
        )

    # Initialize the cache for storing variables needed for visualization.
    visuals_cache = collections.deque(maxlen=cfg.visualization.history_size)

    # Init the visualization visdom env.
    if cfg.visualization.visdom:
        viz = Visdom(
            server=cfg.visualization.visdom_server,
            port=cfg.visualization.visdom_port,
            use_incoming_socket=False,
        )
    else:
        viz = None


    # train_dataset, val_dataset, _ = get_colmap_datasets()
    if cfg.data.dataset_type == "synthetic":
        train_dataset, val_dataset, _ = get_synthetic_datasets(cfg.data.dataset_name)
    if cfg.data.dataset_type == "real":
        train_dataset, val_dataset, _ = get_colmap_datasets(cfg.data.dataset_name, cfg.data.image_down_scale, cfg.data.pose_down_scale)

    if cfg.data.precache_rays:
        # Precache the projection rays.
        model.eval()
        with torch.no_grad():
            for dataset in (train_dataset, val_dataset):
                cache_cameras = [e["camera"].to(device) for e in dataset]
                cache_camera_hashes = [e["camera_idx"] for e in dataset]
                model.precache_rays(cache_cameras, cache_camera_hashes)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    # The validation dataloader is just an endless stream of random samples.
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=trivial_collate,
        # sampler=torch.utils.data.RandomSampler(
        #     val_dataset,
        #     replacement=True,
        #     num_samples=cfg.optimizer.max_epochs,
        # ),
        shuffle=True,
    )

    # Set the model to the training mode.
    model.train()

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.optimizer.max_epochs):
        stats.new_epoch()  # Init a new epoch.
        for iteration, batch in enumerate(train_dataloader):
            image, camera, camera_idx, light_xyz, light_dir = batch[0].values()
            image = image.to(device)
            camera = camera.to(device)
            light_xyz = light_xyz.to(device)
            light_dir = light_dir.to(device)

            # Run the forward pass of the model.
            torch.cuda.empty_cache()
            nerf_out, metrics = model(
                camera_idx if cfg.data.precache_rays else None,
                camera,
                image,
                light_xyz, light_dir, cfg.data.light_falloff
            )

            # The loss is a sum of coarse and fine MSEs
            loss = metrics["mse_coarse"]+metrics["mse_refined"]
            # loss = metrics["mse_coarse"]

            # Take the training step.
            if True:
                optimizer.zero_grad()
            loss.backward()
            if True:
                optimizer.step()

            # Update stats with the current metrics.
            stats.update(
                {"loss": float(loss), **metrics},
                stat_set="train",
            )

            if iteration % cfg.stats_print_interval == 0:
                stats.print(stat_set="train")

            # Update the visualization cache.
            if viz is not None:
                visuals_cache.append(
                    {
                        "camera": camera.cpu(),
                        "camera_idx": camera_idx,
                        "image": image.cpu().detach(),
                        # "rgb_fine": nerf_out["rgb_fine"].cpu().detach(),
                        "rgb_coarse": nerf_out["rgb_coarse"].cpu().detach(),
                        "rgb_gt": nerf_out["rgb_gt"].cpu().detach(),
                        "coarse_ray_bundle": nerf_out["coarse_ray_bundle"],
                    }
                )

        # Adjust the learning rate.
        # lr_scheduler.step()

        # Validation
        if epoch % cfg.validation_epoch_interval == 0 and epoch > 0:
            # Sample a validation camera/image.
            val_batch = next(val_dataloader.__iter__())
            val_image, val_camera, camera_idx, light_xyz, light_dir = val_batch[0].values()
            val_image = val_image.to(device)
            val_camera = val_camera.to(device)
            light_xyz = light_xyz.to(device)
            light_dir = light_dir.to(device)
            # bounds = bounds.to(device)

            # Activate eval mode of the model (lets us do a full rendering pass).
            model.eval()
            with torch.no_grad():
            # torch.cuda.empty_cache()
                val_nerf_out, val_metrics = model(
                    camera_idx if cfg.data.precache_rays else None,
                    val_camera,
                    val_image,
                    light_xyz, light_dir, cfg.data.light_falloff)

            # Update stats with the validation metrics.
            stats.update(val_metrics, stat_set="val")
            stats.print(stat_set="val")

            if viz is not None:
                # Plot that loss curves into visdom.
                stats.plot_stats(
                    viz=viz,
                    visdom_env=cfg.visualization.visdom_env,
                    plot_file=None,
                )
                # Visualize the intermediate results.
                visualize_nerf_outputs(
                    val_nerf_out, visuals_cache, viz, cfg.visualization.visdom_env
                )
            # Set the model back to train mode.
            model.train()

        # Checkpoint.
        if (
            epoch % cfg.checkpoint_epoch_interval == 0
            and len(cfg.checkpoint_path) > 0
            and epoch > 0
        ):
            # print(f"Storing checkpoint {checkpoint_path}.")
            # print(model.state_dict().keys())
            # import sys
            # sys.exit()
            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stats": pickle.dumps(stats),
            }
            torch.save(data_to_store, checkpoint_path)


if __name__ == "__main__":
    main()
