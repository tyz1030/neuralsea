# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple
from torchvision import transforms
from torch.nn import functional as F

import torch
import torch.nn as nn
from pytorch3d.renderer import (ray_bundle_to_ray_points,)

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from visdom import Visdom

from .implicit_function import WaterDensityFieldHash, WaterAlbedoField, WaterNormField
from .raymarcher import WaterEmissionAbsorptionRaymarcher
from .raysampler import BoundingPlaneRaysamplerCustom
from .utils import calc_mse, calc_psnr, sample_images_at_mc_locs


class WaterReflectanceFieldRenderer(torch.nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        n_pts_per_ray: int,
        n_rays_per_image: int,
        near_bounding_in_z: float,
        near_to_far_range_in_z: float,
        stratified: bool,
        stratified_test: bool,
        chunk_size_test: int,
        n_hidden_neurons_xyz: int = 256,
        n_layers_xyz: int = 8,
        append_xyz: Tuple[int, ...] = (5,),
        density_noise_std: float = 0.0,
        visualization: bool = False,
        dataset_type = str
    ):
        super().__init__()

        # Water backscatter coefficient B_lambda (will feed into softplus)
        self.bs = nn.Parameter(torch.tensor([-4.0, -3.0, -3.0]), requires_grad=True)
        
        # Water absorption coefficient beta_lambda (will feed into softplus)
        self.absorption_coeff = nn.Parameter(torch.tensor([-1.0, -2.0, -2.0]), requires_grad=True) 

        self.softp = nn.Softplus()

        # Parse out image dimensions.
        image_height, image_width = image_size

        self.raysampler_camera = BoundingPlaneRaysamplerCustom(
            n_pts_per_ray=n_pts_per_ray,
            stratified=stratified,
            stratified_test=stratified_test,
            n_rays_per_image=n_rays_per_image,
            image_height=image_height,
            image_width=image_width,
            near_bounding_in_z=near_bounding_in_z,
            near_to_far_range_in_z=near_to_far_range_in_z,
        )


        # Init the EA raymarcher used by both passes.
        self.raymarcher_camera = WaterEmissionAbsorptionRaymarcher() #?

        self._implicit_density_function = WaterDensityFieldHash(
            n_hidden_neurons_xyz=n_hidden_neurons_xyz,
            n_layers_xyz=n_layers_xyz,
            append_xyz=append_xyz,
        )

        self._implicit_albedo_function = WaterAlbedoField()
        self._implicit_norm_function = WaterNormField()

        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size
        self.visualization = visualization

        self.dataset_type = dataset_type

    def _process_ray_chunk(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
        chunk_idx: int,
        light_xyz: torch.Tensor,
        light_dir: torch.Tensor,
        light_falloff: Optional[str],
    ) -> dict:
        """
        Samples and renders a chunk of rays.

        Args:
            camera_hash: A unique identifier of a pre-cached camera.
                If `None`, the cache is not searched and the sampled rays are
                calculated from scratch.
            camera: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
                ('batch_size', ·, ·, 3).
            chunk_idx: The index of the currently rendered ray chunk.
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.
        """

        ray_bundle = self.raysampler_camera(
            cameras=camera, 
            chunksize=self._chunk_size_test,
            chunk_idx=chunk_idx,
        )

        raw_densities, features, embeds_world = self._implicit_density_function(ray_bundle, self.absorption_coeff)

        if light_falloff=='inverse_linear':
            irradiance = 1/(ray_bundle.lengths*torch.linalg.vector_norm(ray_bundle.directions, dim = -1, keepdim=True))
        elif light_falloff == 'inverse_square':
            irradiance = 1/torch.square(ray_bundle.lengths*torch.linalg.vector_norm(ray_bundle.directions, dim = -1, keepdim=True))


        cos_lightnorm_to_ray = torch.abs(torch.matmul(F.normalize(ray_bundle.directions, dim = -1), light_dir))
        dir_arrived_light = (cos_lightnorm_to_ray.unsqueeze(-1)*irradiance).unsqueeze(-1)

        albedos = self._implicit_albedo_function(embeds_world)
        norm = self._implicit_norm_function(embeds_world)
        cos_surfacenorm_to_lightray = torch.abs(torch.sum(F.normalize(ray_bundle.directions, dim = -1).unsqueeze(-2)*norm, dim=-1))
        
        reflected_light = albedos*cos_surfacenorm_to_lightray.unsqueeze(-1)*dir_arrived_light
        
        abso_coeff = self.softp(self.absorption_coeff)
        rgb_coarse, rgb_refined, rgb_corrected, weights, norm_map = self.raymarcher_camera(ray_bundle, raw_densities, reflected_light, abso_coeff, norm)

        bs = self.softp(self.bs).cuda()

        rgb_coarse = rgb_coarse+bs
        rgb_refined = rgb_refined+bs

        rgb_coarse = torch.clamp(rgb_coarse, max=1.0, min = 0.000001)
        rgb_refined = torch.clamp(rgb_refined, max=1.0, min = 0.000001)
        rgb_corrected = torch.clamp(rgb_corrected, max=1.0, min = 0.000001)


        ###
        # If the dark part of the real world data (linear image) is rendered in low quality, then uncomment the following to do tone mapping on estimated radiance
        if self.dataset_type == "real":
            rgb_coarse = torch.pow(rgb_coarse, 0.45)
            rgb_refined = torch.pow(rgb_refined, 0.45)
            rgb_corrected = torch.pow(rgb_corrected, 0.45)

        norm_map = torch.clamp(-norm_map, max = 1.0, min = -1.0)
        norm_map = norm_map/2+0.5

        if image is not None:
            # Sample the ground truth images at the xy locations of the
            # rendering ray pixels.
            rgb_gt = sample_images_at_mc_locs(
                image[..., :3][None],
                ray_bundle.xys,
            )
            if self.dataset_type == "real":
                rgb_gt = torch.pow(rgb_gt, 0.45)
        else:
            rgb_gt = None

        out = {"rgb_coarse": rgb_coarse,"rgb_refined": rgb_refined, "rgb_gt": rgb_gt, "norm_map": norm_map, "rgb_corrected": rgb_corrected}
        if self.visualization:
            # Store the coarse rays/weights only for visualization purposes.
            out["coarse_ray_bundle"] = type(ray_bundle)(
                *[v.detach().cpu() for k, v in ray_bundle._asdict().items()]
            )
            out["coarse_weights"] = weights.detach().cpu()
        return out

    def forward(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
        light_xyz: torch.Tensor,
        light_dir: torch.Tensor,
        light_falloff: Optional[str],
    ) -> Tuple[dict, dict]:
        """
        Performs the coarse and fine rendering passes of the radiance field
        from the viewpoint of the input `camera`.
        Afterwards, both renders are compared to the input ground truth `image`
        by evaluating the peak signal-to-noise ratio and the mean-squared error.

        The rendering result depends on the `self.training` flag:
            - In the training mode (`self.training==True`), the function renders
              a random subset of image rays (Monte Carlo rendering).
            - In evaluation mode (`self.training==False`), the function renders
              the full image. In order to prevent out-of-memory errors,
              when `self.training==False`, the rays are sampled and rendered
              in batches of size `chunksize`.

        Args:
            camera_hash: A unique identifier of a pre-cached camera.
                If `None`, the cache is not searched and the sampled rays are
                calculated from scratch.
            camera: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
                ('batch_size', ·, ·, 3).
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.

                The shape of `rgb_coarse`, `rgb_fine`, `rgb_gt` depends on the
                `self.training` flag:
                    If `==True`, all 3 tensors are of shape
                    `(batch_size, n_rays_per_image, 3)` and contain the result
                    of the Monte Carlo training rendering pass.
                    If `==False`, all 3 tensors are of shape
                    `(batch_size, image_size[0], image_size[1], 3)` and contain
                    the result of the full image rendering pass.
            metrics: `dict` containing the error metrics comparing the fine and
                coarse renders to the ground truth:
                `mse_coarse`: Mean-squared error between the coarse render and
                    the input `image`
                `mse_fine`: Mean-squared error between the fine render and
                    the input `image`
                `psnr_coarse`: Peak signal-to-noise ratio between the coarse render and
                    the input `image`
                `psnr_fine`: Peak signal-to-noise ratio between the fine render and
                    the input `image`
        """
        if not self.training:
            # Full evaluation pass.
            n_chunks = self.raysampler_camera.get_n_chunks(
                self._chunk_size_test,
                camera.R.shape[0],
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                camera_hash,
                camera,
                image,
                chunk_idx,
                light_xyz,
                light_dir,
                # bounds_new,
                light_falloff
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            out = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_coarse", "rgb_refined", "rgb_gt", "norm_map", "rgb_corrected")
            }
        else:
            out = chunk_outputs[0]

        # Calc the error metrics.
        metrics = {}
        if image is not None:
            for render_pass in ("coarse", "refined"):
                for metric_name, metric_fun in zip(
                    ("mse", "psnr"), (calc_mse, calc_psnr)
                ):  
                    # denominator = (out["rgb_" + render_pass][..., :3].detach()+0.001)
                    metrics[f"{metric_name}_{render_pass}"] = metric_fun(
                        # out["rgb_" + render_pass][..., :3]/denominator,
                        # out["rgb_gt"][..., :3]/denominator,
                        out["rgb_" + render_pass][..., :3],
                        out["rgb_gt"][..., :3],
                    )

        return out, metrics



def visualize_nerf_outputs(
    nerf_out: dict, output_cache: List, viz: Visdom, visdom_env: str
):
    """
    Visualizes the outputs of the `RadianceFieldRenderer`.

    Args:
        nerf_out: An output of the validation rendering pass.
        output_cache: A list with outputs of several training render passes.
        viz: A visdom connection object.
        visdom_env: The name of visdom environment for visualization.
    """

    # Show the training images.
    ims = torch.stack([o["image"] for o in output_cache])
    ims = torch.cat(list(ims), dim=1)
    viz.image(
        ims.permute(2, 0, 1),
        env=visdom_env,
        win="images",
        opts={"title": "train_images"},
    )

    # Show the coarse and fine renders together with the ground truth images.
    ims_full = torch.cat(
        [
            (nerf_out[imvar][0].permute(2, 0, 1)).detach().cpu().clamp(0.0, 1.0)
            for imvar in ("rgb_corrected", "rgb_refined","rgb_coarse", "rgb_gt")
        ],
        dim=2,
    ).flip([1, 2])

    viz.image(
        ims_full,
        env=visdom_env,
        win="images_full",
        opts={"title": "target | coarse | fine | corrected"},
    )

    # Make a 3D plot of training cameras and their emitted rays.
    camera_trace = {
        f"camera_{ci:03d}": o["camera"].cpu() for ci, o in enumerate(output_cache)
    }
    # for ci, o in enumerate(output_cache):
    #     print(torch.sum(ray_bundle_to_ray_points(o["coarse_ray_bundle"])[0, :, -1, 0]<-0.199))
    ray_pts_trace = {
        f"ray_pts_{ci:03d}": Pointclouds(
            ray_bundle_to_ray_points(o["coarse_ray_bundle"])
            .detach()
            .cpu()
            .view(1, -1, 3)
        )
        for ci, o in enumerate(output_cache)
    }
    plotly_plot = plot_scene(
        {
            "training_scene": {
                **camera_trace,
                **ray_pts_trace,
            },
        },
        pointcloud_max_points=5000,
        pointcloud_marker_size=1,
        camera_scale=0.1,
    )
    viz.plotlyplot(plotly_plot, env=visdom_env, win="scenes")
