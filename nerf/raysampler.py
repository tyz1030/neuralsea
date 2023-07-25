# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List
import torch
from pytorch3d.renderer import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


class BoundingPlaneRaysamplerCustom(torch.nn.Module):
    def __init__(
        self,
        n_pts_per_ray: int,
        n_rays_per_image: int,
        image_width: int,
        image_height: int,
        near_bounding_in_z: float,
        near_to_far_range_in_z: float,
        stratified: bool = False,
        stratified_test: bool = False,
    ):
        """
        Args:
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
            n_rays_per_image: Number of Monte Carlo ray samples when training
                (`self.training==True`).
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            stratified: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during training (`self.training==True`).
            stratified_test: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during evaluation (`self.training==False`).
        """

        super().__init__()
        self._stratified = stratified
        self._stratified_test = stratified_test

        self.image_width = image_width
        self.image_height = image_height
        self.n_rays_per_image = n_rays_per_image
        self.n_pts_per_ray = n_pts_per_ray

        self.near_bounding_in_z = near_bounding_in_z
        self.near_to_far_range_in_z = near_to_far_range_in_z

    @torch.no_grad()
    def get_n_chunks(self, chunksize: int, batch_size: int):
        return int(
            math.ceil(
                (self.image_width * self.image_height * batch_size) / chunksize
            )
        )

    @torch.no_grad()
    def _stratify_ray_bundle(self, ray_bundle: RayBundle):
        z_vals = ray_bundle.lengths.clone()
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        # Keep starting point and ending point unchanged by [1:-1]
        z_vals[..., 1:-1] = (lower + (upper - lower) *
                             torch.rand_like(lower))[..., 1:-1]
        return ray_bundle._replace(lengths=z_vals)

    @torch.no_grad()
    def _normalize_by_z_depth(self, ray_bundle: RayBundle):
        ray_bundle = ray_bundle._replace(
            directions=ray_bundle.directions /
            torch.abs(ray_bundle.directions[..., -1].unsqueeze(-1))
        )
        return ray_bundle

    @torch.no_grad()
    def forward(
        self,
        cameras: CamerasBase,
        chunksize: int = None,
        chunk_idx: int = 0,
        **kwargs,
    ) -> RayBundle:

        batch_size = cameras.R.shape[0]  # pyre-ignore
        device = cameras.device

        origin = -torch.matmul(cameras.T, torch.transpose(cameras.R[0], 0, 1))
        z_to_center = origin[0, -1]
        depths = torch.linspace(
            0, 1, self.n_pts_per_ray, device=cameras.device)


        ####TODO

        # Adjust the following to put the scene between near and far bounding planes
        ####
        # z_to_foreground = -z_to_center+0.6
        # z_foreg_to_back = 0.4

        z_to_foreground = -z_to_center+self.near_bounding_in_z
        # z_foreg_to_back = near_to_far_range_in_z

        depths = z_to_foreground + depths*self.near_to_far_range_in_z
        if self.training:
            xys = 2*torch.rand([self.n_rays_per_image, 2])-1.0
            temp_xys = torch.concat([xys[:, 0, None], xys[:, 1, None]], dim=1)
            xy_coords = torch.tensor(
                [[self.image_width, self.image_height]])*temp_xys/2/cameras.focal_length.cpu()

            ray_dirs_cam = torch.cat(
                [xy_coords, torch.ones_like(temp_xys)[:, 0, None]], dim=1)
            ray_dirs_world = torch.matmul(
                ray_dirs_cam.cuda(), cameras.R[0].transpose(0, 1))
            rays_dirs_world_normalized = ray_dirs_world / \
                torch.abs(ray_dirs_world[:, 2])[..., None]
            rays_zs = depths[None, None].expand(
                batch_size, self.n_rays_per_image, self.n_pts_per_ray)

            ray_bundle = RayBundle(
                origins=origin.expand(self.n_rays_per_image, 3),
                directions=rays_dirs_world_normalized,
                # 1*n_rays*n_pts*n_pts_to_lightsource
                lengths=rays_zs.unsqueeze(0),
                xys=xys.cuda()[None, ...],
            )
        else:

            grid_y, grid_x = torch.meshgrid(torch.range(
                0.5, self.image_height-0.5), torch.range(0.5, self.image_width-0.5), indexing='ij')
            xy_grid = torch.cat([grid_x[..., None], grid_y[..., None]], dim=-1)
            xys = xy_grid / \
                torch.tensor([self.image_width, self.image_height])-0.5

            xy_grid_cam = xy_grid - \
                torch.tensor([self.image_width, self.image_height])/2
            xy_coords = xy_grid_cam.view(
                self.image_width*self.image_height, 2)/cameras.focal_length.cpu()

            ray_dirs_cam = torch.cat(
                [xy_coords, torch.ones_like(xy_coords)[:, 0, None]], dim=1)
            ray_dirs_world = torch.matmul(
                ray_dirs_cam.cuda(), cameras.R[0].transpose(0, 1))
            # ray_dirs_world = torch.matmul(ray_dirs_cam.cuda(), cameras.R[0])
            rays_dirs_world_normalized = ray_dirs_world / \
                torch.abs(ray_dirs_world[:, 2])[..., None]
            rays_zs = depths[None, None].expand(
                batch_size, self.image_width*self.image_height, self.n_pts_per_ray)

            full_ray_bundle = RayBundle(
                origins=origin.expand(
                    self.image_width*self.image_height, 3)[None, ...],
                directions=rays_dirs_world_normalized[None, ...],
                lengths=rays_zs,  # 1*n_rays*n_pts*n_pts_to_lightsource
                xys=2*xys.cuda()[None, ...],
            )
            # for v in full_ray_bundle:
            #     print(v.shape)
            # import sys
            # sys.exit()

            n_pixels = full_ray_bundle.directions.shape[:-1].numel()
            # print(n_pixels)
            # In case we test, we take only the requested chunk.
            if chunksize is None:
                chunksize = n_pixels * batch_size
            start = chunk_idx * chunksize * batch_size
            end = min(start + chunksize, n_pixels)
            sel_rays = torch.arange(
                start,
                end,
                dtype=torch.long,
                device=full_ray_bundle.lengths.device,
            )

            # Take the "sel_rays" rays from the full ray bundle.
            ray_bundle = RayBundle(
                *[
                    v.view(n_pixels, -1)[sel_rays]
                    .view(batch_size, sel_rays.numel() // batch_size, -1)
                    .to(device)
                    for v in full_ray_bundle
                ]
            )
        if ((self._stratified and self.training)
                or (self._stratified_test and not self.training)):
            ray_bundle = self._stratify_ray_bundle(ray_bundle)

        return ray_bundle
