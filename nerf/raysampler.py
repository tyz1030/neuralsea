# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional

import torch
# from pytorch3d.renderer import MonteCarloRaysampler, MultinomialRaysampler, RayBundle
from pytorch3d.renderer import RayBundle
from .raysampling import MonteCarloRaysampler, MultinomialRaysampler

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf, sample_pdf_python
import torch.nn.functional as F

class LightSourceRaySampler(torch.nn.Module):
    def __init__(self, n_pts_per_ray) -> None:
        super().__init__()
        self.n_pts_per_ray = n_pts_per_ray

    @torch.no_grad()
    def forward(self, 
                points_world: torch.Tensor,
                light_source_xyz: torch.Tensor,
                )->torch.Tensor:

        # batch_size = points_world.shape[0]
        # spatial_size = points_world.shape[1:-1] # number of original rays * n_points on orignal rays
        
        rays_directions_world = light_source_xyz-points_world
        rays_lengths = torch.linalg.vector_norm(rays_directions_world, dim=-1)

        rays_directions_world_normalized = rays_directions_world/rays_lengths.unsqueeze(-1)

        rays_zs = torch.linspace(
            0,1,
            self.n_pts_per_ray,
            dtype=points_world.dtype,
            device=points_world.device,
        )

        rays_zs = torch.pow(rays_zs, 2)

        if (rays_lengths.shape[0] is not 1):
            print("raysampler.py batch size has to be 1")
            import sys
            sys.exit()

        # print(rays_lengths.squeeze(0).unsqueeze(-1).shape)
        # print(rays_zs.expand(rays_lengths.shape[1], 1, rays_zs.shape[0]).shape)
        rays_zs = torch.bmm(rays_lengths.squeeze(0).unsqueeze(-1), rays_zs.expand(rays_lengths.shape[1], 1, rays_zs.shape[0])) #batch outer product
        # print(points_world.shape)
        # print(rays_directions_world_normalized.shape)
        # print(rays_zs.unsqueeze(0).shape)

        return RayBundle(
                    points_world,
                    rays_directions_world_normalized,
                    rays_zs.unsqueeze(0), # 1*n_rays*n_pts*n_pts_to_lightsource
                    torch.empty(0),
                )

class BackgroundWallRaysampler(torch.nn.Module):
    """
    Modifies the raysampler of NeRF for medium scattering effects
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: int,
        image_width: int,
        image_height: int,
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

        self._grid_raysampler = MultinomialRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        # Initialize the Monte Carlo ray sampler.
        self._mc_raysampler = MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=n_rays_per_image,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )
    
    @torch.no_grad()
    def get_n_chunks(self, chunksize: int, batch_size: int):
        """
        Returns the total number of `chunksize`-sized chunks
        of the raysampler's rays.

        Args:
            chunksize: The number of rays per chunk.
            batch_size: The size of the batch of the raysampler.

        Returns:
            n_chunks: The total number of chunks.
        """
        return int(
            math.ceil(
                (self._grid_raysampler._xy_grid.numel() * 0.5 * batch_size) / chunksize
            )
        )

    @torch.no_grad()
    def _stratify_ray_bundle(self, ray_bundle: RayBundle):
        """
        Stratifies the lengths of the input `ray_bundle`.

        More specifically, the stratification replaces each ray points' depth `z`
        with a sample from a uniform random distribution on
        `[z - delta_depth, z+delta_depth]`, where `delta_depth` is the difference
        of depths of the consecutive ray depth values.

        Args:
            `ray_bundle`: The input `RayBundle`.

        Returns:
            `stratified_ray_bundle`: `ray_bundle` whose `lengths` field is replaced
                with the stratified samples.
        """
        z_vals = ray_bundle.lengths.clone()
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        # Keep starting point and ending point unchanged by [1:-1]
        z_vals[..., 1:-1] = (lower + (upper - lower) * torch.rand_like(lower))[..., 1:-1]
        return ray_bundle._replace(lengths=z_vals)

    @torch.no_grad()
    def _normalize_by_z_depth(self, ray_bundle: RayBundle):
        ray_bundle = ray_bundle._replace(
            directions=ray_bundle.directions/torch.abs(ray_bundle.directions[..., 0, None])
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
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            chunksize: The number of rays per chunk.
                Active only when `self.training==False`.
            chunk_idx: The index of the ray chunk. The number has to be in
                `[0, self.get_n_chunks(chunksize, batch_size)-1]`.
                Active only when `self.training==False`.
            camera_hash: A unique identifier of a pre-cached camera. If `None`,
                the cache is not searched and the rays are calculated from scratch.
            caching: If `True`, activates the caching mode that returns the `RayBundle`
                that should be stored into the cache.
        Returns:
            A named tuple `RayBundle` with the following fields:
                origins: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the locations of ray origins in the world coordinates.
                directions: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the directions of each ray in the world coordinates.
                lengths: A tensor of shape
                    `(batch_size, n_rays_per_image, n_pts_per_ray)`
                    containing the z-coordinate (=depth) of each ray in world units.
                xys: A tensor of shape
                    `(batch_size, n_rays_per_image, 2)`
                    containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]  # pyre-ignore
        device = cameras.device

        if self.training:
            # Sample random rays from scratch.
            ray_bundle = self._mc_raysampler(cameras)
            ray_bundle = self._normalize_by_z_depth(ray_bundle)
        else:
            # We generate a full ray grid from scratch.
            full_ray_bundle = self._grid_raysampler(cameras)
            full_ray_bundle = self._normalize_by_z_depth(full_ray_bundle)
            n_pixels = full_ray_bundle.directions.shape[:-1].numel()

            if self.training:
                # During training we randomly subsample rays.
                sel_rays = torch.randperm(n_pixels, device=device)[
                    : self._mc_raysampler._n_rays_per_image
                ]
            else:
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

class BoundingPlaneRaysampler(torch.nn.Module):
    def __init__(
        self,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: int,
        image_width: int,
        image_height: int,
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

        self._grid_raysampler = MultinomialRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        # Initialize the Monte Carlo ray sampler.
        self._mc_raysampler = MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=n_rays_per_image,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )
    
    @torch.no_grad()
    def get_n_chunks(self, chunksize: int, batch_size: int):
        """
        Returns the total number of `chunksize`-sized chunks
        of the raysampler's rays.

        Args:
            chunksize: The number of rays per chunk.
            batch_size: The size of the batch of the raysampler.

        Returns:
            n_chunks: The total number of chunks.
        """
        return int(
            math.ceil(
                (self._grid_raysampler._xy_grid.numel() * 0.5 * batch_size) / chunksize
            )
        )

    @torch.no_grad()
    def _stratify_ray_bundle(self, ray_bundle: RayBundle):
        """
        Stratifies the lengths of the input `ray_bundle`.

        More specifically, the stratification replaces each ray points' depth `z`
        with a sample from a uniform random distribution on
        `[z - delta_depth, z+delta_depth]`, where `delta_depth` is the difference
        of depths of the consecutive ray depth values.

        Args:
            `ray_bundle`: The input `RayBundle`.

        Returns:
            `stratified_ray_bundle`: `ray_bundle` whose `lengths` field is replaced
                with the stratified samples.
        """
        z_vals = ray_bundle.lengths.clone()
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        # Keep starting point and ending point unchanged by [1:-1]
        z_vals[..., 1:-1] = (lower + (upper - lower) * torch.rand_like(lower))[..., 1:-1]
        return ray_bundle._replace(lengths=z_vals)

    @torch.no_grad()
    def _normalize_by_z_depth(self, ray_bundle: RayBundle):
        ray_bundle = ray_bundle._replace(
            directions=ray_bundle.directions/torch.abs(ray_bundle.directions[..., -1].unsqueeze(-1))
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
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            chunksize: The number of rays per chunk.
                Active only when `self.training==False`.
            chunk_idx: The index of the ray chunk. The number has to be in
                `[0, self.get_n_chunks(chunksize, batch_size)-1]`.
                Active only when `self.training==False`.
            camera_hash: A unique identifier of a pre-cached camera. If `None`,
                the cache is not searched and the rays are calculated from scratch.
            caching: If `True`, activates the caching mode that returns the `RayBundle`
                that should be stored into the cache.
        Returns:
            A named tuple `RayBundle` with the following fields:
                origins: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the locations of ray origins in the world coordinates.
                directions: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the directions of each ray in the world coordinates.
                lengths: A tensor of shape
                    `(batch_size, n_rays_per_image, n_pts_per_ray)`
                    containing the z-coordinate (=depth) of each ray in world units.
                xys: A tensor of shape
                    `(batch_size, n_rays_per_image, 2)`
                    containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]  # pyre-ignore
        device = cameras.device

        if self.training:
            # Sample random rays from scratch.
            torch.manual_seed(142)
            ray_bundle = self._mc_raysampler(cameras)
            ray_bundle = self._normalize_by_z_depth(ray_bundle)
            print(ray_bundle.directions.shape)
            print(cameras.R)
            r_trans = torch.transpose(cameras.R, 1, 2)
            print(r_trans.shape)
            print(ray_bundle.directions.shape)
            out = torch.matmul(ray_bundle.directions.squeeze(), cameras.R.squeeze()) 
            print(torch.min(out/out[:, -1, None], dim=0))
            import sys
            sys.exit()
        else:
            # We generate a full ray grid from scratch.
            full_ray_bundle = self._grid_raysampler(cameras)
            full_ray_bundle = self._normalize_by_z_depth(full_ray_bundle)
            
            print(ray_bundle.directions.shape)
            print(cameras.R)
            r_trans = torch.transpose(cameras.R, 1, 2)
            print(r_trans.shape)
            print(ray_bundle.directions.shape)
            out = torch.matmul(ray_bundle.directions.squeeze(), cameras.R.squeeze()) 
            print(torch.min(out/out[:, -1, None], dim=0))

            n_pixels = full_ray_bundle.directions.shape[:-1].numel()


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


class BoundingPlaneRaysamplerCustom(torch.nn.Module):
    def __init__(
        self,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: int,
        image_width: int,
        image_height: int,
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
        z_vals[..., 1:-1] = (lower + (upper - lower) * torch.rand_like(lower))[..., 1:-1]
        return ray_bundle._replace(lengths=z_vals)

    @torch.no_grad()
    def _normalize_by_z_depth(self, ray_bundle: RayBundle):
        ray_bundle = ray_bundle._replace(
            directions=ray_bundle.directions/torch.abs(ray_bundle.directions[..., -1].unsqueeze(-1))
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
        depths = torch.linspace(0,1,self.n_pts_per_ray, device = cameras.device)
        
        # z_to_foreground = -z_to_center+0.5
        # z_foreg_to_back = 0.3

        z_to_foreground = -z_to_center-0.2
        z_foreg_to_back = 0.4

        depths = z_to_foreground + depths*z_foreg_to_back
        if self.training:
            xys = 2*torch.rand([self.n_rays_per_image, 2])-1.0
            temp_xys = torch.concat([xys[:, 0, None], xys[:, 1, None]], dim=1)
            xy_coords = torch.tensor([[self.image_width, self.image_height]])*temp_xys/2/cameras.focal_length.cpu()

            ray_dirs_cam = torch.cat([xy_coords, torch.ones_like(temp_xys)[:, 0, None]], dim = 1)
            ray_dirs_world = torch.matmul(ray_dirs_cam.cuda(), cameras.R[0].transpose(0, 1))
            rays_dirs_world_normalized = ray_dirs_world/torch.abs(ray_dirs_world[:, 2])[..., None]
            rays_zs = depths[None, None].expand(batch_size, self.n_rays_per_image, self.n_pts_per_ray)

            ray_bundle = RayBundle(
                    origins=origin.expand(self.n_rays_per_image, 3),
                    directions=rays_dirs_world_normalized,
                    lengths=rays_zs.unsqueeze(0), # 1*n_rays*n_pts*n_pts_to_lightsource
                    xys=xys.cuda()[None, ...],
                )
        else:

            grid_y, grid_x = torch.meshgrid(torch.range(0.5, self.image_height-0.5), torch.range(0.5, self.image_width-0.5), indexing='ij')
            xy_grid = torch.cat([grid_x[..., None], grid_y[..., None]], dim = -1)
            xys = xy_grid/torch.tensor([self.image_width, self.image_height])-0.5
            
            xy_grid_cam = xy_grid-torch.tensor([self.image_width, self.image_height])/2
            xy_coords = xy_grid_cam.view(self.image_width*self.image_height, 2)/cameras.focal_length.cpu()
            
            ray_dirs_cam = torch.cat([ xy_coords, torch.ones_like(xy_coords)[:, 0, None]], dim = 1)
            ray_dirs_world = torch.matmul(ray_dirs_cam.cuda(), cameras.R[0].transpose(0, 1))
            # ray_dirs_world = torch.matmul(ray_dirs_cam.cuda(), cameras.R[0])
            rays_dirs_world_normalized = ray_dirs_world/torch.abs(ray_dirs_world[:, 2])[..., None]
            rays_zs = depths[None, None].expand(batch_size, self.image_width*self.image_height, self.n_pts_per_ray)

            # print(ray_dirs_cam)
            # print(ray_dirs_cam.shape)
            # import sys
            # sys.exit()
            full_ray_bundle = RayBundle(
                    origins=origin.expand(self.image_width*self.image_height, 3)[None, ...],
                    directions=rays_dirs_world_normalized[None, ...],
                    lengths=rays_zs, # 1*n_rays*n_pts*n_pts_to_lightsource
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


class ProbabilisticRaysampler(torch.nn.Module):
    """
    Implements the importance sampling of points along rays.
    The input is a `RayBundle` object with a `ray_weights` tensor
    which specifies the probabilities of sampling a point along each ray.

    This raysampler is used for the fine rendering pass of NeRF.
    As such, the forward pass accepts the RayBundle output by the
    raysampling of the coarse rendering pass. Hence, it does not
    take cameras as input.
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        stratified: bool,
        stratified_test: bool,
        add_input_samples: bool = True,
    ):
        """
        Args:
            n_pts_per_ray: The number of points to sample along each ray.
            stratified: If `True`, the input `ray_weights` are assumed to be
                sampled at equidistant intervals.
            stratified_test: Same as `stratified` with the difference that this
                setting is applied when the module is in the `eval` mode
                (`self.training==False`).
            add_input_samples: Concatenates and returns the sampled values
                together with the input samples.
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._stratified = stratified
        self._stratified_test = stratified_test
        self._add_input_samples = add_input_samples

    def forward(
        self,
        input_ray_bundle: RayBundle,
        ray_weights: torch.Tensor,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            input_ray_bundle: An instance of `RayBundle` specifying the
                source rays for sampling of the probability distribution.
            ray_weights: A tensor of shape
                `(..., input_ray_bundle.legths.shape[-1])` with non-negative
                elements defining the probability distribution to sample
                ray points from.

        Returns:
            ray_bundle: A new `RayBundle` instance containing the input ray
                points together with `n_pts_per_ray` additional sampled
                points per ray.
        """

        # Calculate the mid-points between the ray depths.
        z_vals = input_ray_bundle.lengths
        batch_size = z_vals.shape[0]

        # Carry out the importance sampling.
        with torch.no_grad():
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf_python(
                z_vals_mid.view(-1, z_vals_mid.shape[-1]),
                ray_weights.view(-1, ray_weights.shape[-1])[..., 1:-1],
                self._n_pts_per_ray,
                det=not (
                    (self._stratified and self.training)
                    or (self._stratified_test and not self.training)
                ),
            ).view(batch_size, z_vals.shape[1], self._n_pts_per_ray)

        if self._add_input_samples:
            # Add the new samples to the input ones.
            z_vals = torch.cat((z_vals, z_samples), dim=-1)
        else:
            z_vals = z_samples
        # Resort by depth.
        z_vals, _ = torch.sort(z_vals, dim=-1)

        return RayBundle(
            origins=input_ray_bundle.origins,
            directions=input_ray_bundle.directions,
            lengths=z_vals,
            xys=input_ray_bundle.xys,
        )

class NeRFRaysampler(torch.nn.Module):
    """
    Implements the raysampler of NeRF.

    Depending on the `self.training` flag, the raysampler either samples
    a chunk of random rays (`self.training==True`), or returns a subset of rays
    of the full image grid (`self.training==False`).
    The chunking of rays allows for efficient evaluation of the NeRF implicit
    surface function without encountering out-of-GPU-memory errors.

    Additionally, this raysampler supports pre-caching of the ray bundles
    for a set of input cameras (`self.precache_rays`).
    Pre-caching the rays before training greatly speeds-up the ensuing
    raysampling step of the training NeRF iterations.
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: int,
        image_width: int,
        image_height: int,
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

        self._grid_raysampler = MultinomialRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        # Initialize the Monte Carlo ray sampler.
        self._mc_raysampler = MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=n_rays_per_image,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        # create empty ray cache
        self._ray_cache = {}

    def get_n_chunks(self, chunksize: int, batch_size: int):
        """
        Returns the total number of `chunksize`-sized chunks
        of the raysampler's rays.

        Args:
            chunksize: The number of rays per chunk.
            batch_size: The size of the batch of the raysampler.

        Returns:
            n_chunks: The total number of chunks.
        """
        return int(
            math.ceil(
                (self._grid_raysampler._xy_grid.numel() * 0.5 * batch_size) / chunksize
            )
        )

    def _print_precaching_progress(self, i, total, bar_len=30):
        """
        Print a progress bar for ray precaching.
        """
        position = round((i + 1) / total * bar_len)
        pbar = "[" + "█" * position + " " * (bar_len - position) + "]"
        print(pbar, end="\r")

    def precache_rays(self, cameras: List[CamerasBase], camera_hashes: List):
        """
        Precaches the rays emitted from the list of cameras `cameras`,
        where each camera is uniquely identified with the corresponding hash
        from `camera_hashes`.

        The cached rays are moved to cpu and stored in `self._ray_cache`.
        Raises `ValueError` when caching two cameras with the same hash.

        Args:
            cameras: A list of `N` cameras for which the rays are pre-cached.
            camera_hashes: A list of `N` unique identifiers of each
                camera from `cameras`.
        """
        print(f"Precaching {len(cameras)} ray bundles ...")
        full_chunksize = (
            self._grid_raysampler._xy_grid.numel()
            // 2
            * self._grid_raysampler._n_pts_per_ray
        )
        if self.get_n_chunks(full_chunksize, 1) != 1:
            raise ValueError("There has to be one chunk for precaching rays!")
        for camera_i, (camera, camera_hash) in enumerate(zip(cameras, camera_hashes)):
            ray_bundle = self.forward(
                camera,
                caching=True,
                chunksize=full_chunksize,
            )
            if camera_hash in self._ray_cache:
                raise ValueError("There are redundant cameras!")
            self._ray_cache[camera_hash] = RayBundle(
                *[v.to("cpu").detach() for v in ray_bundle]
            )
            self._print_precaching_progress(camera_i, len(cameras))
        print("")

    def _stratify_ray_bundle(self, ray_bundle: RayBundle):
        """
        Stratifies the lengths of the input `ray_bundle`.

        More specifically, the stratification replaces each ray points' depth `z`
        with a sample from a uniform random distribution on
        `[z - delta_depth, z+delta_depth]`, where `delta_depth` is the difference
        of depths of the consecutive ray depth values.

        Args:
            `ray_bundle`: The input `RayBundle`.

        Returns:
            `stratified_ray_bundle`: `ray_bundle` whose `lengths` field is replaced
                with the stratified samples.
        """
        z_vals = ray_bundle.lengths
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        z_vals = lower + (upper - lower) * torch.rand_like(lower)
        return ray_bundle._replace(lengths=z_vals)

    def _normalize_raybundle(self, ray_bundle: RayBundle):
        """
        Normalizes the ray directions of the input `RayBundle` to unit norm.
        """
        ray_bundle = ray_bundle._replace(
            directions=torch.nn.functional.normalize(ray_bundle.directions, dim=-1)
        )
        return ray_bundle

    def forward(
        self,
        cameras: CamerasBase,
        chunksize: int = None,
        chunk_idx: int = 0,
        camera_hash: str = None,
        caching: bool = False,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            chunksize: The number of rays per chunk.
                Active only when `self.training==False`.
            chunk_idx: The index of the ray chunk. The number has to be in
                `[0, self.get_n_chunks(chunksize, batch_size)-1]`.
                Active only when `self.training==False`.
            camera_hash: A unique identifier of a pre-cached camera. If `None`,
                the cache is not searched and the rays are calculated from scratch.
            caching: If `True`, activates the caching mode that returns the `RayBundle`
                that should be stored into the cache.
        Returns:
            A named tuple `RayBundle` with the following fields:
                origins: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the locations of ray origins in the world coordinates.
                directions: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the directions of each ray in the world coordinates.
                lengths: A tensor of shape
                    `(batch_size, n_rays_per_image, n_pts_per_ray)`
                    containing the z-coordinate (=depth) of each ray in world units.
                xys: A tensor of shape
                    `(batch_size, n_rays_per_image, 2)`
                    containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]  # pyre-ignore
        device = cameras.device

        if (camera_hash is None) and (not caching) and self.training:
            # Sample random rays from scratch.
            ray_bundle = self._mc_raysampler(cameras, min_depth=min_depth, max_depth = max_depth)
            ray_bundle = self._normalize_raybundle(ray_bundle)
        else:
            if camera_hash is not None:
                # The case where we retrieve a camera from cache.
                if batch_size != 1:
                    raise NotImplementedError(
                        "Ray caching works only for batches with a single camera!"
                    )
                full_ray_bundle = self._ray_cache[camera_hash]
            else:
                # We generate a full ray grid from scratch.
                full_ray_bundle = self._grid_raysampler(cameras, min_depth=min_depth, max_depth = max_depth)
                full_ray_bundle = self._normalize_raybundle(full_ray_bundle)
            #     print("aherere")
            # print(torch.max(full_ray_bundle.xys[..., 1]))
            # print((full_ray_bundle.xys.shape))
            n_pixels = full_ray_bundle.directions.shape[:-1].numel()

            if self.training:
                # During training we randomly subsample rays.
                sel_rays = torch.randperm(n_pixels, device=device)[
                    : self._mc_raysampler._n_rays_per_image
                ]
            else:
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

        if (
            (self._stratified and self.training)
            or (self._stratified_test and not self.training)
        ) and not caching:  # Make sure not to stratify when caching!
            ray_bundle = self._stratify_ray_bundle(ray_bundle)

        # print(torch.max(ray_bundle.xys))
        # print("---------------------")
        return ray_bundle
