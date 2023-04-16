# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.renderer import EmissionAbsorptionRaymarcher, RayBundle
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
)
from typing import Optional, Tuple, Union


def _shifted_cumprod(x, shift: int = 1, dir: int = 0):
    """
    Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of
    ones and removes `shift` trailing elements to/from the last dimension
    of the result.
    """
    x_cumprod = torch.cumprod(x, dim=-2)
    if dir:
        x_cumprod_shift = torch.cat(
            [torch.ones_like(x_cumprod[..., :shift, :]), x_cumprod[..., shift:, :]], dim=-2
        )
    else:
        x_cumprod_shift = torch.cat(
            [torch.ones_like(x_cumprod[..., :shift, :]), x_cumprod[..., :-shift, :]], dim=-2
        )

    return x_cumprod_shift



class WaterAbsorptionOnlyRaymarcher(torch.nn.Module):
    """
    Raymarch using the Absorption-Only (AO) algorithm.

    The algorithm independently renders each ray by analyzing density and
    feature values sampled at (typically uniformly) spaced 3D locations along
    each ray. The density values `rays_densities` are of shape
    `(..., n_points_per_ray, 1)`, their values should range between [0, 1], and
    represent the opaqueness of each point (the higher the less transparent).
    The algorithm only measures the total amount of light absorbed along each ray
    and, besides outputting per-ray `opacity` values of shape `(...,)`,
    does not produce any feature renderings.

    The algorithm simply computes `total_transmission = prod(1 - rays_densities)`
    of shape `(..., 1)` which, for each ray, measures the total amount of light
    that passed through the volume.
    It then returns `opacities = 1 - total_transmission`.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, ray_bundle, raw_densities: torch.Tensor, **kwargs
    ) -> Union[None, torch.Tensor]:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray)` whose values range in [0, 1].

        Returns:
            opacities: A tensor of per-ray opacity values of shape `(..., 1)`.
                Its values range between [0, 1] and denote the total amount
                of light that has been absorbed for each ray. E.g. a value
                of 0 corresponds to the ray completely passing through a volume.
        """
        # _check_density_bounds(rays_densities)
        
        deltas = torch.cat(
            (
                ray_bundle.lengths[..., 1:] - ray_bundle.lengths[..., :-1],
                1e10 * torch.ones_like(ray_bundle.lengths[..., :1]),
            ),
            dim=-1,
        )[..., None]
        densities = (-deltas * raw_densities).exp()
        total_transmission = _shifted_cumprod(1e-10+densities)
        return total_transmission


class WaterEmissionAbsorptionRaymarcher(EmissionAbsorptionRaymarcher):
    def __init__(self) -> None:
        super().__init__()
        self.m = torch.nn.Sigmoid()
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(
        self,
        rays_bundle: RayBundle,
        raw_densities: torch.Tensor,
        reflected_light: torch.Tensor,
        sigma_den: torch.Tensor,
        rwd,
        norm_grad: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:

        ref_mask = (self.m(3*(raw_densities-3))+0.01)/1.01
        sca_mask = self.m(-3*(raw_densities-3))

        deltas = torch.cat(
            (
                (rays_bundle.lengths[..., 1:] - rays_bundle.lengths[..., :-1])*torch.linalg.vector_norm(rays_bundle.directions, dim = -1, keepdim=True),
                1e10 * torch.ones_like(rays_bundle.lengths[..., :1]),
            ),
            dim=-1,
        )[..., None]

        attn = torch.exp(-rays_bundle.lengths[..., 0].unsqueeze(-1)*torch.linalg.vector_norm(rays_bundle.directions, dim = -1, keepdim=True)*rwd*2).unsqueeze(-2)

        rays_densities = 1 - (-deltas * raw_densities * 2).exp() #
        absorption = _shifted_cumprod(eps + (-deltas * (raw_densities*ref_mask+rwd*sca_mask) *2).exp(), shift=self.surface_thickness, dir = 0)
        # absorption = _shifted_cumprod(eps + (-deltas * (raw_densities*ref_mask) *2).exp(), shift=self.surface_thickness, dir = 0)
        weights = attn*rays_densities*absorption
        # weights = rays_densities * absorption


        refined_density = ref_mask*raw_densities
        refined_rays_densities = 1 - (-deltas * refined_density*2).exp() #
        refined_absorption = _shifted_cumprod((eps) + (-deltas * (raw_densities*ref_mask+rwd*sca_mask) *2).exp(), shift=self.surface_thickness, dir = 0)
        refined_weights = attn*refined_rays_densities * refined_absorption
        
        corrected_absorption = _shifted_cumprod((eps) + (-deltas * refined_density *2).exp(), shift=self.surface_thickness, dir = 0)
        corrected_weights = refined_rays_densities * corrected_absorption

        features = (weights * reflected_light).sum(dim=-2)
        refined_features = (refined_weights * reflected_light).sum(dim=-2)
        corrected_features = (corrected_weights * reflected_light).sum(dim=-2)

        norm_map = (weights * norm_grad).sum(dim=-2)

        return features, refined_features,corrected_features, weights, norm_map



class EmissionAbsorptionNeRFRaymarcher(EmissionAbsorptionRaymarcher):
    """
    This is essentially the `pytorch3d.renderer.EmissionAbsorptionRaymarcher`
    which additionally returns the rendering weights. It also skips returning
    the computation of the alpha-mask which is, in case of NeRF, equal to 1
    everywhere.

    The weights are later used in the NeRF pipeline to carry out the importance
    ray-sampling for the fine rendering pass.

    For more details about the EmissionAbsorptionRaymarcher please refer to
    the documentation of `pytorch3d.renderer.EmissionAbsorptionRaymarcher`.
    """

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.

        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            weights: A tensor of shape `(..., n_points_per_ray)` containing
                the ray-specific emission-absorption distribution.
                Each ray distribution `(..., :)` is a valid probability
                distribution, i.e. it contains non-negative values that integrate
                to 1, such that `weights.sum(dim=-1)==1).all()` yields `True`.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities, shift=self.surface_thickness
        )
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)

        return features, weights
