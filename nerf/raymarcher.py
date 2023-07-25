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
        abso_coeff,
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

        attn = torch.exp(-rays_bundle.lengths[..., 0].unsqueeze(-1)*torch.linalg.vector_norm(rays_bundle.directions, dim = -1, keepdim=True)*abso_coeff*2).unsqueeze(-2)

        rays_densities = 1 - (-deltas * raw_densities * 2).exp() #
        absorption = _shifted_cumprod(eps + (-deltas * (raw_densities*ref_mask+abso_coeff*sca_mask) *2).exp(), shift=self.surface_thickness, dir = 0)
        # absorption = _shifted_cumprod(eps + (-deltas * (raw_densities*ref_mask) *2).exp(), shift=self.surface_thickness, dir = 0)
        weights = attn*rays_densities*absorption
        # weights = rays_densities * absorption


        refined_density = ref_mask*raw_densities
        refined_rays_densities = 1 - (-deltas * refined_density*2).exp() #
        refined_absorption = _shifted_cumprod((eps) + (-deltas * (raw_densities*ref_mask+abso_coeff*sca_mask) *2).exp(), shift=self.surface_thickness, dir = 0)
        refined_weights = attn*refined_rays_densities * refined_absorption
        
        corrected_absorption = _shifted_cumprod((eps) + (-deltas * refined_density *2).exp(), shift=self.surface_thickness, dir = 0)
        corrected_weights = refined_rays_densities * corrected_absorption

        features = (weights * reflected_light).sum(dim=-2)
        refined_features = (refined_weights * reflected_light).sum(dim=-2)
        corrected_features = (corrected_weights * reflected_light).sum(dim=-2)

        norm_map = (weights * norm_grad).sum(dim=-2)

        return features, refined_features, corrected_features, weights, norm_map
