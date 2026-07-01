"""Tilt series geometry, projection and subtilt extraction for cryo-ET."""

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_3d import Rx, Ry, Rz, T
from torch_grid_utils import dft_center
from torch_subpixel_crop import subpixel_crop_2d


def _as_tensor(data, device: torch.device | str) -> torch.Tensor:
    if isinstance(data, np.ndarray) and not data.flags.writeable:
        data = data.copy()
    return torch.as_tensor(data, device=device).float()


class TiltSeries:
    """Tilt series that enables projection and subtilt extraction."""

    def __init__(
        self,
        tilt_angles: torch.Tensor,
        tilt_axis_angle: torch.Tensor,
        sample_translations: torch.Tensor,
        images: torch.Tensor,  # (b, h, w)
        pixel_spacing: float,
        x_tilts: torch.Tensor | float = 0.0,
        device: torch.device | str = "cpu",
    ):
        self.images = _as_tensor(images, device)
        self.tilt_angles = _as_tensor(tilt_angles, device)
        self.tilt_axis_angle = _as_tensor(tilt_axis_angle, device)
        self.sample_translations = _as_tensor(sample_translations, device)
        # X-axis tilt (IMOD XAXISTILT / XTILTFILE), scalar or per-tilt, in degrees.
        self.x_tilts = _as_tensor(x_tilts, device)
        self.pixel_spacing = pixel_spacing
        self.device = device

    @property
    def sample_translations_px(self) -> torch.Tensor:
        """Sample translations in pixels."""
        return self.sample_translations / self.pixel_spacing

    @property
    def projection_matrices(self) -> torch.Tensor:
        """Matrices that project points from 3D -> 2D."""
        shifts_3d = F.pad(self.sample_translations_px, (1, 0), value=0)
        # X-axis tilt is an intrinsic property of the specimen, so it is applied
        # to sample points first (innermost), before the per-view stage tilt.
        rx = Rx(self.x_tilts, zyx=True, device=self.device)
        r0 = Ry(self.tilt_angles, zyx=True, device=self.device)
        r1 = Rz(self.tilt_axis_angle, zyx=True, device=self.device)
        t2 = T(shifts_3d, device=self.device)
        return t2 @ r1 @ r0 @ rx

    def to(self, device: torch.device | str) -> None:
        """Move all objects of the tilt series to the device."""
        self.device = device
        self.images = self.images.to(device)
        self.tilt_angles = self.tilt_angles.to(device)
        self.tilt_axis_angle = self.tilt_axis_angle.to(device)
        self.sample_translations = self.sample_translations.to(device)
        self.x_tilts = self.x_tilts.to(device)

    def project_points(self, points_zyx: torch.Tensor) -> torch.Tensor:
        """Project 3D points to 2D image coordinates.

        - points are 3D zyx coordinates
        - points are positions relative to center of tomogram
        - projected 2D points are relative to center of 2D image
        """
        points_zyx = torch.as_tensor(points_zyx, device=self.device).float()

        # Convert from Angstroms to pixels for projection
        points_zyx_px = points_zyx / self.pixel_spacing

        # Apply projection matrices
        M_yx = self.projection_matrices[..., [1, 2], :]  # (ntilts, 2, 4)
        points_zyxw = homogenise_coordinates(points_zyx_px)
        projected_yx = M_yx @ einops.rearrange(
            points_zyxw, "nparticles zyxw -> nparticles 1 zyxw 1"
        )
        projected_yx = einops.rearrange(
            projected_yx, "nparticles ntilts yx 1 -> nparticles ntilts yx"
        )
        return projected_yx  # (points, tilts, yx)

    def extract_particle_tilt_series(
        self, points_zyx: torch.Tensor, sidelength: int, return_rfft: bool = True
    ) -> torch.Tensor:
        """Extract a subtilt-series at a 3D location in the sample."""
        projected_yx = self.project_points(points_zyx)
        projected_yx += dft_center(
            self.images.shape[-2:], rfft=False, fftshift=True, device=self.device
        )
        images = subpixel_crop_2d(
            image=self.images,
            positions=projected_yx,
            sidelength=sidelength,
            return_rfft=return_rfft,
            decenter=return_rfft,
        )
        return images
