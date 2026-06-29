"""Tilt series geometry, projection and subtilt extraction for cryo-ET."""

from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_3d import Ry, Rz, T
from torch_grid_utils import dft_center
from torch_subpixel_crop import subpixel_crop_2d


def _as_tensor(data, device: torch.device | str) -> torch.Tensor:
    if isinstance(data, np.ndarray) and not data.flags.writeable:
        data = data.copy()
    return torch.as_tensor(data, device=device).float()


def _normalize_on_central_crop(tilt_stack: np.ndarray) -> np.ndarray:
    # Normalize on central 25% crop to avoid edge artifacts
    h, w = tilt_stack.shape[-2:]
    h_crop = slice(int(0.375 * h), int(0.625 * h))
    w_crop = slice(int(0.375 * w), int(0.625 * w))
    crop = tilt_stack[:, h_crop, w_crop]
    tilt_stack -= np.mean(crop, axis=(-2, -1), keepdims=True)
    tilt_stack /= np.std(crop, axis=(-2, -1), keepdims=True)
    return tilt_stack


class TiltSeries:
    """Tilt series that enables projection and subtilt extraction."""

    def __init__(
        self,
        tilt_angles: torch.Tensor,
        tilt_axis_angle: torch.Tensor,
        sample_translations: torch.Tensor,
        images: torch.Tensor,  # (b, h, w)
        pixel_spacing: float,
        device: torch.device | str = "cpu",
    ):
        self.images = _as_tensor(images, device)
        self.tilt_angles = _as_tensor(tilt_angles, device)
        self.tilt_axis_angle = _as_tensor(tilt_axis_angle, device)
        self.sample_translations = _as_tensor(sample_translations, device)
        self.pixel_spacing = pixel_spacing
        self.device = device

    @property
    def sample_translations_px(self) -> torch.Tensor:
        """Sample translations in pixels."""
        return self.sample_translations / self.pixel_spacing

    @classmethod
    def from_aretomo_output(
        cls,
        aln_path: Path | str,
        pixel_spacing: float,
        image_path: Path | str | None = None,
        device: torch.device | str = "cpu",
    ) -> "TiltSeries":
        """Initialize TiltSeries from an AreTomo .aln file."""
        import alnfile
        import mrcfile

        aln_path = Path(aln_path)
        df = alnfile.read(aln_path)

        if image_path is None:
            image_path = aln_path.with_suffix(".mrc")

        # Extract XY shifts and convert to YX convention
        corrected_shifts_xy = df[["tx", "ty"]].to_numpy()
        corrected_shifts_yx = corrected_shifts_xy[:, ::-1].copy()

        # Convert shifts from pixels to Angstroms
        corrected_shifts_yx_ang = corrected_shifts_yx * pixel_spacing

        # Load tilt stack and extract valid tilts
        tilt_stack_full = mrcfile.read(image_path)
        idx_valid = df["sec"].values - 1  # Convert from 1-indexed to 0-indexed
        tilt_stack = tilt_stack_full[idx_valid]
        tilt_stack = tilt_stack.astype(np.float32)

        tilt_stack = _normalize_on_central_crop(tilt_stack)
        return cls(
            images=tilt_stack,
            tilt_angles=df["tilt"].to_numpy(),
            tilt_axis_angle=df["rot"].to_numpy(),
            sample_translations=corrected_shifts_yx_ang,
            pixel_spacing=pixel_spacing,
            device=device,
        )

    @classmethod
    def from_etomo_directory(
        cls,
        etomo_dir: Path | str,
        pixel_spacing: float,
        device: torch.device | str = "cpu",
    ) -> "TiltSeries":
        """Initialize TiltSeries from an ETOMO directory."""
        import etomofiles
        import mrcfile

        etomo_dir = Path(etomo_dir)
        df = etomofiles.read(etomo_dir)
        df = df.loc[~df["excluded"]].reset_index(drop=True)
        # Get IMOD xf components from dataframe
        # df_to_xf(df, yx=True) returns (n_tilts, 2, 3) array
        # Each matrix is [[A22, A21, DY], [A12, A11, DX]] (ready for torch-tomogram yz)
        xf = etomofiles.df_to_xf(df, yx=True)
        m, shifts = xf[:, :, :2], xf[:, :, 2]
        # Convert IMOD's backward projection model to torch-tomogram's forward model
        # IMOD: image -> sample
        #   > the 2d matrix from the .xf file represents a 2d transform to align
        #   > the images so that they represent projections of a solid body
        #   > tilted around the Y axis
        # torch-tomogram: sample -> image
        #   > the shifts are applied after rotation and projection and shift the
        #   > projected sample to the image position
        #
        #  Roation matrix are orthogonal, so inversion = transposition :
        #  np.einsum('nij,nj->ni', np.linalg.inv(m), shifts)
        #    = np.einsum('nji,nj->ni', m, shifts)
        #
        #  Negate shifts for forward projection model
        corrected_shifts = -np.einsum('nji,nj->ni', m, shifts)
        corrected_shifts = np.ascontiguousarray(corrected_shifts)

        # Convert shifts from pixels to Angstroms
        corrected_shifts_ang = corrected_shifts * pixel_spacing

        # Load tilt stack
        tilt_stack_name = df['image_path'][0].split("[")[0]
        tilt_stack_path = Path(tilt_stack_name)
        if not tilt_stack_path.is_absolute():
            tilt_stack_path = etomo_dir / tilt_stack_path
        tilt_stack_full = mrcfile.read(tilt_stack_path)
        tilt_stack = tilt_stack_full[df['idx_tilt'].to_numpy()]
        tilt_stack = tilt_stack.astype(np.float32)

        tilt_stack = _normalize_on_central_crop(tilt_stack)

        return cls(
            images=tilt_stack,
            tilt_angles=df['tlt'].to_numpy(),
            tilt_axis_angle=df['tilt_axis_angle'].to_numpy(),
            sample_translations=corrected_shifts_ang,
            pixel_spacing=pixel_spacing,
            device=device,
        )

    @property
    def projection_matrices(self) -> torch.Tensor:
        """Matrices that project points from 3D -> 2D."""
        shifts_3d = F.pad(self.sample_translations_px, (1, 0), value=0)
        r0 = Ry(self.tilt_angles, zyx=True, device=self.device)
        r1 = Rz(self.tilt_axis_angle, zyx=True, device=self.device)
        t2 = T(shifts_3d, device=self.device)
        return t2 @ r1 @ r0

    def to(self, device: torch.device | str) -> None:
        """Move all objects of the tilt series to the device."""
        self.device = device
        self.images = self.images.to(device)
        self.tilt_angles = self.tilt_angles.to(device)
        self.tilt_axis_angle = self.tilt_axis_angle.to(device)
        self.sample_translations = self.sample_translations.to(device)

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
