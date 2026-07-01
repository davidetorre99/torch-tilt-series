"""Load TiltSeries instances from external alignment formats."""

from pathlib import Path

import numpy as np
import torch

from torch_tilt_series.tilt_series import TiltSeries


def _normalize_on_central_crop(tilt_stack: np.ndarray) -> np.ndarray:
    # Normalize on central 25% crop to avoid edge artifacts
    h, w = tilt_stack.shape[-2:]
    h_crop = slice(int(0.375 * h), int(0.625 * h))
    w_crop = slice(int(0.375 * w), int(0.625 * w))
    crop = tilt_stack[:, h_crop, w_crop]
    tilt_stack -= np.mean(crop, axis=(-2, -1), keepdims=True)
    tilt_stack /= np.std(crop, axis=(-2, -1), keepdims=True)
    return tilt_stack


def from_aretomo_output(
    aln_path: Path | str,
    pixel_spacing: float,
    image_path: Path | str | None = None,
    device: torch.device | str = "cpu",
) -> TiltSeries:
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
    return TiltSeries(
        images=tilt_stack,
        tilt_angles=df["tilt"].to_numpy(),
        tilt_axis_angle=df["rot"].to_numpy(),
        sample_translations=corrected_shifts_yx_ang,
        pixel_spacing=pixel_spacing,
        device=device,
    )


def from_etomo_directory(
    etomo_dir: Path | str,
    pixel_spacing: float,
    device: torch.device | str = "cpu",
) -> TiltSeries:
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
    corrected_shifts = -np.einsum("nji,nj->ni", m, shifts)
    corrected_shifts = np.ascontiguousarray(corrected_shifts)

    # Convert shifts from pixels to Angstroms
    corrected_shifts_ang = corrected_shifts * pixel_spacing

    # X-axis tilt: IMOD combines a global scalar (XAXISTILT in tilt.com) with
    # per-view values (XTILTFILE / .xtilt). They are additive.
    # IMOD's XAXISTILT sign convention is opposite to our Rx convention.
    xaxistilt = -(df["xaxistilt"].iloc[0] or 0.0)
    x_tilts_perview = np.nan_to_num(df["xtilt"].to_numpy(), nan=0.0)
    x_tilts = xaxistilt + x_tilts_perview
    if np.all(x_tilts == x_tilts[0]):
        x_tilts = float(x_tilts[0])

    # Load tilt stack
    tilt_stack_name = df["image_path"][0].split("[")[0]
    tilt_stack_path = Path(tilt_stack_name)
    if not tilt_stack_path.is_absolute():
        tilt_stack_path = etomo_dir / tilt_stack_path
    tilt_stack_full = mrcfile.read(tilt_stack_path)
    tilt_stack = tilt_stack_full[df["idx_tilt"].to_numpy()]
    tilt_stack = tilt_stack.astype(np.float32)

    tilt_stack = _normalize_on_central_crop(tilt_stack)

    return TiltSeries(
        images=tilt_stack,
        tilt_angles=df["tlt"].to_numpy(),
        tilt_axis_angle=df["tilt_axis_angle"].to_numpy(),
        sample_translations=corrected_shifts_ang,
        x_tilts=x_tilts,
        pixel_spacing=pixel_spacing,
        device=device,
    )
