import numpy as np
import pytest
import torch

import torch_tilt_series
from torch_tilt_series import TiltSeries

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_tilt_series(device="cpu", size=32, square=False):
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))
    images = torch.zeros((3, size, size))
    if square:
        c = size // 2
        images[:, c - 2:c + 2, c - 2:c + 2] = 1.0
    return TiltSeries(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
        pixel_spacing=1.0,
        device=device,
    )


def test_imports_with_version():
    assert isinstance(torch_tilt_series.__version__, str)


@pytest.mark.parametrize("device", DEVICES)
def test_construction(device):
    ts = make_tilt_series(device)
    assert ts.images.shape == (3, 32, 32)
    assert ts.images.dtype == torch.float32
    assert ts.tilt_angles.shape == (3,)
    assert device in str(ts.images.device)


def test_sample_translations_px():
    tilt_angles = torch.tensor([0.0])
    sample_translations = torch.tensor([[10.0, 20.0]])
    ts = TiltSeries(
        tilt_angles=tilt_angles,
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=sample_translations,
        images=torch.zeros((1, 8, 8)),
        pixel_spacing=2.0,
    )
    assert torch.allclose(ts.sample_translations_px, torch.tensor([[5.0, 10.0]]))


@pytest.mark.parametrize("device", DEVICES)
def test_projection_matrices(device):
    ts = make_tilt_series(device)
    matrices = ts.projection_matrices
    assert matrices.shape == (3, 4, 4)
    assert matrices.dtype == torch.float32
    assert device in str(matrices.device)


def test_projection_matrix_identity_at_zero_geometry():
    ts = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
        images=torch.zeros((1, 8, 8)),
        pixel_spacing=1.0,
    )
    assert torch.allclose(ts.projection_matrices[0], torch.eye(4), atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_project_points_origin(device):
    ts = make_tilt_series(device)
    points_zyx = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    projected_yx = ts.project_points(points_zyx)
    assert projected_yx.shape == (1, 3, 2)
    assert projected_yx.dtype == torch.float32
    assert device in str(projected_yx.device)
    assert torch.allclose(projected_yx, torch.zeros_like(projected_yx), atol=1e-5)


def test_project_points_value_at_zero_tilt():
    ts = make_tilt_series()
    point = torch.tensor([[0.0, 7.0, -4.0]])
    projected_yx = ts.project_points(point)
    # tilt index 1 has tilt_angle == 0, so the in-plane (y, x) is preserved
    assert torch.allclose(projected_yx[0, 1], torch.tensor([7.0, -4.0]), atol=1e-5)


def test_project_points_batch_shapes():
    ts = make_tilt_series()
    points = torch.zeros((5, 3))
    assert ts.project_points(points).shape == (5, 3, 2)


@pytest.mark.parametrize("device", DEVICES)
def test_extract_particle_tilt_series(device):
    ts = make_tilt_series(device, square=True)
    points_zyx = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    real = ts.extract_particle_tilt_series(points_zyx, sidelength=8, return_rfft=False)
    assert real.shape == (1, 3, 8, 8)
    assert real.dtype == torch.float32
    assert device in str(real.device)
    # the square sits at the image centre, so the extracted patch is non-zero
    assert float(real.abs().sum()) > 0

    rfft = ts.extract_particle_tilt_series(points_zyx, sidelength=8, return_rfft=True)
    assert rfft.shape == (1, 3, 8, 5)
    assert rfft.dtype == torch.complex64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_move():
    ts = make_tilt_series("cpu")
    assert "cpu" == str(ts.images.device)
    ts.to("cuda")
    assert "cuda" in str(ts.images.device)
    assert "cuda" in str(ts.tilt_angles.device)
    assert "cuda" in str(ts.tilt_axis_angle.device)
    assert "cuda" in str(ts.sample_translations.device)


def test_from_aretomo_output(tmp_path):
    alnfile = pytest.importorskip("alnfile")  # noqa: F841
    mrcfile = pytest.importorskip("mrcfile")

    aln = (
        "# AreTomo Alignment / Priims bprmMn\n"
        "# RawSize = 16 16 3\n"
        "# NumPatches = 0\n"
        "# AlphaOffset =     0.00\n"
        "# BetaOffset =      0.00\n"
        "# SEC     ROT      GMAG     TX       TY     SMEAN   SFIT  SCALE  BASE   TILT\n"
        "    1   0.0000  1.00000   1.000   2.000   1.00   1.00  1.00  0.00  -30.00\n"
        "    2   0.0000  1.00000   0.000   0.000   1.00   1.00  1.00  0.00    0.00\n"
        "    3   0.0000  1.00000  -1.000  -2.000   1.00   1.00  1.00  0.00   30.00\n"
    )
    aln_path = tmp_path / "ts.aln"
    aln_path.write_text(aln)
    images = np.random.default_rng(0).normal(size=(3, 16, 16)).astype(np.float32)
    mrcfile.write(tmp_path / "ts.mrc", images, overwrite=True)

    pixel_spacing = 2.0
    ts = TiltSeries.from_aretomo_output(aln_path, pixel_spacing=pixel_spacing)

    assert ts.images.shape == (3, 16, 16)
    assert ts.pixel_spacing == pixel_spacing
    assert torch.allclose(ts.tilt_angles, torch.tensor([-30.0, 0.0, 30.0]))
    assert torch.allclose(ts.tilt_axis_angle, torch.zeros(3))
    # tx/ty are stored as (y, x) in Angstroms == (ty, tx) * pixel_spacing
    expected = torch.tensor([[2.0, 1.0], [0.0, 0.0], [-2.0, -1.0]]) * pixel_spacing
    assert torch.allclose(ts.sample_translations, expected)
