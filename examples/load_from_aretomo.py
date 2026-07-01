import torch
from pathlib import Path
from torch_tilt_series import TiltSeries


# Paths to AreTomo alignment file and raw tilt stack
ALN_PATH = Path("/path/to/your/alignment.aln")
TILT_STACK_PATH = Path("/path/to/your/tilt_stack.mrc")

# Choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pixel spacing in Angstroms (required)
# AreTomo .aln files contain shifts in pixels, this converts them to Angstroms
PIXEL_SPACING = 6.192

# Load tilt series from AreTomo output
tilt_series = TiltSeries.from_aretomo_output(
    aln_path=ALN_PATH,
    pixel_spacing=PIXEL_SPACING,
    image_path=TILT_STACK_PATH,
    device=DEVICE,
)

# Project a 3D point (zyx, Angstroms, relative to tomogram center) into each tilt
points_zyx = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
projected_yx = tilt_series.project_points(points_zyx)

# Extract a subtilt-series around the point
particle_tilt_series = tilt_series.extract_particle_tilt_series(
    points_zyx, sidelength=64, return_rfft=False
)
