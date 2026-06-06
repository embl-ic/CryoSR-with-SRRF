# Jake Ankrum
# EMBL IC
# 5/22/2026

# Load the necessary libraries
import matplotlib
import numpy as np
from nanopyx.methods.esrrf.eSRRF_workflow import eSRRF
from readlif.reader import LifFile
from matplotlib import pyplot as plt
from nanopyx.methods.drift_alignment import DriftEstimator
import nanopyx.methods.esrrf.parameter_sweep as psweep
from pathlib import Path
import sys

# Supress compiler warnings for non GPU devices
import warnings
from pyopencl import CompilerWarning

warnings.filterwarnings("ignore", category=CompilerWarning)

# Set up paralel processing for eSRRF (comment out when not using HPC)
corr = sys.argv[1] if len(sys.argv) > 1 else "AVG"
print(f"Using temporal correlation method: {corr}...")

# Load the .lif file based on the path to the data directory relative to the script location
data_dir = Path(__file__).parent.resolve() / "data_raw"
lif_path = LifFile(
    data_dir / "Sample_Confocals_Oversampled" / "JF503_Lifeact_1000Hz_Analog.lif"
)

img_object = lif_path.get_image(0)

# Extarct frames as a z stack
num_timepoints = img_object.dims.t
frames_stack = []

# Append each frame to list
for t in range(num_timepoints):
    frame = img_object.get_frame(t=t)
    frames_stack.append(frame)

# Convert list to numpy array
img_stack = np.stack(frames_stack)
img_uncorrected = np.mean(img_stack, axis=0)

# Create instances of all required classes
drif_est = DriftEstimator()

# Enforce first correction
img_stack_corrected = drif_est.estimate(
    img_stack, apply=True, ref_option=0, time_averaging=10
)
img_corrected = np.mean(img_stack_corrected, axis=0)

# Graph residual drift after correction
resid = img_corrected - img_uncorrected

# Display the resulting super-resolution image
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

ax1.imshow(img_uncorrected, cmap="gray")
ax1.set_title("Average Uncorrected Stack")
ax1.axis("off")

ax2.imshow(img_corrected, cmap="gray")
ax2.set_title("Average Corrected Stack")
ax2.axis("off")

im = ax3.imshow(np.abs(resid), cmap="viridis")
ax3.cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
ax3.set_title("Residual Drift")
ax3.axis("off")

plt.savefig("driftcorrection.png", dpi=300, bbox_inches="tight")
# plt.show()

# Execute a parameter sweep to optimize the drift correction parameters for the given dataset
sensitivities, radii = psweep.run_esrrf_parameter_sweep(
    img_stack_corrected,
    magnification=5,
    sensitivities=[1, 1.5, 2, 2.5, 3, 3.5, 4],
    radii=[1, 1.5, 2, 2.5, 3, 3.5, 4],
    temporal_correlation=corr,
    plot_sweep=True,
)

print(
    f"Selecting sensitivity {sensitivities} and ring radius {radii} for eSRRF reconstruction..."
)

# Average an the image stack to get a reference image for visualization
reference_image = np.mean(img_stack, axis=0)

# Set the parameters for eSRRF
magnification = 5  # Adjust for subpixel resoltion
ring_radius = radii  # Officially calculated in parameter sweep
sensitivity = sensitivities  # Officially calcualted in parameter sweep
frames_per_timepoint = 500  # Number of frames for each SR image (greater frames can improve resolution but increase processing time)
temporal_correlation = corr  # Temporal correlation method ('AVG', 'VAR', or 'TAC2')
do_intensity_weighting = True

# Run eSRRF on the loaded image
esrrf_result = eSRRF(
    img_stack_corrected,
    magnification,
    ring_radius,
    sensitivity,
    frames_per_timepoint,
    temporal_correlation,
    do_intensity_weighting,
    _force_run_type="opencl",
)

# Display the resulting super-resolution image
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

ax1.imshow(reference_image, cmap="gray")
ax1.set_title(f"Reference Image (Average of {num_timepoints} Frames)")
ax1.axis("off")

ax2.imshow(esrrf_result, cmap="gray")
ax2.set_title(
    f"eSRRF Frames:{frames_per_timepoint} , Ring Radius: {radii}, Sensitivity: {sensitivities}, Correlation: {corr}"
)
ax2.axis("off")

plt.savefig("eSRRF.png", dpi=300, bbox_inches="tight")
# plt.show()
