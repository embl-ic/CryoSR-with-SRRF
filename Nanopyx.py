# Jake Ankrum
# EMBL IC
# 5/22/2026

# Load the necessary libraries
import numpy as np
from nanopyx.methods.esrrf.eSRRF_workflow import eSRRF
from readlif.reader import LifFile
from matplotlib import pyplot as plt
from nanopyx.methods.drift_alignment import DriftEstimator
from custom_methods.custom_sweep_wrapper import run_esrrf_parameter_sweep as psweep
from post_analysis import analyze
from pathlib import Path
from tifffile import imwrite

# Supress compiler warnings for non GPU devices
import warnings
from pyopencl import CompilerWarning

warnings.filterwarnings("ignore", category=CompilerWarning)

# Set number of frames to temporally correlate
start = 0
end = 200

# Batch frames for
n_frames = 100

# Set number of frames to average for drift correction
n_est = 10

# Set magnificaiton (subpixel resolution)
mag = 2

# Chose method of temporal correlation
corr = "AVG"

# Load the .lif file based on the path to the data directory relative to the script location
data_dir = Path(__file__).parent.resolve() / "srrf_pauli"
lif_path = LifFile(data_dir / "20251110 Lifeact" / "FITC_LifeAct_1.lif")

# Extract a specific series from the .lif file
img_object = lif_path.get_image(1)

# Extarct frames as a z stack
num_timepoints = img_object.dims.t
frames_stack = []

# Append a predefined number of frames to stack
for t in range(0, num_timepoints):
    frame = img_object.get_frame(t=t)
    frames_stack.append(frame)

# Convert list to numpy array
img_stack = np.stack(frames_stack[start:end])
img_uncorrected = np.mean(img_stack, axis=0)

# Create instances of all required classes
drif_est = DriftEstimator()

# Enforce first correction
img_stack_corrected = drif_est.estimate(img_stack, apply=True, ref_option=5, time_averaging=n_est)
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

# plt.savefig(f"drift_{n_frames}frames_{mag}mag_{corr}.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# Execute a parameter sweep to optimize the drift correction parameters for the given dataset
sensitivities, radii, qnr, out_array = psweep(
    img_stack_corrected,
    magnification=mag,
    sensitivities=[1, 2],
    radii=[1, 2],
    temporal_correlation=corr,
    plot_sweep=True,
    n_frames=n_frames,
)

print(f"Selecting sensitivity {sensitivities} and ring radius {radii} for eSRRF reconstruction...")
print(f"Maximum QnR value from parameter sweep: {qnr:.2f}...")
print(out_array)


# Average an the image stack to get a reference image for visualization
reference_image = np.mean(img_stack, axis=0)

# Set the parameters for eSRRF
magnification = mag  # Adjust for subpixel resoltion
ring_radius = radii  # Officially calculated in parameter sweep
sensitivity = sensitivities  # Officially calcualted in parameter sweep
frames_per_timepoint = (
    n_frames  # Number of frames for each SR image (greater frames can improve resolution but increase processing time)
)
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
ax1.set_title(f"Reference Image (Average of {n_frames} Frames)")
ax1.axis("off")

ax2.imshow(esrrf_result[1], cmap="gray")
ax2.set_title(
    f"eSRRF Frames: {n_frames} , Ring Radius: {ring_radius}, Sensitivity: {sensitivity}, Correlation: {corr}, Magnification: {magnification}"
)
ax2.axis("off")

# plt.savefig(f"eSRRF_{n_frames}frames_{ring_radius}radii_{sensitivity}sens_{magnification}mag_{corr}.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# Run post imaging analysis
frc, decor, units = analyze(
    reference_image,
    esrrf_result,
    plot_error=True,
    pixel_size=108.5,
    units="nm",
    frame_1=0,
    frame_2=1,
)
print(f"FRC resolution: {frc:.3f} {units} \nDecorrelation resolution: {decor:.3f} {units}")
