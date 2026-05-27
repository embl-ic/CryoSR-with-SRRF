# Jake Ankrum
# EMBL IC
# 5/22/2026

# Load the necessary libraries
import numpy as np
from nanopyx import eSRRF
from readlif.reader import LifFile
from matplotlib import pyplot as plt
from pathlib import Path

# For non GPU devices
import warnings
from pyopencl import CompilerWarning

# Supress compiler warnings for non GPU devices
warnings.filterwarnings("ignore", category=CompilerWarning)

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
for t in range(num_timepoints - 990):
    frame = img_object.get_frame(t=t)
    frames_stack.append(frame)

# Convert list to numpy array
img_stack = np.stack(frames_stack)

# Average an the image stack to get a reference image for visualization
reference_image = np.mean(img_stack, axis=0)

# Set the parameters for eSRRF
magnification = 5  # Adjust for microscope magnification and pixel size
ring_radius = 1.5  # Officially calculated in deconvolution
sensitivity = 1.0  # Adjust based on noise level and desired resolution enhancement
frames_per_timepoint = 10  # Number of frames for each SR image (greater frames can improve resolution but increase processing time)
temporal_correlation = "AVG"  # Temporal correlation method ('AVG', 'VAR', or 'TAC2')
do_intensity_weighting = True

# Run eSRRF on the loaded image
esrrf_result = eSRRF(
    img_stack,
    magnification,
    ring_radius,
    sensitivity,
    frames_per_timepoint,
    temporal_correlation,
    do_intensity_weighting,
)

# Display the resulting super-resolution image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.imshow(reference_image, cmap="gray")
ax1.set_title(f"Reference Image (Average of {num_timepoints} Frames)")
ax1.axis("off")

ax2.imshow(esrrf_result, cmap="gray")
ax2.set_title(f"eSRRF Super-Resolution Image (Average of {frames_per_timepoint} Frames")
ax2.axis("off")

plt.tight_layout()
plt.show()
