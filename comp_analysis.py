# Jake Ankrum
# 07/13/2026
# EMBL - Zimmerman Team

from readlif.reader import LifFile as lif
from nanopyx.methods.drift_alignment import DriftEstimator
from nanopyx.core.analysis.parameter_sweep import ParameterSweep
from nanopyx.methods.esrrf.eSRRF_workflow import eSRRF
from custom_sweep import run_esrrf_parameter_sweep as psweep
from tifffile import imwrite
import warnings
from pyopencl import CompilerWarning
import os
import matplotlib.pyplot as plt
import numpy as np

# * Mute the double precision GPU warning for my laptop
warnings.filterwarnings("ignore", category=CompilerWarning)

# ? Begin Sample Loading
# * Fils paths of samples
series1 = "./data_raw/labled_cells/cell_culture_demo/JF571/sample_28_05_2026/SUM159_LifeAct_JF571_Grid1_oversampled_stacks.lif"
series2 = (
    "./data_raw/labled_cells/cell_culture_demo/Br4/Br4_Actin_SUM159_SiO2_10_6_2026.lif"
)

# * Name samples (dye, frequency, temperture)
name1 = "series1"
name2 = "series2"

# Chose save location
save_dir = "./data_analyzed/fluctuation_comparison"
folder = f"{name1}_vs_{name2}"

# * If using two series, set equal to True
two_series = False

# * Select the indices for two series (index1 = series1 and index2 = series2)
idx1 = 0
idx2 = 4

# ? Begin Parametarizaiton
# Drift correction
drift_avg = 5  # Avg frames used to

# Parameter Sweep
sens_rg = [1, 2, 3, 4]  # List of sensativities to try
radii_rg = [3, 4, 5, 6, 7]  # List of raddi to try

# SRRF
mag = 2  # Upsampling factor
temporal_correlation = "AVG"  # C orrelation method
it_w = True  # Toggle intensity weighting
man_sens_idx1 = None  # Mannual overside for sensitivity setting for movie 1
man_rad_idx1 = None  # Mannual overide for radii setting fo rmovie 1
man_sens_idx2 = None  # Mannual overside for sensitivity setting for movie 2
man_rad_idx2 = None  # Mannual overide for radii setting fo rmovie 2
sr_idx1 = 0  # If batching, select index of the preferred reconstruction (movie 1)
sr_idx2 = 0  # If batching, select index of the preferred reconstruction (movie 2)

# Physical Constants
pix_size = 108.5  # Pixel size in nanometers
wave = 500  # Wavelength
max_res = (
    2 * wave
) / pix_size  # Worst possible resolution in pixels (twice the wavelength)
min_res = wave / (2 * pix_size * mag)  # Best possible resolution in pixels

# Sample batching and slicing
start = 0  # First frame to include in analysis
stop = 50  # Last frame to include in analysis
n_batch = None  # Number of frames for super resolution

# Save individual plots with descriptive titles and filenames for backreference
save_png = False

# ? Begin movie extraction
# Extract the appropriate movie, convert to array, and slice the proper data range
movie1 = lif(series1).get_image(idx1).as_array(dims=[4])[start:stop]

# Handle second movie based on existance of a second input file
if two_series:
    movie2 = lif(series2).get_image(idx2).as_array(dims=[4])[start:stop]
else:
    movie2 = lif(series1).get_image(idx2).as_array(dims=[4])[start:stop]

# ? Begin cross-correlative XY stabilization
# Apply correction to both data sets
movie1_corr = DriftEstimator(verbose=False).estimate(
    movie1, apply=True, ref_options=5, time_averaging=drift_avg
)

movie2_corr = DriftEstimator(verbose=False).estimate(
    movie2,
    apply=True,
    ref_options=5,
    time_averaging=drift_avg,
)

# ? Begin parameter sweep on both data sets
params1, stats1 = psweep(
    movie1_corr,
    mag,
    sens_rg,
    radii_rg,
    temporal_correlation,
    plot_sweep=False,
    n_frames=n_batch,
)


params2, stats2 = psweep(
    movie2_corr,
    mag,
    sens_rg,
    radii_rg,
    temporal_correlation,
    plot_sweep=False,
    n_frames=n_batch,
)
# ? Begin preformance of analysis
# Vertically stack frc1 and frc2 arrays for easy splitting after normalization
frc1 = stats1[2]
frc2 = stats2[2]

rsp1 = stats1[1]
rsp2 = stats2[1]

# Calculate qnr scores
uqnr1 = ParameterSweep().calculate_qnr_score(rsp1, frc1, min_res, max_res)
uqnr2 = ParameterSweep().calculate_qnr_score(rsp2, frc2, min_res, max_res)

# Determine senativity and ring_radius indices with option for mannual override
sens_idx1 = params1[0] if man_sens_idx1 is None else man_sens_idx1
rad_idx1 = params1[1] if man_rad_idx1 is None else man_rad_idx1

sens_idx2 = params2[0] if man_sens_idx2 is None else man_sens_idx2
rad_idx2 = params2[1] if man_rad_idx2 is None else man_rad_idx2

# Extract sensativity and ring_radius parameters
sensitivity1 = sens_rg[sens_idx1]
ring_radius1 = radii_rg[rad_idx1]

sensitivity2 = sens_rg[sens_idx2]
ring_radius2 = radii_rg[rad_idx2]

print(
    f"Using senstivity {sensitivity1} and radius {ring_radius1} for movie 1"
    if man_sens_idx1 or man_rad_idx1 is None
    else f"MANNUAL OVERRIDE: Using senstivity {sensitivity1} and radius {ring_radius1} for movie 1"
)
print(
    f"Using senstivity {sensitivity2} and radius {ring_radius2} for movie 2"
    if man_sens_idx2 or man_rad_idx2 is None
    else f"MANNUAL OVERRIDE: Using senstivity {sensitivity2} and radius {ring_radius2} for movie 2"
)

# Zero uses all frames according to eSRRF authors
n_frames = 0 if n_batch is None else n_batch

# Take time average of both movies for reference
movie1_slice = movie1_corr[start:n_batch, :, :] if n_batch is not None else movie1_corr
movie2_slice = movie2_corr[start:n_batch, :, :] if n_batch is not None else movie2_corr

movie1_avg = np.mean(movie1_slice, axis=0)
movie2_avg = np.mean(movie2_slice, axis=0)

# ? Being eSRRF Processing
essrf1 = eSRRF(
    movie1_corr,
    mag,
    ring_radius1,
    sensitivity1,
    n_frames,
    temporal_correlation,
    doIntensityWeighting=it_w,
)

essrf2 = eSRRF(
    movie2_corr,
    mag,
    ring_radius2,
    sensitivity2,
    n_frames,
    temporal_correlation,
    doIntensityWeighting=it_w,
)

# Determine the scores based on selected parameters
uqnr_select1 = uqnr1[sens_idx1, rad_idx1]
uqnr_select2 = uqnr2[sens_idx2, rad_idx2]

rsp_select1 = rsp1[sens_idx1, rad_idx1]
rsp_select2 = rsp2[sens_idx2, rad_idx2]

frc_select1 = pix_size * frc1[sens_idx1, rad_idx1]
frc_select2 = pix_size * frc2[sens_idx2, rad_idx2]

# ? Begin presentation
# Plot qnr plots
figure1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# qnr1 cmap (left)
ax1.imshow(uqnr1, cmap="cool")
ax1.set_xticks(np.arange(len(radii_rg)), labels=radii_rg)
ax1.set_yticks(np.arange(len(sens_rg)), labels=sens_rg)
for i in range(len(sens_rg)):
    for j in range(len(radii_rg)):
        ax1.text(j, i, round(uqnr1[i, j], 2), ha="center", va="center", color="black")
ax1.set_xlabel("Radii")
ax1.set_ylabel("Sensitivities")
ax1.set_title(f"{name1} Parameter Colormap")
figure1.tight_layout()

# qnr2 cmap (right)
ax2.imshow(uqnr2, cmap="cool")
ax2.set_xticks(np.arange(len(radii_rg)), labels=radii_rg)
ax2.set_yticks(np.arange(len(sens_rg)), labels=sens_rg)
for i in range(len(sens_rg)):
    for j in range(len(radii_rg)):
        ax2.text(j, i, round(uqnr2[i, j], 2), ha="center", va="center", color="black")
ax2.set_xlabel("Radii")
ax2.set_ylabel("Sensitivities")
ax2.set_title(f"{name2} Parameter Colormap")
figure1.tight_layout()

# Name plots
essrf1_plot = f"{name1}: eSRRF of {movie1_slice.shape[0]} frames | UQnR: {uqnr_select1:.3f} | FRC: {frc_select1:.2f}nm | RSP: {rsp_select1:.2f}"
essrf2_plot = f"{name2}: eSRRF of {movie2_slice.shape[0]} frames | UQnR: {uqnr_select2:.3f} | FRC: {frc_select2:.2f}nm | RSP: {rsp_select2:.2f}"
avg1_plot = f"{name1}: Temporal Average of {movie1_slice.shape[0]} frames from frame {start} to frame {start + movie1_slice.shape[0]}"
avg2_plot = f"{name2}: Temporal Average of {movie2_slice.shape[0]} frames from frame {start} to frame {start + movie2_slice.shape[0]}"

# Insert axis if necessary to handle batching
essrf1 = np.expand_dims(essrf1, axis=0) if essrf1.ndim == 2 else essrf1
essrf2 = np.expand_dims(essrf2, axis=0) if essrf2.ndim == 2 else essrf2

# Raise Value error for out of bounds indexing
if essrf1.shape[0] <= sr_idx1 or essrf2.shape[0] <= sr_idx2:
    raise ValueError(
        "The selected reconstruction index does not exist"
        if essrf1.shape[0] != 1
        else "Cannot select a reconstruction index higher than 0 without batching"
    )

print(f"Plotting reconstruction index {sr_idx1} for movie 1 and {sr_idx2} for movie 2")

# Plot eSRRF and averages
figure2, ax = plt.subplots(2, 2, figsize=(16, 16))
ax = np.ravel(ax)

ax[0].imshow(np.log1p(essrf1[sr_idx1]), cmap="gist_heat")
ax[0].axis("off")
ax[0].set_title(essrf1_plot)


ax[1].imshow(np.log1p(essrf2[sr_idx2]), cmap="gist_heat")
ax[1].axis("off")
ax[1].set_title(essrf2_plot)

ax[2].imshow(movie1_avg, cmap="gist_heat")
ax[2].axis("off")
ax[2].set_title(avg1_plot)

ax[3].imshow(movie2_avg, cmap="gist_heat")
ax[3].axis("off")
ax[3].set_title(avg2_plot)
plt.tight_layout()

# Name output files
essrf1_output = f"{name1}_{ring_radius1}radii_{sensitivity1}sens_{mag}mag_{temporal_correlation}.png"
essrf2_output = f"{name2}_{ring_radius2}radii_{sensitivity2}sens_{mag}mag_{temporal_correlation}.png"
avg1_output = f"{name1}_t_avg.png"
avg2_output = f"{name2}_t_avg.png"


# Define saving fucntion
def save_plot(title, img_data, output):
    plt.figure(figsize=(10, 10))
    plt.imshow(img_data, cmap="gray")
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(f"{save_dir}/{folder}/{output}", dpi=300, bbox_inches="tight")
    plt.close()


# ? Save Data
if save_png:
    # Make a folder to save all images
    os.makedirs(f"{save_dir}/{folder}", exist_ok=True)

    # Create data dictionary
    data = {
        "names": [essrf1_plot, essrf2_plot, avg1_plot, avg2_plot],
        "input": [essrf1[sr_idx1], essrf2[sr_idx2], movie1_avg, movie2_avg],
        "output": [essrf1_output, essrf2_output, avg1_output, avg2_output],
    }

    for i in range(len(data["names"])):
        # Create a figure for each plot
        save_plot(
            data["names"][i],
            data["input"][i],
            data["output"][i],
        )

        # Save .tif files for each dataset for further manipulaiton
        imwrite(f"{data['output'].strip('.png')}.tif", data["input"])
