# Jake Ankrum
# 07/13/2026
# EMBL - Zimmerman Team

import functools
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from liffile import LifFile as lif
from nanopyx.methods.drift_alignment import DriftEstimator
from nanopyx.methods.squirrel.resolution import calculate_decorr_analysis as decorr
from nanopyx.methods.squirrel.resolution import calculate_frc as frc
from pyopencl import CompilerWarning
from tifffile import imwrite

from custom_methods.custom_eSRRF_workflow import eSRRF
from custom_methods.custom_sweep import ParameterSweep
from custom_methods.custom_sweep_wrapper import run_esrrf_parameter_sweep as psweep
from decay_curve_plots import plot_decay

# * Mute the double precision GPU warning for my laptop
warnings.filterwarnings("ignore", category=CompilerWarning)
warnings.filterwarnings("once", category=UserWarning)

# ? Begin Sample Loading
# * Fils paths of samples
series1 = "./data_raw/labled_cells/fluctuation_analysis/Cryo/br2_cryo_confocal/br2_cryo_confocal.lif"
series2 = "./data_raw/labled_cells/fluctuation_analysis/RT/Br2-Fluorescein-RT-confocal.lif"

# * Name samples (dye, frequency, temperture)
sample1 = "br2_cryo"
freq1 = 8000

sample2 = "br2_RT"
freq2 = 8000

# * If using two series, set equal to True
two_series = True

# * Select the indices for two series (index1 = series1 and index2 = series2)
idx1 = 1
idx2 = 2

# TODO add the frequency in this portion of the name
if two_series:
    name1 = f"{sample1}_{freq1}hz_idx{idx1}"
    name2 = f"{sample2}_{freq2}hz_idx{idx2}"
else:
    name1 = f"{sample1}_{freq1}hz_idx{idx1}"
    name2 = f"{sample1}_{freq2}hz_idx{idx2}"


# ? Begin Parametarizaiton
# * Drift correction
drift_avg = 50  # Number of frames averaged to correct drift

# * Parameter Sweep
do_sweep = True
sens_rg = [1, 2, 3, 4, 5]  # List of sensativities to try
radii_rg = [1, 2, 3, 4, 5]  # List of raddi to try

# * SRRF
mag = 3  # Upsampling factor
temporal_correlation = "VAR"  # Correlation method
it_w = True  # Toggle intensity weighting
decorrelation = True  # Uses Decorrelation for resolution when true; else uses FRC for resolution
man_sens_idx1 = None  # Mannual overside for sensitivity setting for movie 1
man_rad_idx1 = None  # Mannual overide for radii setting fo rmovie 1
man_sens_idx2 = None  # Mannual overside for sensitivity setting for movie 2
man_rad_idx2 = None  # Mannual overide for radii setting fo rmovie 2
sr_idx1 = 0  # If batching, select index of the preferred reconstruction (movie 1)
sr_idx2 = 0  # If batching, select index of the preferred reconstruction (movie 2)

# * Physical Constants
pix_size = 128.75  # Pixel size in nanometers
wave = 500  # Wavelength
max_res = (2 * wave) / pix_size  # Worst possible resolution in pixels (twice the wavelength)
min_res = wave / (2 * pix_size * mag)  # Best possible resolution in pixels

# * Sample batching and slicing
start = 0  # First frame to include in analysis
stop = 250  # Last frame to include in analysis
n_batch = None  # Number of frames for super resolution

# * Decay Curve plots and data
calculate_decay = True
limit1 = 1.75  # Extent of the inset in the decay graph of sample 1
limit2 = 1.75  # Extent of the inset in the decay graph of sample 2

# * Save individual plots with descriptive titles and filenames for backreference
save_results = True

# ? Begin movie extraction
# Extract the appropriate movie
lif_obj1 = lif(series1).images[idx1]
lif_obj2 = lif(series2).images[idx2]

# Convert to array, and slice the proper data range
movie1 = lif_obj1.asarray()[start:stop]

# Handle second movie based on existance of a second input file
if two_series:
    movie2 = lif_obj2.asarray()[start:stop]
else:
    movie2 = lif_obj1.asarray()[start:stop]

# ? Begin cross-correlative XY stabilization
# Apply correction to both data sets
movie1_corr = DriftEstimator(verbose=False).estimate(movie1, apply=True, ref_options=0, time_averaging=drift_avg)

movie2_corr = DriftEstimator(verbose=False).estimate(
    movie2,
    apply=True,
    ref_options=0,
    time_averaging=drift_avg,
)

# ? Extract Decay Curve Data
if calculate_decay:
    decay_fig1, (raw_data1, fit_data1) = plot_decay(lif_obj1, name1, limit1)
    decay_fig2, (raw_data2, fit_data2) = plot_decay(lif_obj2, name2, limit2)

# ? Begin parameter sweep on both data sets
if do_sweep:
    params1, stats1 = psweep(
        movie1_corr,
        mag,
        sens_rg,
        radii_rg,
        temporal_correlation,
        use_decorr=decorrelation,
        plot_sweep=False,
        n_frames=n_batch,
    )

    params2, stats2 = psweep(
        movie2_corr,
        mag,
        sens_rg,
        radii_rg,
        temporal_correlation,
        use_decorr=decorrelation,
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
    if man_sens_idx1 is None or man_rad_idx1 is None
    else f"MANNUAL OVERRIDE: Using senstivity {sensitivity1} and radius {ring_radius1} for movie 1"
)
print(
    f"Using senstivity {sensitivity2} and radius {ring_radius2} for movie 2"
    if man_sens_idx2 is None or man_rad_idx2 is None
    else f"MANNUAL OVERRIDE: Using senstivity {sensitivity2} and radius {ring_radius2} for movie 2"
)

# Zero uses all frames according to eSRRF authors
n_frames = 0 if n_batch is None else n_batch

# Take time average of both movies for reference
movie1_slice = movie1_corr[start:n_batch, :, :] if n_batch is not None else movie1_corr
movie2_slice = movie2_corr[start:n_batch, :, :] if n_batch is not None else movie2_corr

movie1_sum = np.sum(movie1_slice, axis=0)
movie2_sum = np.sum(movie2_slice, axis=0)

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

# Compute scores of averaged movies
if decorrelation:
    # Use Decorrelation Analysis to estiamte resolution
    sum_res1 = decorr(movie1_sum, pixel_size=pix_size, units="nm")
    sum_res2 = decorr(movie2_sum, pixel_size=pix_size, units="nm")

    # Denote resolution measurement for book keeping purposes
    res_type = "Decorr"
else:
    # Create even and odd data sets for FRC comparison
    even1 = np.mean(movie1_slice[::2, :, :], axis=0)
    odd1 = np.mean(movie1_slice[1::2, :, :], axis=0)

    even2 = np.mean(movie2_slice[::2, :, :], axis=0)
    odd2 = np.mean(movie2_slice[1::2, :, :], axis=0)

    # Use Fourier Ring Correlation to estimate resoltion
    sum_res1 = frc(even1, odd1, pixel_size=pix_size, units="nm")
    sum_res2 = frc(even2, odd2, pixel_size=pix_size, units="nm")

    # Denote resolution measurement for book keeping purposes
    res_type = "FRC"


# ? Begin presentation

if do_sweep:
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
    ax2.set_title(f"{name2} Parameter Colormap ({res_type})")
    figure1.tight_layout()

# Name plots
essrf1_plot = f"{name1}: eSRRF of {movie1_slice.shape[0]} frames | UQnR: {uqnr_select1:.3f} | Res: {frc_select1:.2f}nm | RSP: {rsp_select1:.2f}"
essrf2_plot = f"{name2}: eSRRF of {movie2_slice.shape[0]} frames | UQnR: {uqnr_select2:.3f} | Res: {frc_select2:.2f}nm | RSP: {rsp_select2:.2f}"
sum1_plot = f"{name1}: Temporal Sum of {movie1_slice.shape[0]} frames ({start}-{start + movie1_slice.shape[0]}) | Res: {sum_res1:.2f}nm"
sum2_plot = f"{name2}: Temporal Sum of {movie2_slice.shape[0]} frames ({start}-{start + movie2_slice.shape[0]})| Res: {sum_res2:.2f}nm"

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

ax[2].imshow(movie1_sum, cmap="gist_heat")
ax[2].axis("off")
ax[2].set_title(sum1_plot)

ax[3].imshow(movie2_sum, cmap="gist_heat")
ax[3].axis("off")
ax[3].set_title(sum2_plot)
figure2.tight_layout()

# Name output files
essrf1_output = f"{name1}_{ring_radius1}radii_{sensitivity1}sens_{mag}mag_{temporal_correlation}_sridx{sr_idx1}"
essrf2_output = f"{name2}_{ring_radius2}radii_{sensitivity2}sens_{mag}mag_{temporal_correlation}_sridx{sr_idx2}"
sum1_output = f"{name1}_t_sum"
sum2_output = f"{name2}_t_sum"

save_dir = "./data_analyzed/fluctuation_comparison"
save_decay_dir = "./data_analyzed/decay_comparison"
folder = f"{name1}_vs_{name2}"
subfolder = f"{temporal_correlation}correlation_{res_type}resolution"


# Define saving fucntion
def save_plot(title, img_data, output):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.log1p(img_data), cmap="gist_heat")
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(f"{save_dir}/{folder}/{subfolder}/{output}.png", dpi=300, bbox_inches="tight")
    plt.close()


# Define a decorator function to handle exceptions when saving decay data
def save_or_skip(func):
    @functools.wraps(func)
    def save_or_skip_wrapper(raw_data_frame, fitted_data_frame, save_path, sample, name):
        try:
            func(raw_data_frame, fitted_data_frame, save_path, sample, name)
        except ValueError as e:
            if "already exists" in str(e):
                warnings.warn(f"Cannot save {name} decay data. Sheet already exists!")
            else:
                raise

    return save_or_skip_wrapper


# Define decay file organizer
@save_or_skip
def data_as_excel(
    raw_data_frame: pd.DataFrame, fitted_data_frame: pd.DataFrame, save_path: str, sample: str, name: str
) -> None:

    # Define the common path
    cpath = f"{save_path}/{sample}"

    # Create an excel file if one does not exists
    if not any(Path(cpath).glob("*.xlsx")):
        raw_data_frame.to_excel(f"{cpath}/{sample}_rawdata.xlsx", sheet_name=name)
        fitted_data_frame.to_excel(f"{cpath}/{sample}_fitdata.xlsx", sheet_name=name)

    else:
        # Append Raw Data
        with pd.ExcelWriter(f"{cpath}/{sample}_rawdata.xlsx", mode="a") as writer1:
            raw_data_frame.to_excel(writer1, sheet_name=name)

        # Append Fitted Data
        with pd.ExcelWriter(f"{cpath}/{sample}_fitdata.xlsx", mode="a") as writer2:
            fitted_data_frame.to_excel(writer2, sheet_name=name)


# ? Save Data
if save_results:
    # Make a folder to save all images
    os.makedirs(f"{save_dir}/{folder}/{subfolder}", exist_ok=True)
    os.makedirs(f"{save_dir}/{folder}/original_data", exist_ok=True)
    os.makedirs(f"{save_decay_dir}/{sample1}", exist_ok=True)
    os.makedirs(f"{save_decay_dir}/{sample2}", exist_ok=True)

    # Create data dictionary
    data = {
        "names": [essrf1_plot, essrf2_plot, sum1_plot, sum2_plot],
        "input": [np.log1p(essrf1[sr_idx1]), np.log1p(essrf2[sr_idx1]), movie1_sum, movie2_sum],
        "output": [essrf1_output, essrf2_output, sum1_output, sum2_output],
        "series": [essrf1, essrf2, movie1_sum, movie2_sum],
    }

    # Iterate over all data
    for i in range(len(data["names"])):
        # Create a figure for each plot
        save_plot(
            data["names"][i],
            data["input"][i],
            data["output"][i],
        )

        # Save .tif files for each dataset for further manipulaiton
        imwrite(f"{save_dir}/{folder}/{subfolder}/{data['output'][i]}.tif", data["series"][i])

        # Save .tif files for the original data as a unique sub folder (only if subfolder is empty)
        empty = True
        for _ in os.scandir(f"{save_dir}/{folder}/original_data"):
            empty = False
            break

        if empty:
            imwrite(f"{save_dir}/{folder}/original_data/{name1}_original.tif", movie1)
            imwrite(f"{save_dir}/{folder}/original_data/{name2}_original.tif", movie2)

    # Save Parameter Sweep
    figure1.savefig(f"{save_dir}/{folder}/{subfolder}/parameter_sweep_{res_type}.png", dpi=300, bbox_inches="tight")

    # Save Comparitive Plot
    figure2.savefig(f"{save_dir}/{folder}/{subfolder}/comparative_chart_{res_type}.png", dpi=300, bbox_inches="tight")

    # Save Decay Plots
    decay_fig1.savefig(f"{save_decay_dir}/{sample1}/{sample1}_{freq1}.png", dpi=300, bbox_inches="tight")
    decay_fig2.savefig(f"{save_decay_dir}/{sample2}/{sample2}_{freq2}.png", dpi=300, bbox_inches="tight")

    # Save data for decay plots
    data_as_excel(raw_data1, fit_data1, save_decay_dir, sample1, name1)
    data_as_excel(raw_data2, fit_data2, save_decay_dir, sample2, name2)

    print("Data saved successfully!")
