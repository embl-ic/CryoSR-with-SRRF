import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from liffile import LifFile as lif
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load image
image = lif("./data_raw/labled_cells/fluctuation_analysis/Cryo/br2_cryo_confocal/br2_cryo_confocal.lif").images[2]


def plot_decay(im: NDArray, output_path: str, name: str, inset: bool = True) -> None:

    # Extract metadata from the .lif file
    attributes = im.attrs["HardwareSetting"]["ATLConfocalSettingDefinition"]

    # Extract data from .lif file
    time = np.arange(0, attributes["CycleCount"]) * attributes["FrameTime"]
    intensity = np.mean(im.asarray(), axis=(1, 2))

    # Determine length of cropped region
    cutoff = 1
    limit = np.searchsorted(time, cutoff, side="right")

    # Crop region of interest
    crop_time = time[0:limit]
    crop_intensity = intensity[0:limit]

    try:
        # ? Use a linear regression of log(f(x)) to provide intial guess of parameters

        # Transform data with a natural log
        baseline = np.median(intensity)  # Subtract the constant
        mask = crop_intensity - baseline > 0  # Only select values > 0 wiht a boolean mask
        y_log = crop_intensity[mask]  # Apply mask to the y axis
        x_log = crop_time[mask]  # Apply mask to x axis
        y_est = np.log(y_log - baseline)  # transform logirthmically

        # Fit the linear regression
        (m, b), _ = curve_fit(lambda x, m, b: m * x + b, x_log, y_est)

        # Use logrithmic regression to generate an intial guesses
        p0 = [np.exp(b), np.exp(b) / 2, m, 2 * m, np.median(intensity)]

        # Fit the double exponential
        (a_fit, a2_fit, k_fit, k2_fit, c_fit), _, info, _, _ = curve_fit(
            lambda x, a, a2, k, k2, c: a * np.exp(k * x) + a2 * np.exp(k2 * x) + c,
            crop_time,
            crop_intensity,
            p0,
            full_output=True,
        )

        # Fit the single exponential
        (A_fit, K_fit, C_fit), _, INFO, _, _ = curve_fit(
            lambda X, A, K, C: A * np.exp(K * X) + C, crop_time, crop_intensity, [p0[0], p0[2], p0[4]], full_output=True
        )

        # Create the fitted dataset
        x_fit = np.linspace(0, np.max(crop_time), 1000)
        y_dbl = a_fit * np.exp(k_fit * x_fit) + a2_fit * np.exp(k2_fit * x_fit) + c_fit
        y_sgl = A_fit * np.exp(K_fit * x_fit) + C_fit

        # Plot the data
        figure1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

        ax1.scatter(crop_time, crop_intensity, label="Raw Data", color="gray")
        ax1.plot(
            x_fit,
            y_dbl,
            label=f"{a_fit:.2f}exp({k_fit:.2f}x) + {a2_fit:.2f}exp({k2_fit:.2f}x) + {c_fit:.2f}",
            color="black",
            alpha=0.75,
        )

        # Find largest value for limit purposes
        lim = max(np.abs(info["fvec"]).max(), np.abs(INFO["fvec"]).max())

        ax1.plot(x_fit, y_sgl, label=f"{A_fit:.2f}exp({K_fit:.2f}x) + {C_fit:.2f}", color="blue", alpha=0.75)
        ax1.set_title(f"{name}: Z-Axis Profile")
        ax1.set_xlabel("Time (Seconds)")
        ax1.set_ylabel("Intensity (Counts)")
        ax1.legend()
        ax1.grid("On")

        ax2.plot(crop_time, np.zeros_like(crop_time), color="black")
        ax2.scatter(crop_time, info["fvec"], label="Double Exponential Residuals", color="red")
        ax2.scatter(crop_time, INFO["fvec"], label="Single Exponential Residuals", color="blue")
        ax2.set_title(f"{name}: Residual Plot")
        ax2.set_xlabel("Time (Seonds)")
        ax2.set_ylabel("Error (Counts)")
        ax2.set_ylim([-1.1 * lim, 1.1 * lim])
        ax2.legend()
        ax2.grid("On")
        figure1.tight_layout()

        print("FLuorescence decay sucessfully fitted to an exponential model!")

    except RuntimeError:
        # Fit to a linear model
        (M, B), _, infodict, _, _ = curve_fit(
            lambda X0, M, B: M * X0 + B, time, intensity, [-1, p0[0]], full_output=True
        )

        # Create a fitted dataset
        x_fit = np.linspace(0, max(time), 1000)
        y_fit = M * x_fit + B

        # Find the r2 score
        y_r2 = M * time + B
        r2 = r2_score(intensity, y_r2)

        # Pack variables
        y_prd = [FittedCurve(y_fit, infodict["fvec"], f"$y = {M:.2f}x+{B:.2f}$", "orange")]
        info = Data(time, intensity, x_fit, y_prd)
        setup = Config("Time (Seconds)", "Intensity (Counts)", [0.4, 0.1, 0.55, 0.55], 1, name, r2)

        # Call plotting function
        inset_fitted_plot(info, setup)


@dataclass
class Data:
    x_vals: NDArray
    y_vals: NDArray
    x_fitted: NDArray
    y_fitted: list


@dataclass
class Config:
    xlabel: str
    ylabel: str
    extent: list[float]
    cutoff: int
    name: str
    r2_score: float | None = None
    size: tuple = (8, 12)
    legend_loc: str = "upper right"


@dataclass
class FittedCurve:
    vals: NDArray
    residuals: NDArray
    equation: str
    color: str


def inset_fitted_plot(data: Data, config: Config, save_path: str | None = None, **kwargs) -> None:
    """Plot fitted decay data with an inset zoom and residual panel."""

    # Initialize figure and axes
    figure1, (ax1, ax2) = plt.subplots(2, 1, figsize=config.size)

    # Plot orginal data
    ax1.plot(data.x_vals, data.y_vals, label="Raw Data", color="gray")

    # Set up and plot inset
    ax1_inset = ax1.inset_axes(config.extent, xlim=[0, config.cutoff])
    ax1_inset.scatter(data.x_vals, data.y_vals, color="gray")
    ax1_inset.grid("on")

    # Plot residual graph
    ax2.plot(data.x_vals, np.zeros_like(data.x_vals), color="black")

    # Add the fitted data to the main, inset, and residual plots
    maxes = []
    for curve in data.y_fitted:
        # Plot all fitted curves in the main graph
        ax1.plot(data.x_fitted, curve.vals, label=curve.equation, color=curve.color)
        # Plot all fitted curves in the inset
        ax1_inset.plot(data.x_fitted, curve.vals, label=curve.equation, color=curve.color)
        # Plot the residuals of all fitted curves
        ax2.scatter(data.x_vals, curve.residuals, label=curve.equation, color=curve.color)
        # Store the absolute max of each residual array
        maxes.append(max(np.abs(curve.residuals)))

    # Calculate max absolute value for residual plot
    lim = max(maxes)

    # Annotate main graph
    ax1.indicate_inset_zoom(ax1_inset, edgecolor="black", alpha=1)
    ax1.set_title(f"{config.name} | $R^2$: {config.r2_score:.2f}" if config.r2_score is not None else f"{config.name}")
    ax1.set_xlabel(config.xlabel)
    ax1.set_ylabel(config.ylabel)
    ax1.set_ylim(0, max(data.y_vals) * 1.2)
    ax1.set_xlim(0, max(data.x_vals))
    ax1.legend(loc=config.legend_loc)

    # Annotate the residual plot
    ax2.set_title(f"{config.name}: Residual Plot")
    ax2.set_xlabel(config.xlabel)
    ax2.set_ylabel("Errors (Counts)")
    ax2.set_ylim(-1.1 * lim, 1.1 * lim)
    ax2.set_xlim(0, config.cutoff)
    ax2.legend()
    ax2.grid("on")

    figure1.tight_layout()

    # TODO consider a separate heirarchial_saving function
    # Save results
    if save_path:
        # Make specified output directory
        os.makedirs(save_path, exist_ok=True)

        # Save decay curve as a png
        # TODO label each sample with a frequency (likely via kwargs or optional args)
        if not os.path.exists(f"{save_path}/{config.name}/Decay_Curve.png"):
            plt.savefig(f"{save_path}/{config.name}/Decay_Curve.png", dpi=300, bbox_inches="tight")

        # Save the raw data as a csv
        if not os.path.exists(f"{save_path}/{config.name}_Decay_Curve_RawData.csv"):
            save_as = {f"{config.xlabel}": data.x_vals, f"{config.ylabel}": data.y_vals}
            df = pd.DataFrame(save_as)
            df.to_csv(f"{save_path}/{config.name}_Decay_Curve.csv")
        else:
            # TODO open data frame, append, and save modified data frame
            pass


plot_decay(image, None, "FITC_1000Hz_Cryo")
