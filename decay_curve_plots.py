from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from liffile import LifFile as lif
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load test image
image = lif("./data_raw/labled_cells/fluctuation_analysis/Cryo/br2_cryo_confocal/br2_cryo_confocal.lif").images[2]


def plot_decay(im: NDArray, name: str, boundry: int) -> tuple[plt.figure, pd.DataFrame]:

    # Extract metadata from the .lif file
    attributes = im.attrs["HardwareSetting"]["ATLConfocalSettingDefinition"]

    # Extract data from .lif file
    time = np.arange(0, attributes["CycleCount"]) * attributes["FrameTime"]
    intensity = np.mean(im.asarray(), axis=(1, 2))

    # Determine length of cropped region
    cutoff = boundry
    limit = np.searchsorted(time, cutoff, side="right")

    # Crop region of interest
    crop_time = time[0:limit]
    crop_intensity = intensity[0:limit]

    try:
        # ? Use a linear regression of log(f(x)) to provide intial guess of parameters

        # Transform data with a natural log
        baseline = np.min(crop_intensity)  # Subtract the constant
        mask = crop_intensity - baseline > 0  # Only select values > 0 with a boolean mask
        y_log = crop_intensity[mask]  # Apply mask to the y axis
        x_log = crop_time[mask]  # Apply mask to x axis
        y_est = np.log(y_log - baseline)  # transform logirthmically

        # Fit the linear regression
        (m, b), _ = curve_fit(lambda x, m, b: m * x + b, x_log, y_est)

        # Use logrithmic regression to generate an intial guesses
        p0 = [np.exp(b), np.exp(b) / 2, m, 2 * m, np.median(intensity)]

        # ? Attempt to fit an exponential function to cryo data
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
        y_sgl = A_fit * np.exp(K_fit * x_fit) + C_fit
        y_dbl = a_fit * np.exp(k_fit * x_fit) + a2_fit * np.exp(k2_fit * x_fit) + c_fit

        # Find r2_scores
        y_sgl_r2 = A_fit * np.exp(K_fit * crop_time) + C_fit
        y_dbl_r2 = a_fit * np.exp(k_fit * crop_time) + a2_fit * np.exp(k2_fit * crop_time) + c_fit
        r2_scores = [r2_score(crop_intensity, y_sgl_r2), r2_score(crop_intensity, y_dbl_r2)]

        # Pack variables
        # fmt: off
        y_prd = [
            FittedCurve(y_sgl, INFO["fvec"], f"{A_fit:.2f}e^({K_fit:.2f}x) + {C_fit:.2f}", "blue"),
            FittedCurve(y_dbl, info["fvec"], f"{a_fit:.2f}e^({k_fit:.2f}x) + {a2_fit:.2f}e^({k2_fit:.2f}x) + {c_fit:.2f}", "red")
        ]
        info = Data(time, intensity,x_fit, y_prd)
        setup = Config("Time (Seconds)", "Intensity (Counts)", [0.4, 0.1, 0.55, 0.55], cutoff, name, r2_scores)
        # fmt: on

        # Call plotting function
        fig = inset_fitted_plot(info, setup)

        print("Fluorescence decay sucessfully fitted to an exponential model!")

    # ? Fall back to a linear lit for room temperature data
    except RuntimeError:
        print("Exponential fit failed, falling back to a linear model...")

        # Fit to a linear model
        (M, B), _, infodict, _, _ = curve_fit(
            lambda X0, M, B: M * X0 + B, time, intensity, [-1, p0[0]], full_output=True
        )

        # Create a fitted dataset
        x_fit = np.linspace(0, max(time), 1000)
        y_fit = M * x_fit + B

        # Find the r2 score
        y_r2 = M * time + B
        r2_scores = [r2_score(intensity, y_r2)]

        # Pack variables
        y_prd = [FittedCurve(y_fit, infodict["fvec"], f"$y = {M:.2f}x+{B:.2f}$", "orange")]
        info = Data(time, intensity, x_fit, y_prd)
        setup = Config("Time (Seconds)", "Intensity (Counts)", [0.4, 0.1, 0.55, 0.55], cutoff, name, r2_scores)

        # Call plotting function
        fig = inset_fitted_plot(info, setup)

        print("Sucessfully fit to a linear model!")

    # Create dataframe for raw values
    data_frame_raw = pd.DataFrame({f"{setup.xlabel}": info.x_vals, f"{setup.ylabel}": info.y_vals})

    # Create a dataframe for fitted values
    data_frame_fitted = pd.DataFrame({f"{setup.xlabel}": info.x_fitted})

    # Determine exponetial or linearity
    moniker = ("Single Exp", "Double Exp") if len(y_prd) > 1 else "Linear"

    # Add a collum for each fit of the dependent variable
    for i, curve in enumerate(y_prd):
        data_frame_fitted[f"{setup.ylabel} {moniker[i]}"] = curve.vals

    # Pack as a tuple
    data = (data_frame_raw, data_frame_fitted)

    return fig, data


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
    extent: list[float]  # x0, y0, width, height
    cutoff: int
    name: str
    r2_score: list | None = None
    size: tuple = (8, 12)
    legend_loc: str = "upper right"
    x_r2: int = 0
    y_r2: int = 0
    font_r2: int = 14


@dataclass
class FittedCurve:
    vals: NDArray
    residuals: NDArray
    equation: str
    color: str


def inset_fitted_plot(data: Data, config: Config, save_path: str | None = None, **kwargs) -> plt.Figure:
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
    for i, curve in enumerate(data.y_fitted):
        # Plot all fitted curves in the main graph
        ax1.plot(data.x_fitted, curve.vals, label=curve.equation, color=curve.color)
        # Plot all fitted curves in the inset
        ax1_inset.plot(data.x_fitted, curve.vals, label=curve.equation, color=curve.color)
        # Plot the residuals of all fitted curves
        ax2.scatter(data.x_vals[0 : len(curve.residuals)], curve.residuals, label=curve.equation, color=curve.color)
        # Annotate r2_scores (if selected)
        if config.r2_score is not None:
            ax1_inset.text(
                0.8 + config.x_r2,
                0.6 - (i * 0.065) + config.y_r2,
                f"$R^2$: {config.r2_score[i]:.3f}",
                color=curve.color,
                transform=ax1.transAxes,
                fontsize=config.font_r2,
            )
        # Store the absolute max of each residual array
        maxes.append(max(np.abs(curve.residuals)))

    # Calculate max absolute value for residual plot
    lim = max(maxes)

    # Annotate main graph
    ax1.indicate_inset_zoom(ax1_inset, edgecolor="black", alpha=1)
    ax1.set_title(f"{config.name}")
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

    # Formatting
    figure1.tight_layout()

    return figure1
