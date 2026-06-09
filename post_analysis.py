# Jake Ankrum
# EMBL IC
# 5/22/2026

from nanopyx.methods.squirrel.resolution import calculate_decorr_analysis
from nanopyx.methods.squirrel.resolution import calculate_frc
from nanopyx.methods.squirrel.error_map import calculate_error_map
from matplotlib import pyplot as plt
import numpy as np


def analyze(
    ref_img: np.ndarray,
    srrf_img: np.ndarray,
    # Shared kwargs
    pixel_size: float = 1,
    units: str = "pixel",
    # Kargs specific to FRC
    frame_1: int = 0,
    frame_2: int = 1,
    plot_frc_curve: bool = False,
    # Kargs specific to decorrelation analysis
    frame_analyze: int = 0,  # For decorrelation and plotting of error map
    radius_min: float = 0,  # Minimal radius for boolean mask
    radius_max: float = 1,  # Maximum radius for boolean mask
    num_rads: int = 50,  # Number of radial divisions
    num_ang: int = 10,  # Number of angular divisions
    roi: tuple = (0, 0, 0, 0),  # xmin, y_min, x_max, y_max
    plot_decorr_analysis: bool = False,
    # Extra Kwargs
    plot_error: bool = False,
):

    # Calculate the FRC curve and resolution
    frc = calculate_frc(
        frame_1=srrf_img[frame_1],
        frame_2=srrf_img[frame_2],
        pixel_size=pixel_size,
        units=units,
        plot_frc_curve=plot_frc_curve,
    )

    # Calculate the decorrelation analysis
    decor = calculate_decorr_analysis(
        frame=srrf_img[frame_analyze],
        rmin=radius_min,
        rmax=radius_max,
        n_r=num_rads,
        n_g=num_ang,
        pixel_size=pixel_size,
        units=units,
        roi=roi,
        plot_decorr_analysis=plot_decorr_analysis,
    )

    # Calculate the error map
    error_map, RSE, RSP = calculate_error_map(ref_img, srrf_img[frame_analyze])

    if plot_error:
        im = plt.imshow(error_map, cmap="viridis")
        plt.colorbar(im)
        plt.title(f"Error Map (RSE: {RSE:.2f}, RSP: {RSP:.2f})")
        plt.axis("off")
        plt.show()

    return frc, decor
