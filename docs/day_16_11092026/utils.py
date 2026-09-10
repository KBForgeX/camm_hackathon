"""Utilities for ATHENA CAMM Hackathon 01.

Keep the notebook focused on the two student functions:
    detect_boundary(image)
    find_atoms(image)

File handling, plotting, ground-truth evaluation, robustness perturbations,
and timing live here.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import pyTEMlib.file_tools as ft

from scipy.spatial import cKDTree
# -----------------------------------------------------------------------------
# Data handling
# -----------------------------------------------------------------------------

def resolve_data_dir() -> Path:
    candidates = [
        Path("data"),
        Path("docs/day_16_11092026/data"),
        Path.cwd() / "data",
        Path.cwd().parent / "data",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the data directory.\n"
        f"Current working directory: {Path.cwd()}\n"
        "Set DATA_DIR manually if needed."
    )


def natural_sort_key(path: Path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.name)
    ]


def list_emd_files(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".emd"],
        key=natural_sort_key,
    )


def robust_normalize(image, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo:
        return np.zeros_like(image, dtype=float)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def load_haadf(file: Path) -> np.ndarray:
    """Load HAADF if available, otherwise Channel_000 or the first 2D dataset."""
    datasets = ft.open_file(str(file))

    if not isinstance(datasets, dict):
        arr = np.squeeze(np.asarray(datasets))
        if arr.ndim != 2:
            raise ValueError(f"No 2D image found in {Path(file).name}")
        return arr

    for key, dset in datasets.items():
        title = str(getattr(dset, "title", ""))
        name = str(getattr(dset, "name", ""))
        if "HAADF" in f"{key} {title} {name}".upper():
            arr = np.squeeze(np.asarray(dset))
            if arr.ndim == 2:
                return arr

    if "Channel_000" in datasets:
        arr = np.squeeze(np.asarray(datasets["Channel_000"]))
        if arr.ndim == 2:
            return arr

    for dset in datasets.values():
        try:
            arr = np.squeeze(np.asarray(dset))
            if arr.ndim == 2:
                return arr
        except Exception:
            pass

    raise ValueError(f"No 2D dataset found in {Path(file).name}")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def show_image_grid(images, titles=None, rows=2, cmap="gray", figsize_per_image=(3.0, 3.0)):
    n = len(images)
    if n == 0:
        print("No images to display.")
        return

    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_image[0] * cols, figsize_per_image[1] * rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, image in enumerate(images):
        image = np.asarray(image)
        if image.dtype == bool:
            axes[i].imshow(image, cmap=cmap, vmin=0, vmax=1)
        else:
            axes[i].imshow(
                image,
                cmap=cmap,
                vmin=np.percentile(image, 1),
                vmax=np.percentile(image, 99.5),
            )
        if titles is not None:
            axes[i].set_title(str(titles[i]), fontsize=10)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.show()


def plot_ground_truth_grid(
    images,
    files,
    titles,
    gt_dir: Path,
    rows=2,
    line_width=1.0,
    uncertainty=0.03,
):
    n = len(images)

    if n == 0:
        print("No images to display.")
        return

    cols = int(np.ceil(n / rows))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.5 * cols, 3.5 * rows),
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()
    gt_dir = Path(gt_dir)

    for i, (image, file, title) in enumerate(zip(images, files, titles)):
        ax = axes[i]
        image = np.asarray(image)

        ax.imshow(
            image,
            cmap="gray",
            vmin=np.percentile(image, 1),
            vmax=np.percentile(image, 99.5),
        )

        gt_file = gt_dir / f"{Path(file).stem}_ground_truth.npz"

        if gt_file.exists():
            gt = np.load(gt_file, allow_pickle=True)
            points_xy = np.asarray(gt["points_xy"])

            x = points_xy[:, 0]
            y = points_xy[:, 1]

            # ---------------------------------------------------------
            # 10% uncertainty shadow
            # ---------------------------------------------------------
            uncertainty_px = uncertainty * image.shape[0]

            ax.fill_between(
                x,
                y - uncertainty_px,
                y + uncertainty_px,
                alpha=0.18,
                color="royalblue",
                linewidth=0.0,
            )
            ax.plot(
                x,
                y - uncertainty_px,
                color="royalblue",
                linewidth=0.8,
                linestyle="--",
            )

            ax.plot(
                x,
                y + uncertainty_px,
                color="royalblue",
                linewidth=0.8,
                linestyle="--",
            )

            # Ground-truth line
            ax.plot(
                x,
                y,
                color="navy",
                linewidth=line_width,
                linestyle="-",
            )

        else:
            ax.text(
                0.5,
                0.03,
                "GT missing",
                transform=ax.transAxes,
                ha="center",
            )

        ax.set_title(f"{title} — GT", fontsize=10)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.show()


def plot_boundary_result(image, boundary, title="", display_dilation=3):
    image = np.asarray(image)
    boundary = np.asarray(boundary, dtype=bool)
    boundary_display = (
        ndimage.binary_dilation(boundary, iterations=display_dilation)
        if display_dilation > 0 else boundary
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        image,
        cmap="gray",
        vmin=np.percentile(image, 1),
        vmax=np.percentile(image, 99.5),
    )
    overlay = np.ma.masked_where(~boundary_display, boundary_display)
    ax.imshow(overlay, cmap="Reds", alpha=1.0, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    plt.show()


def plot_atom_result(image, positions, title=""):
    image = np.asarray(image)
    positions = np.asarray(positions)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        image,
        cmap="gray",
        vmin=np.percentile(image, 1),
        vmax=np.percentile(image, 99.5),
    )
    if len(positions) > 0:
        ax.scatter(
            positions[:, 1], positions[:, 0],
            s=5, facecolors="red", linewidths=1.2,
        )
    ax.set_title(f"{title}\nDetected atoms: {len(positions)}")
    ax.axis("off")
    plt.show()


# -----------------------------------------------------------------------------
# Challenge 1 — Ground-truth accuracy
# -----------------------------------------------------------------------------

def load_ground_truth(file: Path, gt_dir: Path) -> np.ndarray:
    gt_file = Path(gt_dir) / f"{Path(file).stem}_ground_truth.npz"
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth not found:\n{gt_file}")
    gt = np.load(gt_file, allow_pickle=True)
    return gt["boundary_mask"].astype(bool)


def boundary_metrics(prediction, ground_truth, tolerance_px=5) -> dict:
    """Tolerance-aware boundary precision, recall, F1, and nearest-boundary errors."""
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)

    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have the same shape.")
    if ground_truth.sum() == 0:
        raise ValueError("Ground-truth boundary is empty.")
    if prediction.sum() == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_pred_to_gt_distance": np.nan,
            "mean_gt_to_pred_distance": np.inf,
            "mean_symmetric_boundary_error": np.inf,
        }

    distance_to_gt = ndimage.distance_transform_edt(~ground_truth)
    distance_to_prediction = ndimage.distance_transform_edt(~prediction)

    precision = float(np.mean(distance_to_gt[prediction] <= tolerance_px))
    recall = float(np.mean(distance_to_prediction[ground_truth] <= tolerance_px))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    pred_to_gt = float(np.mean(distance_to_gt[prediction]))
    gt_to_pred = float(np.mean(distance_to_prediction[ground_truth]))

    return {
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
        "mean_pred_to_gt_distance": pred_to_gt,
        "mean_gt_to_pred_distance": gt_to_pred,
        "mean_symmetric_boundary_error": (pred_to_gt + gt_to_pred) / 2,
    }


def evaluate_against_ground_truth(detector: Callable, images, files, titles, gt_dir: Path, tolerance_px=5):
    rows = []
    for image, file, title in zip(images, files, titles):
        gt = load_ground_truth(file, gt_dir)
        prediction = detector(image)
        metrics = boundary_metrics(prediction, gt, tolerance_px=tolerance_px)
        rows.append({"image": title, **metrics})
    return pd.DataFrame(rows)


def summarize_accuracy(results: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "mean_precision": results["precision"].mean(),
            "mean_recall": results["recall"].mean(),
            "mean_f1": results["f1"].mean(),
            "mean_boundary_error_px": results["mean_symmetric_boundary_error"].mean(),
        },
        name="Clean boundary accuracy",
    )


# -----------------------------------------------------------------------------
# Challenge 1 — Robustness
# -----------------------------------------------------------------------------

def image_dynamic_range(image) -> float:
    image = np.asarray(image, dtype=float)
    low, high = np.percentile(image, [1, 99])
    return max(float(high - low), float(np.finfo(float).eps))


def add_gaussian_noise(image, noise_fraction=0.05, seed=0):
    rng = np.random.default_rng(seed)
    image = np.asarray(image, dtype=float)
    sigma = noise_fraction * image_dynamic_range(image)
    return image + rng.normal(0, sigma, image.shape)


def add_gaussian_blur(image, sigma=1.0):
    return ndimage.gaussian_filter(np.asarray(image, dtype=float), sigma=sigma)


def change_brightness(image, offset_fraction=0.10):
    image = np.asarray(image, dtype=float)
    return image + offset_fraction * image_dynamic_range(image)


def change_contrast(image, factor=0.8):
    image = np.asarray(image, dtype=float)
    mean = np.mean(image)
    return mean + factor * (image - mean)


def add_intensity_gradient(image, strength=0.20, angle_degrees=45):
    image = np.asarray(image, dtype=float)
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    xx = xx / max(width - 1, 1) - 0.5
    yy = yy / max(height - 1, 1) - 0.5
    theta = np.deg2rad(angle_degrees)
    gradient = np.cos(theta) * xx + np.sin(theta) * yy
    max_abs = np.max(np.abs(gradient))
    if max_abs > 0:
        gradient = gradient / max_abs
    return image + strength * image_dynamic_range(image) * gradient


def make_robustness_tests() -> dict[str, Callable]:
    return {
        "Clean": lambda image: image,
        "Gaussian noise 5%": lambda image: add_gaussian_noise(image, 0.05, seed=42),
        "Gaussian blur sigma=1": lambda image: add_gaussian_blur(image, sigma=1.0),
        "Brightness +10%": lambda image: change_brightness(image, offset_fraction=0.10),
        "Contrast -20%": lambda image: change_contrast(image, factor=0.80),
        "Intensity gradient 20%": lambda image: add_intensity_gradient(image, strength=0.20, angle_degrees=45),
    }


# def evaluate_robustness(detector: Callable, images, files, titles, gt_dir: Path, tests=None, tolerance_px=5):
#     """Evaluate every perturbed image against the same human ground truth."""
#     if tests is None:
#         tests = make_robustness_tests()

#     rows = []
#     for condition, perturb in tests.items():
#         for image, file, title in zip(images, files, titles):
#             gt = load_ground_truth(file, gt_dir)
#             prediction = detector(perturb(np.asarray(image).copy()))
#             metrics = boundary_metrics(prediction, gt, tolerance_px=tolerance_px)
#             rows.append(
#                 {
#                     "condition": condition,
#                     "image": title,
#                     "precision": metrics["precision"],
#                     "recall": metrics["recall"],
#                     "f1": metrics["f1"],
#                     "mean_boundary_error_px": metrics["mean_symmetric_boundary_error"],
#                 }
#             )
#     return pd.DataFrame(rows)


from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd


def evaluate_robustness(
    detector,
    images,
    files,
    titles,
    gt_dir: Path,
    tests=None,
    tolerance_px=5,
    n_jobs=None,
):
    """
    Faster robustness evaluation.

    Improvements:
    1. Ground truth is loaded only once per image.
    2. Images are converted to NumPy only once.
    3. Conditions/images are evaluated in parallel.
    """

    if tests is None:
        tests = make_robustness_tests()

    # ------------------------------------------------------------
    # Cache everything that does not change between conditions
    # ------------------------------------------------------------
    images_np = [
        np.asarray(image)
        for image in images
    ]

    ground_truths = [
        load_ground_truth(file, gt_dir)
        for file in files
    ]

    # ------------------------------------------------------------
    # One evaluation job
    # ------------------------------------------------------------
    def evaluate_one(condition, perturb, image, gt, title):

        perturbed = perturb(image.copy())

        prediction = detector(perturbed)

        metrics = boundary_metrics(
            prediction,
            gt,
            tolerance_px=tolerance_px,
        )

        return {
            "condition": condition,
            "image": title,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mean_boundary_error_px":
                metrics["mean_symmetric_boundary_error"],
        }

    # ------------------------------------------------------------
    # Create jobs
    # ------------------------------------------------------------
    jobs = [
        (
            condition,
            perturb,
            image,
            gt,
            title,
        )
        for condition, perturb in tests.items()
        for image, gt, title in zip(
            images_np,
            ground_truths,
            titles,
        )
    ]

    # ------------------------------------------------------------
    # Run jobs in parallel
    # ------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:

        futures = [
            executor.submit(
                evaluate_one,
                condition,
                perturb,
                image,
                gt,
                title,
            )
            for condition, perturb, image, gt, title in jobs
        ]

        rows = [
            future.result()
            for future in futures
        ]

    return pd.DataFrame(rows)


def summarize_robustness(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("condition", sort=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_boundary_error_px=("mean_boundary_error_px", "mean"),
        )
        .reset_index()
    )
    clean = summary.loc[summary["condition"] == "Clean", "mean_f1"]
    if len(clean) == 0:
        raise ValueError("Robustness results do not contain a Clean condition.")
    summary["f1_drop_from_clean"] = float(clean.iloc[0]) - summary["mean_f1"]
    return summary


# -----------------------------------------------------------------------------
# Speed
# -----------------------------------------------------------------------------

def evaluate_speed(detector: Callable, images, repeats=5):
    """Time only detector(image); loading, plotting, and scoring are excluded."""
    if len(images) == 0:
        raise ValueError("No images supplied for speed evaluation.")

    _ = detector(images[0])  # warm-up
    rows = []

    for repeat in range(repeats):
        for image_index, image in enumerate(images):
            start = time.perf_counter()
            _ = detector(image)
            rows.append(
                {
                    "repeat": repeat,
                    "image": image_index,
                    "runtime_seconds": time.perf_counter() - start,
                }
            )
    return pd.DataFrame(rows)


def summarize_speed(results: pd.DataFrame) -> pd.Series:
    mean_time = float(results["runtime_seconds"].mean())
    return pd.Series(
        {
            "mean_seconds_per_image": mean_time,
            "median_seconds_per_image": float(results["runtime_seconds"].median()),
            "slowest_seconds": float(results["runtime_seconds"].max()),
            "images_per_second": 1.0 / mean_time,
        },
        name="Speed",
    )


# -----------------------------------------------------------------------------
# Challenge 2 — Atom-finding helpers
# -----------------------------------------------------------------------------

def validate_atom_output(image, positions) -> bool:
    positions = np.asarray(positions)
    assert positions.ndim == 2, f"Expected 2D array, got {positions.shape}"
    assert positions.shape[1] == 2, f"Expected shape (N, 2), got {positions.shape}"
    assert np.issubdtype(positions.dtype, np.number), "Coordinates must be numeric"

    if len(positions) > 0:
        assert np.all(np.isfinite(positions)), "Coordinates contain NaN or inf"
        rows, cols = positions[:, 0], positions[:, 1]
        assert np.all((rows >= 0) & (rows < image.shape[0])), "Atom row outside image"
        assert np.all((cols >= 0) & (cols < image.shape[1])), "Atom column outside image"
    return True


def atom_match_score(
    reference,
    candidate,
    tolerance=2.0
):
    """
    Fast comparison of two atom-coordinate sets.

    An atom is considered matched if another atom is found
    within `tolerance` pixels.

    Returns:
        precision
        recall
        F1
    """

    reference = np.asarray(
        reference,
        dtype=float
    ).reshape(-1, 2)

    candidate = np.asarray(
        candidate,
        dtype=float
    ).reshape(-1, 2)

    # ------------------------------------------------------------
    # Empty cases
    # ------------------------------------------------------------

    if len(reference) == 0 and len(candidate) == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0
        }

    if len(reference) == 0 or len(candidate) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    # ------------------------------------------------------------
    # Build fast spatial lookup structures
    # ------------------------------------------------------------

    reference_tree = cKDTree(
        reference
    )

    candidate_tree = cKDTree(
        candidate
    )

    # ------------------------------------------------------------
    # Precision:
    # How many candidate atoms are close to reference atoms?
    # ------------------------------------------------------------

    candidate_distances, _ = reference_tree.query(
        candidate,
        k=1
    )

    precision = np.mean(
        candidate_distances <= tolerance
    )

    # ------------------------------------------------------------
    # Recall:
    # How many reference atoms are recovered?
    # ------------------------------------------------------------

    reference_distances, _ = candidate_tree.query(
        reference,
        k=1
    )

    recall = np.mean(
        reference_distances <= tolerance
    )

    # ------------------------------------------------------------
    # F1
    # ------------------------------------------------------------

    if precision + recall == 0:
        f1 = 0.0

    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def atom_repeatability(detector: Callable, images, repeats=3, tolerance=2.0) -> float:
    scores = []
    for image in images:
        reference = detector(image)
        for _ in range(repeats - 1):
            candidate = detector(image)
            scores.append(atom_match_score(reference, candidate, tolerance=tolerance)["f1"])
    return float(np.mean(scores)) if scores else np.nan


def atom_noise_robustness(
    detector,
    images,
    noise_levels=(0.01, 0.03, 0.05, 0.10),
    tolerance=2.0,
    seed=42,
):

    rows = []

    # ============================================================
    # Calculate CLEAN detections only once
    # ============================================================

    clean_results = []

    for image in images:

        clean_results.append(
            detector(image)
        )

    # ============================================================
    # Test noise levels
    # ============================================================

    for noise_level in noise_levels:

        f1_scores = []
        count_ratios = []

        for image_index, image in enumerate(images):

            reference = clean_results[
                image_index
            ]

            noisy = add_gaussian_noise(
                image,
                noise_fraction=noise_level,
                seed=seed + image_index
            )

            candidate = detector(
                noisy
            )

            result = atom_match_score(
                reference,
                candidate,
                tolerance=tolerance
            )

            f1_scores.append(
                result["f1"]
            )

            if len(reference) > 0:

                count_ratios.append(
                    len(candidate)
                    / len(reference)
                )

        rows.append({

            "noise_level_fraction":
                noise_level,

            "mean_F1_vs_clean":
                np.mean(f1_scores),

            "min_F1_vs_clean":
                np.min(f1_scores),

            "mean_count_ratio":
                (
                    np.mean(count_ratios)
                    if count_ratios
                    else np.nan
                )
        })

    return pd.DataFrame(rows)


def atom_photometric_robustness(detector: Callable, images, transforms=((0.8, 0.0), (1.2, 0.0), (1.0, 0.1)), tolerance=2.0):
    rows = []
    for gain, offset_fraction in transforms:
        f1_scores = []
        for image in images:
            image = np.asarray(image, dtype=float)
            reference = detector(image)
            transformed = gain * image + offset_fraction * image_dynamic_range(image)
            candidate = detector(transformed)
            f1_scores.append(atom_match_score(reference, candidate, tolerance=tolerance)["f1"])
        rows.append(
            {
                "gain": gain,
                "offset_fraction": offset_fraction,
                "mean_F1_vs_clean": np.mean(f1_scores),
            }
        )
    return pd.DataFrame(rows)
