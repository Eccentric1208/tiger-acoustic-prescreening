"""
Heatmap generation from sensor detection counts.

Input: a list of (sensor_id, lat, lon, detection_count) records — one per
AudioMoth in the deployment grid.

Output: a 2D kernel density estimate (KDE) surface visualising where prey
activity was concentrated across the survey area. This is what a ranger
uses to pick camera-trap locations for the AITE census.

Design choices:

  1. WEIGHTED KDE. Each sensor's contribution to the density is weighted by
     its detection count. A sensor with 50 detections pulls the heatmap
     harder than one with 3.

  2. LAT/LON PROJECTION. We convert (lat, lon) to a local metric grid
     using an equirectangular projection anchored on the survey centroid.
     Fine for the ~5km x 5km scale of a 25-sensor grid; would need proper
     UTM for regional scales.

  3. BANDWIDTH. The KDE bandwidth (~how far each detection "spreads") is
     set based on sensor spacing. Default assumes 500m grid spacing per
     the project spec; adjust if your deployment differs.

Run from project root as a smoke test:
    python -m src.mapping.heatmap
which builds a synthetic 25-sensor grid with a fake tiger hotspot and
saves the heatmap PNG.
"""

from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


# ------------ data types ------------

@dataclass
class Sensor:
    """One AudioMoth's location and detection count."""
    sensor_id: str
    lat: float
    lon: float
    detection_count: int


# ------------ projection ------------

def project_to_meters(
    sensors: List[Sensor],
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Convert (lat, lon) coordinates to local (x, y) in meters, anchored on
    the centroid of the sensor grid.

    Uses equirectangular projection: fine for small areas (up to ~10km),
    breaks down for continental scales. For AITE reserve-level surveys
    (typically 5-50km blocks) it's accurate to within a couple of meters
    of a proper UTM projection.

    Returns (x_meters, y_meters, (center_lat, center_lon)).
    """
    lats = np.array([s.lat for s in sensors])
    lons = np.array([s.lon for s in sensors])
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    # 1 degree of latitude ≈ 111,320 meters (approximately constant)
    # 1 degree of longitude ≈ 111,320 * cos(latitude) meters (shrinks at poles)
    METERS_PER_DEG_LAT = 111_320.0
    meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(center_lat))

    x = (lons - center_lon) * meters_per_deg_lon
    y = (lats - center_lat) * METERS_PER_DEG_LAT
    return x, y, (center_lat, center_lon)


# ------------ KDE computation ------------

def compute_heatmap(
    sensors: List[Sensor],
    grid_resolution: int = 200,
    bandwidth_m: float = 500.0,
    padding_m: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a weighted 2D KDE across the sensor grid.

    Args:
        sensors: list of Sensor records
        grid_resolution: number of grid points per axis (higher = smoother)
        bandwidth_m: kernel bandwidth in meters. Default 500m matches the
            project's 500m sensor spacing — each detection spreads roughly
            one grid cell before falling off.
        padding_m: extend the heatmap this far past the outermost sensor,
            so the edges aren't clipped.

    Returns (xx, yy, density) — meshgrid coordinates and the density surface.
    Density values are relative; the absolute scale doesn't matter for
    visualisation (matplotlib handles the colorbar).
    """
    x, y, _ = project_to_meters(sensors)
    counts = np.array([s.detection_count for s in sensors], dtype=np.float64)

    # If no detections anywhere, return a flat zero surface — don't crash.
    if counts.sum() == 0:
        x_min, x_max = x.min() - padding_m, x.max() + padding_m
        y_min, y_max = y.min() - padding_m, y.max() + padding_m
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution),
        )
        return xx, yy, np.zeros_like(xx)

    # scipy's gaussian_kde takes weights but wants them normalized.
    weights = counts / counts.sum()

    # Bandwidth is expressed as a fraction of the data's std deviation, so we
    # need to convert our bandwidth-in-meters to that scale.
    data = np.vstack([x, y])
    data_std = float(np.std(data))
    bw_factor = bandwidth_m / data_std if data_std > 0 else 1.0

    kde = gaussian_kde(data, weights=weights, bw_method=bw_factor)

    # Build the evaluation grid.
    x_min, x_max = x.min() - padding_m, x.max() + padding_m
    y_min, y_max = y.min() - padding_m, y.max() + padding_m
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(grid_points).reshape(xx.shape)

    return xx, yy, density


# ------------ plotting ------------

def plot_heatmap(
    sensors: List[Sensor],
    save_path: Path,
    title: str = "Prey alarm-call activity heatmap",
    grid_resolution: int = 200,
    bandwidth_m: float = 500.0,
) -> None:
    """
    Render the KDE surface with sensor markers overlaid.

    - Colored heatmap = prey activity density (red = hot, blue = cold)
    - Black circles = sensor locations, sized by detection count
    - Text labels = sensor IDs
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    xx, yy, density = compute_heatmap(
        sensors, grid_resolution=grid_resolution, bandwidth_m=bandwidth_m
    )
    x, y, _ = project_to_meters(sensors)
    counts = np.array([s.detection_count for s in sensors])

    fig, ax = plt.subplots(figsize=(10, 9))

    # Heatmap surface. imshow needs origin='lower' because y=0 is at the
    # bottom in our coordinate system, not the top (which is the image default).
    im = ax.imshow(
        density,
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        origin="lower",
        cmap="hot_r",  # reversed 'hot' — high density = red, low = white
        alpha=0.85,
        aspect="equal",
    )

    # Sensor markers. Marker size scales with detection count so a ranger
    # can quickly see which sensors were noisy at a glance.
    max_count = max(int(counts.max()), 1)
    sizes = 50 + 250 * (counts / max_count)
    ax.scatter(
        x, y,
        s=sizes,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
    )

    # Label each sensor with ID + count
    for s, xi, yi in zip(sensors, x, y):
        ax.annotate(
            f"{s.sensor_id}\n({s.detection_count})",
            (xi, yi),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            color="black",
            zorder=4,
        )

    ax.set_xlabel("East–West offset from grid centre (m)")
    ax.set_ylabel("North–South offset from grid centre (m)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Detection density (relative)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ------------ CSV loader (for real deployments) ------------

def load_sensors_csv(path: str) -> List[Sensor]:
    """
    Load a sensor deployment CSV. Expected columns:
        sensor_id, lat, lon, detection_count
    """
    sensors: List[Sensor] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sensors.append(Sensor(
                sensor_id=row["sensor_id"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                detection_count=int(row["detection_count"]),
            ))
    return sensors


# ------------ smoke test ------------

def _synthetic_grid_with_hotspot() -> List[Sensor]:
    """
    Build a fake 25-sensor 5x5 grid at 500m spacing, with an activity
    hotspot in the north-east quadrant — mimicking what a real tiger
    territory might produce.
    """
    # Centre the grid somewhere plausible in central Indian tiger habitat
    # (Pench-adjacent latitude). Purely for realism in the coordinate values.
    center_lat = 21.7500
    center_lon = 79.2500
    METERS_PER_DEG_LAT = 111_320.0
    m_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(center_lat))
    spacing_m = 500.0
    spacing_deg_lat = spacing_m / METERS_PER_DEG_LAT
    spacing_deg_lon = spacing_m / m_per_deg_lon

    sensors: List[Sensor] = []
    for row in range(5):
        for col in range(5):
            # (row, col) = (0,0) is south-west; (4,4) is north-east
            offset_row = row - 2  # -2..+2
            offset_col = col - 2
            lat = center_lat + offset_row * spacing_deg_lat
            lon = center_lon + offset_col * spacing_deg_lon

            # Hotspot: high counts in NE quadrant (row 3-4, col 3-4)
            in_hotspot = row >= 3 and col >= 3
            base = 45 if in_hotspot else 5
            noise = np.random.randint(-3, 4)
            count = max(0, base + noise)

            sensors.append(Sensor(
                sensor_id=f"AM{row}{col}",
                lat=lat,
                lon=lon,
                detection_count=count,
            ))
    return sensors


if __name__ == "__main__":
    print("Heatmap smoke test\n")

    np.random.seed(42)  # reproducible synthetic counts
    sensors = _synthetic_grid_with_hotspot()

    print(f"Loaded {len(sensors)} sensors")
    total = sum(s.detection_count for s in sensors)
    print(f"Total detections across grid: {total}")

    save_path = Path("results/plots/heatmap_demo.png")
    plot_heatmap(sensors, save_path, title="Demo heatmap (synthetic hotspot NE)")
    print(f"\nHeatmap -> {save_path}")

    # Also write a demo sensors CSV so the user has an example schema
    csv_path = Path("results/sensors_demo.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor_id", "lat", "lon", "detection_count"])
        for s in sensors:
            writer.writerow([s.sensor_id, f"{s.lat:.6f}", f"{s.lon:.6f}",
                             s.detection_count])
    print(f"Demo sensor CSV -> {csv_path}")
    print("\nOpen the PNG — you should see red concentrated in the NE quadrant.")