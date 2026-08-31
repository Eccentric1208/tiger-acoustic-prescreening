"""
Ranger-ready PDF export.

Overlays the KDE heatmap on a downloaded satellite image of the survey area
and exports as an A3 print-ready PDF, complete with legend, north arrow,
sensor markers, and metadata. This is the actual deliverable a forest
ranger takes into the field for camera-trap placement.

Design choices:

  1. USER-SUPPLIED SATELLITE IMAGE. We deliberately do NOT scrape map tiles
     — that would require internet access at deployment sites (there is
     none) and hits Google/Mapbox ToS grey areas. Instead, the ranger
     downloads an image ahead of time (Google Earth screenshot, Bhuvan
     export, whatever they have) and provides its corner coordinates.

  2. GEOREFERENCING VIA CORNERS. Given the satellite image plus its
     (lat, lon) bounding box, we can place every sensor pixel-accurately.
     This is a simplification — real georeferencing uses affine transforms
     for rotated/skewed images — but for a rectangular screenshot aligned
     to N/S/E/W, corners are enough.

  3. A3 LANDSCAPE. Standard for field maps. Bigger than the phone-sized
     PNG heatmap, small enough to fit in a ranger's day pack.

  4. TRANSPARENT HEATMAP OVERLAY. Alpha-blended so the underlying terrain
     stays visible — rangers place cameras based on both prey activity
     AND terrain features (ridges, water sources, game trails).

Run from project root as a smoke test:
    python -m src.mapping.pdf_export
which uses the demo sensor CSV from Phase 5 and a synthetic satellite
image (green rectangle) to demonstrate the export.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
import numpy as np

from src.mapping.heatmap import (
    Sensor,
    compute_heatmap,
    project_to_meters,
    load_sensors_csv,
)


# ------------ data types ------------

@dataclass
class MapBounds:
    """
    Lat/lon corners of the satellite image, assumed axis-aligned (image
    top = north, image right = east).
    """
    north_lat: float
    south_lat: float
    east_lon: float
    west_lon: float


# ------------ georeferencing ------------

def sensor_to_image_coords(
    sensors: list,
    bounds: MapBounds,
    image_width_px: int,
    image_height_px: int,
) -> tuple:
    """
    Convert each sensor's (lat, lon) to pixel (x, y) inside the satellite image.

    Image origin is TOP-LEFT (matplotlib imshow convention), so:
      - x=0 is the west edge, x=image_width_px is the east edge
      - y=0 is the north edge, y=image_height_px is the south edge (y grows down)
    """
    xs, ys = [], []
    for s in sensors:
        # Fraction across the image, from west edge and from north edge
        x_frac = (s.lon - bounds.west_lon) / (bounds.east_lon - bounds.west_lon)
        y_frac = (bounds.north_lat - s.lat) / (bounds.north_lat - bounds.south_lat)
        xs.append(x_frac * image_width_px)
        ys.append(y_frac * image_height_px)
    return np.array(xs), np.array(ys)


# ------------ main PDF export ------------

def export_pdf(
    sensors: list,
    satellite_image_path: Optional[str],
    bounds: MapBounds,
    pdf_path: str,
    survey_name: str = "AITE Camera-Trap Placement",
    reserve_name: str = "",
    dates: str = "",
    heatmap_alpha: float = 0.55,
    bandwidth_m: float = 500.0,
) -> None:
    """
    Build and save the ranger-facing A3 PDF.

    Args:
        sensors: list of Sensor records with detection_count populated
        satellite_image_path: path to the downloaded satellite image (PNG/JPG).
            If None, uses a plain light-grey background — useful for
            demonstrating the layout without a real satellite image.
        bounds: MapBounds describing the corners of the satellite image
        pdf_path: where to save the output PDF
        survey_name / reserve_name / dates: printed in the header
        heatmap_alpha: transparency of the heatmap overlay (0=invisible, 1=opaque)
        bandwidth_m: KDE bandwidth in meters (see heatmap.py)
    """
    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)

    # ---- load background ----
    if satellite_image_path and Path(satellite_image_path).exists():
        satellite = plt.imread(satellite_image_path)
        img_h, img_w = satellite.shape[:2]
    else:
        # Fallback: solid pale-green rectangle so the layout is visible even
        # without a real satellite image.
        img_h, img_w = 1000, 1400
        satellite = np.ones((img_h, img_w, 3)) * np.array([0.85, 0.90, 0.80])

    # ---- compute KDE and reproject to image pixels ----
    xx_m, yy_m, density = compute_heatmap(
        sensors, grid_resolution=300, bandwidth_m=bandwidth_m
    )
    # We need the heatmap in pixel coordinates so it aligns with the satellite
    # image. Convert the meter-grid corners to lat/lon, then to pixels.
    _, _, (center_lat, center_lon) = project_to_meters(sensors)
    METERS_PER_DEG_LAT = 111_320.0
    m_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(center_lat))

    # Corner (west, south) and (east, north) of the heatmap in lat/lon
    heatmap_west_lon = center_lon + xx_m.min() / m_per_deg_lon
    heatmap_east_lon = center_lon + xx_m.max() / m_per_deg_lon
    heatmap_south_lat = center_lat + yy_m.min() / METERS_PER_DEG_LAT
    heatmap_north_lat = center_lat + yy_m.max() / METERS_PER_DEG_LAT

    # Convert to pixel coords in the satellite image
    def latlon_to_px(lat, lon):
        x_frac = (lon - bounds.west_lon) / (bounds.east_lon - bounds.west_lon)
        y_frac = (bounds.north_lat - lat) / (bounds.north_lat - bounds.south_lat)
        return x_frac * img_w, y_frac * img_h

    hx_min, hy_max = latlon_to_px(heatmap_south_lat, heatmap_west_lon)
    hx_max, hy_min = latlon_to_px(heatmap_north_lat, heatmap_east_lon)

    # Sensor pixel coordinates
    sensor_px_x, sensor_px_y = sensor_to_image_coords(sensors, bounds, img_w, img_h)
    counts = np.array([s.detection_count for s in sensors])

    # ---- draw ----
    # A3 landscape ≈ 16.5 x 11.7 inches
    fig = plt.figure(figsize=(16.5, 11.7))

    # Main map takes most of the figure; a narrow right column holds the
    # legend, north arrow, and metadata.
    ax_map = fig.add_axes([0.03, 0.05, 0.65, 0.90])
    ax_side = fig.add_axes([0.72, 0.05, 0.26, 0.90])
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    ax_side.axis("off")

    # 1) Satellite background
    ax_map.imshow(satellite, extent=(0, img_w, 0, img_h), origin="upper")

    # 2) Heatmap overlay (with transparency)
    ax_map.imshow(
        density,
        extent=(hx_min, hx_max, hy_min, hy_max),
        origin="upper",
        cmap="hot_r",
        alpha=heatmap_alpha,
        interpolation="bilinear",
    )

    # 3) Sensor markers
    max_count = max(int(counts.max()), 1)
    sizes = 60 + 220 * (counts / max_count)
    ax_map.scatter(
        sensor_px_x, sensor_px_y,
        s=sizes,
        facecolors="white",
        edgecolors="black",
        linewidths=1.4,
        zorder=5,
    )
    for s, xp, yp in zip(sensors, sensor_px_x, sensor_px_y):
        ax_map.annotate(
            f"{s.sensor_id}\n({s.detection_count})",
            (xp, yp),
            xytext=(7, -3),
            textcoords="offset points",
            fontsize=7,
            color="black",
            zorder=6,
        )

    ax_map.set_xlim(0, img_w)
    ax_map.set_ylim(0, img_h)
    ax_map.invert_yaxis()
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_linewidth(1.5)

    # ---- side panel: header, north arrow, legend, metadata ----
    y_cursor = 0.98

    ax_side.text(0.0, y_cursor, survey_name,
                 fontsize=16, fontweight="bold", va="top")
    y_cursor -= 0.04
    if reserve_name:
        ax_side.text(0.0, y_cursor, reserve_name, fontsize=12, va="top")
        y_cursor -= 0.03
    if dates:
        ax_side.text(0.0, y_cursor, dates, fontsize=10, va="top", color="grey")
        y_cursor -= 0.05

    # North arrow
    y_cursor -= 0.02
    arrow_y_top = y_cursor
    arrow_y_bot = arrow_y_top - 0.07
    arrow = FancyArrowPatch(
        (0.15, arrow_y_bot), (0.15, arrow_y_top),
        arrowstyle="-|>", mutation_scale=25,
        color="black", linewidth=2,
        transform=ax_side.transAxes,
    )
    ax_side.add_patch(arrow)
    ax_side.text(0.15, arrow_y_top + 0.01, "N",
                 fontsize=14, fontweight="bold",
                 ha="center", va="bottom")
    y_cursor = arrow_y_bot - 0.03

    # Heatmap legend (gradient bar)
    y_cursor -= 0.02
    ax_side.text(0.0, y_cursor, "Prey activity density",
                 fontsize=11, fontweight="bold", va="top")
    y_cursor -= 0.03
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)[::-1]  # top=high
    legend_x, legend_w = 0.05, 0.12
    legend_h = 0.20
    legend_y_top = y_cursor
    legend_y_bot = legend_y_top - legend_h
    ax_side.imshow(
        gradient, cmap="hot_r", aspect="auto",
        extent=(legend_x, legend_x + legend_w, legend_y_bot, legend_y_top),
        transform=ax_side.transAxes,
    )
    ax_side.text(legend_x + legend_w + 0.02, legend_y_top,
                 "High (place cameras)", fontsize=9, va="top")
    ax_side.text(legend_x + legend_w + 0.02, legend_y_bot,
                 "Low", fontsize=9, va="bottom")
    y_cursor = legend_y_bot - 0.04

    # Sensor legend
    ax_side.text(0.0, y_cursor, "Sensors",
                 fontsize=11, fontweight="bold", va="top")
    y_cursor -= 0.04
    for label, marker_size, offset in [
        ("Low detections", 60, 0),
        ("Medium", 140, 0.04),
        ("High", 280, 0.09),
    ]:
        yy = y_cursor - offset
        ax_side.scatter([0.10], [yy], s=marker_size,
                        facecolors="white", edgecolors="black",
                        linewidths=1.4, transform=ax_side.transAxes)
        ax_side.text(0.22, yy, label, fontsize=9, va="center",
                     transform=ax_side.transAxes)
    y_cursor -= 0.15

    # Metadata block
    y_cursor -= 0.02
    ax_side.text(0.0, y_cursor, "Deployment", fontsize=11,
                 fontweight="bold", va="top")
    y_cursor -= 0.03
    total_detections = int(counts.sum())
    active_sensors = int((counts > 0).sum())
    meta_lines = [
        f"Sensors deployed:   {len(sensors)}",
        f"Sensors with detections: {active_sensors}",
        f"Total detections:    {total_detections}",
        f"KDE bandwidth:       {int(bandwidth_m)} m",
        f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    for line in meta_lines:
        ax_side.text(0.0, y_cursor, line, fontsize=9, va="top",
                     family="monospace")
        y_cursor -= 0.025

    # Footer note
    ax_side.text(
        0.0, 0.02,
        "Bright red = highest prey alarm-call activity.\n"
        "Recommended camera-trap zones follow the red\n"
        "contours, favouring terrain features (ridges,\n"
        "water, game trails) within them.",
        fontsize=8, va="bottom", color="dimgrey",
        transform=ax_side.transAxes,
    )

    # ---- save PDF ----
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)   # no bbox_inches='tight' — it was rescaling everything
    plt.close(fig)


# ------------ smoke test ------------

if __name__ == "__main__":
    print("PDF export smoke test\n")

    # Reuse the demo sensor CSV generated by heatmap.py's smoke test
    csv_path = Path("results/sensors_demo.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run 'python -m src.mapping.heatmap' first."
        )
    sensors = load_sensors_csv(str(csv_path))
    print(f"Loaded {len(sensors)} sensors from {csv_path}")

    # Bounds: 5x5 grid at 500m spacing = ~2km x 2km, centered on our demo
    # coordinates. Add ~500m padding so the map extends past the outer sensors.
    center_lat, center_lon = 21.7500, 79.2500
    METERS_PER_DEG_LAT = 111_320.0
    m_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(center_lat))
    half_span_lat = 1500 / METERS_PER_DEG_LAT
    half_span_lon = 1500 / m_per_deg_lon
    bounds = MapBounds(
        north_lat=center_lat + half_span_lat,
        south_lat=center_lat - half_span_lat,
        east_lon=center_lon + half_span_lon,
        west_lon=center_lon - half_span_lon,
    )

    pdf_path = "results/plots/ranger_map_demo.pdf"
    export_pdf(
        sensors,
        satellite_image_path=None,  # None -> pale green placeholder background
        bounds=bounds,
        pdf_path=pdf_path,
        survey_name="AITE Camera-Trap Placement — Demo",
        reserve_name="Demo Reserve (synthetic data)",
        dates="Survey window: 72 hours (synthetic)",
    )
    print(f"PDF -> {pdf_path}")
    print("\nOpen the PDF. Layout: map on left, legend + metadata on right.")
    print("For real use: supply satellite_image_path pointing to a downloaded")
    print("PNG/JPG, and set bounds to the image's actual lat/lon corners.")