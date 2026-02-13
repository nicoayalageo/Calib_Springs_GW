#!/usr/bin/env python3
"""
calibrate_springs.py
====================
Calibrate MODFLOW 6 models using springs as observation points.

This script processes a DEM-derived drainage network, locates spring discharge
points along that network, and extracts simulated hydraulic heads at the
corresponding MODFLOW 6 drain (DRN) cells.  The output is a PEST-compatible
CSV where each column is a spring name and each value is the simulated head.

Usage
-----
    python calibrate_springs.py config.yaml
    python calibrate_springs.py config.yaml --validate-only
    python calibrate_springs.py config.yaml --keep-intermediates

Dependencies
------------
See requirements.txt.  SAGA GIS is **not** required; all vector snapping is
handled natively with shapely / geopandas.  WhiteboxTools (pip-installable) is
used for flow-routing operations that have no pure-Python alternative.
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import flopy
import geopandas as gpd
import numpy as np
import pandas as pd
import whitebox
import yaml
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger("calibrate_springs")


# ---------------------------------------------------------------------------
# Section 1 – Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All user-configurable parameters, loaded from a YAML file."""

    # Paths – inputs
    model_dir: Path = field(default_factory=Path)
    sim_name: str = ""
    dem_path: Path = field(default_factory=Path)
    springs_path: Path = field(default_factory=Path)
    domain_path: Path = field(default_factory=Path)

    # Paths – outputs
    output_dir: Path = field(default_factory=Path)

    # Coordinate reference system
    crs: str = "EPSG:32719"

    # Model query parameters
    stress_period: list = field(default_factory=lambda: [0, 0])
    layer: int = 0

    # DEM / stream extraction
    stream_threshold: int = 1000

    # Snap & iteration parameters
    snap_distance: float = 20.0
    max_iterations: int = 20

    # Runtime flags (set via CLI, not YAML)
    keep_intermediates: bool = False
    validate_only: bool = False

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as fh:
            raw = yaml.safe_load(fh)

        paths_section = raw.get("paths", {})
        model_section = raw.get("model", {})
        dem_section = raw.get("dem", {})
        run_section = raw.get("run", {})

        return cls(
            model_dir=Path(paths_section.get("model_dir", "")),
            sim_name=model_section.get("sim_name", ""),
            dem_path=Path(paths_section.get("dem", "")),
            springs_path=Path(paths_section.get("springs", "")),
            domain_path=Path(paths_section.get("domain", "")),
            output_dir=Path(paths_section.get("output_dir", "")),
            crs=model_section.get("crs", "EPSG:32719"),
            stress_period=model_section.get("stress_period", [0, 0]),
            layer=model_section.get("layer", 0),
            stream_threshold=dem_section.get("stream_threshold", 1000),
            snap_distance=run_section.get("snap_distance", 20.0),
            max_iterations=run_section.get("max_iterations", 20),
        )

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if everything is OK)."""
        errors: list[str] = []

        for label, p in [
            ("model_dir", self.model_dir),
            ("dem", self.dem_path),
            ("springs", self.springs_path),
            ("domain", self.domain_path),
        ]:
            if not p.exists():
                errors.append(f"Path does not exist: {label} = {p}")

        try:
            CRS.from_user_input(self.crs)
        except Exception:
            errors.append(f"Invalid CRS: {self.crs}")

        if len(self.stress_period) != 2:
            errors.append(
                "stress_period must be [kstp, kper], e.g. [0, 0]"
            )

        return errors


# ---------------------------------------------------------------------------
# Section 2 – DEM processing (WhiteboxTools)
# ---------------------------------------------------------------------------

def process_dem(
    dem_path: Path,
    work_dir: Path,
    stream_threshold: int,
) -> tuple[Path, Path, Path]:
    """Run flow-accumulation workflow and extract a stream vector network.

    Returns (d8_pointer_path, flow_accum_path, drainage_network_path).
    """
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.work_dir = str(work_dir)

    dem_out = work_dir / "DEM_filled.tif"
    d8_ptr = work_dir / "d8_pointer.tif"
    flow_acc = work_dir / "flow_accum.tif"
    stream_ras = work_dir / "streams.tif"
    drainage_shp = work_dir / "drainage_network.shp"

    logger.info("Running flow accumulation workflow …")
    wbt.flow_accumulation_full_workflow(
        dem=str(dem_path),
        out_dem=str(dem_out),
        out_pntr=str(d8_ptr),
        out_accum=str(flow_acc),
        out_type="cells",
    )

    logger.info("Extracting streams (threshold=%d) …", stream_threshold)
    wbt.extract_streams(
        flow_accum=str(flow_acc),
        output=str(stream_ras),
        threshold=stream_threshold,
    )

    logger.info("Converting stream raster to vector …")
    wbt.raster_streams_to_vector(
        str(stream_ras), str(d8_ptr), output=str(drainage_shp)
    )

    return d8_ptr, flow_acc, drainage_shp


# ---------------------------------------------------------------------------
# Section 3 – MODFLOW model helpers
# ---------------------------------------------------------------------------

def load_model(cfg: Config):
    """Load the MODFLOW 6 simulation and return (gwf, head_array)."""
    logger.info("Loading MODFLOW 6 model from %s …", cfg.model_dir)
    sim = flopy.mf6.MFSimulation.load(
        sim_name=cfg.sim_name, sim_ws=str(cfg.model_dir)
    )
    gwf = sim.get_model()
    kstp, kper = cfg.stress_period
    head = gwf.output.head().get_data(kstpkper=(kstp, kper))
    logger.info("Heads loaded for stress period kstp=%d, kper=%d", kstp, kper)
    return gwf, head


def build_domain_cells(gwf, layer: int, crs: str) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of active-cell polygons for *layer*."""
    idomain = gwf.dis.idomain.array
    rows, cols = np.where(idomain[layer] == 1)
    polygons = [
        Polygon(gwf.modelgrid.get_cell_vertices(r, c))
        for r, c in zip(rows, cols)
    ]
    return gpd.GeoDataFrame(geometry=polygons, crs=crs)


def extract_drn_data(gwf, head: np.ndarray, crs: str, stress_period: list):
    """Extract DRN cell polygons and identify active (discharging) cells.

    Returns (gdf_drn_all, gdf_drn_active).
    """
    drn_pkg = gwf.get_package("drn")
    if drn_pkg is None:
        raise RuntimeError(
            "The model does not contain a DRN package. "
            "Spring calibration requires drain boundaries."
        )

    drn_data = drn_pkg.stress_period_data.get_data()[0]
    cbc = gwf.output.budget()
    kstp, kper = stress_period
    drn_cbc = cbc.get_data(text="DRN", kstpkper=(kstp, kper))[0]

    grouped: dict[tuple, list] = defaultdict(list)
    for cell, cbc_cell in zip(drn_data, drn_cbc):
        cellid, elev, cond, *_ = cell
        layer, row, col = cellid
        grouped[(row, col)].append(
            {"layer": layer, "elev": elev, "q": cbc_cell["q"]}
        )

    polygons, elevs, actives, cellids = [], [], [], []
    for (row, col), vals in grouped.items():
        verts = gwf.modelgrid.get_cell_vertices(row, col)
        polygons.append(Polygon(verts))
        elevs.append(np.mean([v["elev"] for v in vals]))
        actives.append(any(v["q"] < 0 for v in vals))
        cellids.append((vals[0]["layer"], row, col))

    gdf = gpd.GeoDataFrame(
        {
            "elev_drn": elevs,
            "activo": actives,
            "n_capas": [len(grouped[(r, c)]) for (r, c) in grouped],
            "cellid": cellids,
        },
        geometry=polygons,
        crs=crs,
    )
    return gdf, gdf[gdf["activo"]].copy()


# ---------------------------------------------------------------------------
# Section 4 – Geometric utilities
# ---------------------------------------------------------------------------

def snap_points_to_lines(
    points_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Snap each point to its nearest location on the nearest line.

    Replaces the SAGA *Snap Points to Lines* tool using pure shapely.
    """
    from shapely.ops import nearest_points

    snapped = points_gdf.copy()
    all_lines = lines_gdf.geometry.values
    new_geoms = []
    for pt in points_gdf.geometry:
        dists = [line.distance(pt) for line in all_lines]
        nearest_line = all_lines[int(np.argmin(dists))]
        projected = nearest_line.interpolate(nearest_line.project(pt))
        new_geoms.append(projected)
    snapped = snapped.set_geometry(new_geoms)
    return snapped


def snap_points_to_points(
    source_gdf: gpd.GeoDataFrame,
    target_gdf: gpd.GeoDataFrame,
    max_distance: float,
) -> gpd.GeoDataFrame:
    """Snap *source* points to the nearest *target* point within *max_distance*.

    Replaces the SAGA *Snap Points to Points* tool using geopandas.
    """
    joined = gpd.sjoin_nearest(
        source_gdf,
        target_gdf,
        how="left",
        max_distance=max_distance,
        distance_col="_snap_dist",
    )
    # Keep only the closest match per source feature
    joined = joined.loc[
        joined.groupby(joined.index)["_snap_dist"].idxmin()
    ]
    joined = joined.drop(columns=["_snap_dist"], errors="ignore")
    return joined


def extract_line_endpoints(
    gdf: gpd.GeoDataFrame, indices: list[int] | None = None
) -> gpd.GeoDataFrame:
    """Extract specific vertices (default: first and last) from lines."""
    if indices is None:
        indices = [0, -1]
    points, attrs = [], []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        lines = (
            list(geom.geoms)
            if isinstance(geom, MultiLineString)
            else [geom]
        )
        for line in lines:
            coords = list(line.coords)
            for i in indices:
                if -len(coords) <= i < len(coords):
                    pt = Point(coords[i])
                    points.append(pt)
                    attrs.append({**row.to_dict(), "vertex_index": i})
    return gpd.GeoDataFrame(attrs, geometry=points, crs=gdf.crs)


def find_line_intersections(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Return points where lines in *gdf1* intersect lines in *gdf2*."""
    crs = gdf1.crs
    gdf2 = gdf2.to_crs(crs)
    sindex = gdf2.sindex

    records = []
    for _, row1 in gdf1.iterrows():
        candidates = list(sindex.intersection(row1.geometry.bounds))
        for ci in candidates:
            row2 = gdf2.iloc[ci]
            inter = row1.geometry.intersection(row2.geometry)
            if inter.is_empty:
                continue
            pts = list(inter.geoms) if inter.geom_type == "MultiPoint" else [inter]
            for pt in pts:
                if pt.geom_type != "Point":
                    continue
                records.append(
                    {
                        **{f"gdf1_{k}": v for k, v in row1.items()},
                        **{f"gdf2_{k}": v for k, v in row2.items()},
                        "geometry": pt,
                    }
                )
    return gpd.GeoDataFrame(records, crs=crs) if records else gpd.GeoDataFrame()


def points_are_equal(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> bool:
    """Check whether two point GeoDataFrames contain the same coordinates."""
    if gdf1.empty or gdf2.empty:
        return gdf1.empty and gdf2.empty
    c1 = {(round(p.x, 3), round(p.y, 3)) for p in gdf1.geometry}
    c2 = {(round(p.x, 3), round(p.y, 3)) for p in gdf2.geometry}
    return c1 == c2


def vector_difference(
    input_gdf: gpd.GeoDataFrame, overlay_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Geometric difference: parts of *input_gdf* not covered by *overlay_gdf*."""
    if overlay_gdf.empty:
        return input_gdf.copy()
    input_gdf = input_gdf.to_crs(overlay_gdf.crs)
    return gpd.overlay(input_gdf, overlay_gdf, how="difference", keep_geom_type=True)


# ---------------------------------------------------------------------------
# Section 5 – Spring processing (trace flowpaths, filter, transfer attributes)
# ---------------------------------------------------------------------------

def trace_flowpaths(
    seed_points_gdf: gpd.GeoDataFrame,
    d8_pointer: Path,
    work_dir: Path,
    crs: str,
    label: str = "flowpaths",
) -> gpd.GeoDataFrame:
    """Trace downslope flow paths from *seed_points_gdf* using WBT.

    Writes temporary shapefiles into *work_dir* and returns the resulting
    polylines as an in-memory GeoDataFrame.
    """
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.work_dir = str(work_dir)

    seed_path = work_dir / f"seed_{label}.shp"
    flow_ras = work_dir / f"flow_{label}.tif"
    flow_shp = work_dir / f"flow_{label}.shp"

    seed_points_gdf.to_file(seed_path)
    wbt.trace_downslope_flowpaths(
        str(seed_path), str(d8_pointer), output=str(flow_ras)
    )
    wbt.raster_streams_to_vector(
        str(flow_ras), str(d8_pointer), output=str(flow_shp)
    )

    gdf = gpd.read_file(flow_shp)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    return gdf


def filter_unique_endpoints(
    flow_lines: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Extract endpoints that appear on exactly one line (non-junction)."""
    all_pts: list[dict] = []
    for idx, row in flow_lines.iterrows():
        geom = row.geometry
        lines = (
            list(geom.geoms)
            if isinstance(geom, MultiLineString)
            else [geom]
        )
        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            for pos, label in [(0, "start"), (-1, "end")]:
                all_pts.append(
                    {
                        "id_linea": idx,
                        "tipo_punto": label,
                        "geometry": Point(coords[pos]),
                    }
                )

    counter: dict[tuple, int] = defaultdict(int)
    for p in all_pts:
        key = (round(p["geometry"].x, 3), round(p["geometry"].y, 3))
        counter[key] += 1

    unique = [
        p
        for p in all_pts
        if counter[
            (round(p["geometry"].x, 3), round(p["geometry"].y, 3))
        ]
        == 1
    ]
    return gpd.GeoDataFrame(unique, crs=flow_lines.crs)


def filter_points_in_domain(
    points: gpd.GeoDataFrame, domain: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Keep only points that fall inside the model domain polygon."""
    inside = gpd.sjoin(
        points, domain, how="inner", predicate="within"
    ).drop_duplicates(subset=["geometry"])
    return inside


def transfer_spring_attributes(
    filtered_pts: gpd.GeoDataFrame,
    snapped_springs: gpd.GeoDataFrame,
    domain_gdf: gpd.GeoDataFrame,
    snap_distance: float,
    crs: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Transfer spring name & elevation to the closest filtered endpoint.

    Returns (attributed_points, spring_buffers).
    """
    # Snap original spring points to the filtered endpoints
    joined = gpd.sjoin_nearest(
        filtered_pts,
        snapped_springs[["name", "altitude", "geometry"]].rename(
            columns={"altitude": "elev_manan"}
        ),
        how="left",
        max_distance=snap_distance,
        distance_col="_snap_dist",
    )
    joined = joined.loc[joined.groupby(joined.index)["_snap_dist"].idxmin()]
    joined = joined.drop(columns=["_snap_dist", "index_right"], errors="ignore")
    joined = joined.drop_duplicates(subset="id_linea", keep="first")

    # Keep only points that intersect the domain cells
    result = gpd.sjoin(
        joined, domain_gdf, predicate="intersects"
    )
    result = result.drop(columns=["index_right"], errors="ignore")

    # Build a small buffer around each spring for spatial matching later
    buf = result.copy()
    buf = buf.set_geometry(buf.geometry.buffer(2))
    buf = buf.set_crs(crs)

    return result, buf


# ---------------------------------------------------------------------------
# Section 6 – Expand upstream (iterative algorithm)
# ---------------------------------------------------------------------------

def expand_upstream(
    stream_net_real: gpd.GeoDataFrame,
    initial_flow_lines: gpd.GeoDataFrame,
    initial_points: gpd.GeoDataFrame,
    max_iterations: int,
    crs: str,
) -> gpd.GeoDataFrame:
    """Iteratively expand the drainage network upstream from spring flow lines.

    At each iteration the algorithm:
    1. Buffers the current seed points by 1 m.
    2. Selects real-network segments touching any buffer.
    3. Merges them with the accumulated drainage via ``pd.concat``
       (replaces the old ``wbt.union`` call that wrote temp files each iteration).
    4. Extracts start-vertices of selected segments.
    5. Removes vertices that coincide with self-intersection points.
    6. Stops when the vertex set is stable.

    Returns the final set of upstream endpoint (seed) points.
    """
    sindex = stream_net_real.sindex
    accumulated_lines = initial_flow_lines.copy()
    vertices_prev: Optional[gpd.GeoDataFrame] = None
    current_points = initial_points.copy()

    for i in range(max_iterations):
        logger.info("Upstream expansion – iteration %d", i + 1)

        # A. Buffer current points
        buf_union = current_points.geometry.buffer(1.0).union_all()

        # B. Select real-network segments touching the buffer
        candidate_idx = list(sindex.intersection(buf_union.bounds))
        candidates = stream_net_real.iloc[candidate_idx]
        selected = candidates[candidates.intersects(buf_union)].copy()

        if selected.empty:
            logger.info("No new segments found – stopping.")
            break

        # C. Merge via pd.concat (replaces wbt.union)
        accumulated_lines = pd.concat(
            [accumulated_lines, selected], ignore_index=True
        )
        accumulated_lines = gpd.GeoDataFrame(
            accumulated_lines, crs=crs
        )

        # D. Start-vertices of selected segments
        if selected.crs is None:
            selected = selected.set_crs(crs)
        new_vertices = extract_line_endpoints(selected, indices=[0])

        # E. Remove vertices at self-intersection points
        intersections = find_line_intersections(selected, selected)
        if not intersections.empty:
            intersections = intersections[
                intersections.geometry.geom_type.isin(["Point", "MultiPoint"])
            ].explode(index_parts=True)
            if not intersections.empty:
                buf_inters = intersections.copy()
                buf_inters = buf_inters.set_geometry(
                    buf_inters.geometry.buffer(1.0)
                )
                buf_inters = gpd.GeoDataFrame(
                    geometry=buf_inters.geometry, crs=crs
                )
                new_vertices = vector_difference(new_vertices, buf_inters)

        # F. Convergence check
        if vertices_prev is not None and points_are_equal(
            new_vertices, vertices_prev
        ):
            logger.info("Upstream expansion converged at iteration %d.", i + 1)
            break

        vertices_prev = new_vertices.copy()
        current_points = new_vertices

    else:
        logger.warning(
            "Upstream expansion did not converge within %d iterations.",
            max_iterations,
        )

    return current_points, accumulated_lines


# ---------------------------------------------------------------------------
# Section 7 – Downstream graph + BFS search
# ---------------------------------------------------------------------------

def build_downstream_graph(
    drainage: gpd.GeoDataFrame, tolerance: float = 1.0
) -> dict[int, list[int]]:
    """Build a directed graph of downstream connectivity.

    For each line, the end-point is matched to the start-point of downstream
    segments within *tolerance* metres.
    """
    end_points = drainage.geometry.apply(
        lambda g: Point(list(g.coords)[-1])
    )
    start_points = drainage.geometry.apply(
        lambda g: Point(list(g.coords)[0])
    )
    gdf_start = gpd.GeoDataFrame(
        {"idx": drainage.index, "geometry": start_points}, crs=drainage.crs
    )
    sindex = gdf_start.sindex

    graph: dict[int, list[int]] = defaultdict(list)
    for seg_idx, end_pt in zip(drainage.index, end_points):
        candidates = list(sindex.intersection(end_pt.buffer(tolerance).bounds))
        for ci in candidates:
            if end_pt.distance(gdf_start.loc[ci].geometry) < tolerance:
                graph[seg_idx].append(gdf_start.loc[ci].idx)
    return graph


def find_first_downstream(
    point_geom,
    drainage: gpd.GeoDataFrame,
    target_layer: gpd.GeoDataFrame,
    graph: dict[int, list[int]],
) -> Optional[pd.Series]:
    """BFS downstream from *point_geom* until a feature in *target_layer* is hit."""
    dists = drainage.geometry.distance(point_geom)
    start_seg = dists.idxmin()

    visited: set[int] = set()
    queue = [start_seg]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        seg_geom = drainage.loc[current].geometry
        hits = target_layer[target_layer.intersects(seg_geom)]
        if not hits.empty:
            return hits.iloc[0]

        queue.extend(graph.get(current, []))

    return None


# ---------------------------------------------------------------------------
# Section 8 – Process a single spring point
# ---------------------------------------------------------------------------

def process_spring_point(
    idx_punto: int,
    puntos: gpd.GeoDataFrame,
    drainage: gpd.GeoDataFrame,
    active_drn_cells: gpd.GeoDataFrame,
    spring_buffers: gpd.GeoDataFrame,
    graph: dict[int, list[int]],
    head: np.ndarray,
) -> dict:
    """Evaluate one spring point: find downstream DRN cell and spring buffer.

    All data is passed explicitly – no global variables.
    """
    punto = puntos.loc[idx_punto]

    celda_drn = find_first_downstream(
        punto.geometry, drainage, active_drn_cells, graph
    )
    buf_match = find_first_downstream(
        punto.geometry, drainage, spring_buffers, graph
    )

    name = buf_match["name"] if buf_match is not None else None
    elev_manan = buf_match.get("elev_manan") if buf_match is not None else None

    elev_drn = None
    if celda_drn is not None and "cellid" in celda_drn.index:
        cellid_raw = celda_drn["cellid"]
        if isinstance(cellid_raw, str):
            try:
                cellid_raw = ast.literal_eval(cellid_raw)
            except Exception:
                logger.warning("Cannot parse cellid: %s", cellid_raw)
                cellid_raw = None
        if isinstance(cellid_raw, tuple) and len(cellid_raw) == 3:
            layer, row, col = cellid_raw
            try:
                elev_drn = float(head[layer, row, col])
            except IndexError:
                logger.warning("Head index out of range: %s", cellid_raw)

    dif = (
        elev_drn - elev_manan
        if elev_manan is not None and elev_drn is not None
        else None
    )

    return {
        "id_punto": idx_punto,
        "name": name,
        "elev_manan": elev_manan,
        "elev_drn": elev_drn,
        "dif": dif,
        "geometry": punto.geometry,
    }


# ---------------------------------------------------------------------------
# Section 9 – Format output & save
# ---------------------------------------------------------------------------

def format_output(
    results: list[dict], crs: str
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Clean results and produce a PEST-compatible pivoted DataFrame."""
    gdf = gpd.GeoDataFrame(results, crs=crs)
    gdf_clean = gdf.dropna(subset=["name", "elev_drn"])
    if gdf_clean.empty:
        logger.warning("No valid spring observations found.")
        return gdf, pd.DataFrame()

    # Keep the observation with highest simulated head per spring
    gdf_filtered = gdf_clean.loc[
        gdf_clean.groupby("name")["elev_drn"].idxmax()
    ].sort_values(["name", "elev_drn"], ascending=[True, False])

    elev = gdf_filtered[["name", "elev_drn"]].copy()
    elev["time"] = 1.0
    pivoted = elev.pivot(index="time", columns="name", values="elev_drn")
    return gdf_filtered, pivoted


def save_results(
    pivoted: pd.DataFrame,
    gdf_filtered: gpd.GeoDataFrame,
    output_dir: Path,
    keep_intermediates: bool,
) -> Path:
    """Write the pivoted CSV (and optionally the shapefile) to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "spring_observations.csv"
    pivoted.to_csv(csv_path)
    logger.info("Results saved to %s", csv_path)

    if keep_intermediates and not gdf_filtered.empty:
        shp_path = output_dir / "spring_results.shp"
        gdf_filtered.to_file(shp_path)
        logger.info("Shapefile saved to %s", shp_path)

    return csv_path


# ---------------------------------------------------------------------------
# Section 10 – Main pipeline
# ---------------------------------------------------------------------------

def main(cfg: Config) -> None:
    """Orchestrate the full calibration pipeline."""
    t0 = time.time()

    # Decide where to write WBT intermediates
    if cfg.keep_intermediates:
        wbt_dir = cfg.output_dir / "intermediates"
        wbt_dir.mkdir(parents=True, exist_ok=True)
        _tmp_ctx = None
    else:
        _tmp_ctx = tempfile.TemporaryDirectory(prefix="springs_")
        wbt_dir = Path(_tmp_ctx.name)

    try:
        # --- Step 1: Process DEM -------------------------------------------
        logger.info("=== Step 1/8: Processing DEM ===")
        d8_ptr, flow_acc, drainage_shp = process_dem(
            cfg.dem_path, wbt_dir, cfg.stream_threshold
        )
        stream_net = gpd.read_file(drainage_shp)
        if stream_net.crs is None:
            stream_net = stream_net.set_crs(cfg.crs)

        # --- Step 2: Load MODFLOW model ------------------------------------
        logger.info("=== Step 2/8: Loading MODFLOW model ===")
        gwf, head = load_model(cfg)
        domain_cells = build_domain_cells(gwf, cfg.layer, cfg.crs)
        _, drn_active = extract_drn_data(
            gwf, head, cfg.crs, cfg.stress_period
        )

        # --- Step 3: Snap springs to drainage network ----------------------
        logger.info("=== Step 3/8: Snapping springs to drainage ===")
        springs = gpd.read_file(cfg.springs_path)
        if springs.crs is None:
            springs = springs.set_crs(cfg.crs)
        snapped_springs = snap_points_to_lines(springs, stream_net)

        # --- Step 4: Trace initial flowpaths -------------------------------
        logger.info("=== Step 4/8: Tracing initial flowpaths ===")
        flow_lines = trace_flowpaths(
            snapped_springs, d8_ptr, wbt_dir, cfg.crs, label="initial"
        )

        # --- Step 5: Filter endpoints & transfer attributes ----------------
        logger.info("=== Step 5/8: Filtering endpoints & transferring attributes ===")
        unique_pts = filter_unique_endpoints(flow_lines)
        domain_poly = gpd.read_file(cfg.domain_path)
        pts_in_domain = filter_points_in_domain(unique_pts, domain_poly)

        attributed_pts, spring_bufs = transfer_spring_attributes(
            pts_in_domain,
            snapped_springs,
            domain_cells,
            cfg.snap_distance,
            cfg.crs,
        )

        # --- Step 6: Expand upstream network -------------------------------
        logger.info("=== Step 6/8: Expanding upstream network ===")
        expanded_pts, _ = expand_upstream(
            stream_net,
            flow_lines,
            attributed_pts,
            cfg.max_iterations,
            cfg.crs,
        )

        # --- Step 7: Re-trace from expanded endpoints ----------------------
        logger.info("=== Step 7/8: Re-tracing from expanded endpoints ===")
        final_flow = trace_flowpaths(
            expanded_pts, d8_ptr, wbt_dir, cfg.crs, label="final"
        )
        # Clip to domain
        final_clipped = gpd.overlay(
            final_flow, domain_cells, how="intersection"
        )
        final_clipped = final_clipped.explode(index_parts=False).reset_index(
            drop=True
        )

        # --- Step 8: Process each spring point -----------------------------
        logger.info("=== Step 8/8: Processing spring points ===")
        graph = build_downstream_graph(final_clipped)

        results = []
        for idx in tqdm(
            list(expanded_pts.index), desc="Processing springs"
        ):
            result = process_spring_point(
                idx,
                expanded_pts,
                final_clipped,
                drn_active,
                spring_bufs,
                graph,
                head,
            )
            results.append(result)

        gdf_filtered, pivoted = format_output(results, cfg.crs)
        csv_path = save_results(
            pivoted, gdf_filtered, cfg.output_dir, cfg.keep_intermediates
        )

        elapsed = time.time() - t0
        logger.info("Pipeline completed in %.1f s", elapsed)
        logger.info("Output: %s", csv_path)
        if not pivoted.empty:
            logger.info(
                "Springs found: %d columns in output CSV", len(pivoted.columns)
            )

    finally:
        if _tmp_ctx is not None:
            _tmp_ctx.cleanup()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(
        description="Calibrate MODFLOW 6 using spring observations."
    )
    parser.add_argument(
        "config", type=Path, help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration without running the pipeline.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Save intermediate shapefiles for debugging.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    cfg = Config.from_yaml(args.config)
    cfg.keep_intermediates = args.keep_intermediates
    cfg.validate_only = args.validate_only

    errors = cfg.validate()
    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        sys.exit(1)
    logger.info("Configuration validated successfully.")

    if args.validate_only:
        logger.info("--validate-only: exiting without running the pipeline.")
        sys.exit(0)

    main(cfg)


if __name__ == "__main__":
    cli()
