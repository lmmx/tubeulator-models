"""Load, summarise, and filter TfL GTFS data to Greater London."""

from __future__ import annotations

from datetime import date

import city2graph as c2g
import geopandas as gpd
import osmnx as ox

from .defaults import repo_root, resolve_analysis, resolve_data, resolve_filter


__all__ = ["load_gtfs", "build_travel_graph", "filter_to_region", "load_london_graph"]


def load_gtfs() -> dict:
    """Parse the GTFS zip from the path declared in defaults.toml."""
    path = repo_root() / resolve_data()["gtfs_path"]
    return c2g.load_gtfs(path)


def build_travel_graph(
    gtfs_data: dict,
    calendar_start: str,
    calendar_end: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Summarise trips into stop-level nodes and edges, projected to target CRS."""
    crs = resolve_data()["target_crs"]
    nodes, edges = c2g.travel_summary_graph(
        gtfs_data,
        calendar_start=calendar_start,
        calendar_end=calendar_end,
    )
    return nodes.to_crs(epsg=crs), edges.to_crs(epsg=crs)


def filter_to_region(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    region: str | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Clip nodes and edges to a named region boundary (via OSMnx geocoding)."""
    crs = resolve_data()["target_crs"]
    boundary = ox.geocode_to_gdf(region or resolve_filter()["region"]).to_crs(epsg=crs)

    nodes_in = gpd.sjoin(nodes, boundary, how="inner").drop(columns=["index_right"])
    edges_in = gpd.sjoin(edges, boundary, how="inner").drop(columns=["index_right"])

    from_ok = edges_in.index.get_level_values("from_stop_id").isin(nodes_in.index)
    to_ok = edges_in.index.get_level_values("to_stop_id").isin(nodes_in.index)
    edges_in = edges_in[from_ok & to_ok]

    return nodes_in, edges_in


def load_london_graph(
    calendar_start: str | None = None,
    calendar_end: str | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """End-to-end convenience: load → summarise → filter to Greater London."""
    cfg = resolve_analysis()
    today = date.today().strftime("%Y%m%d")
    start = calendar_start or cfg.get("calendar_start") or today
    end = calendar_end or cfg.get("calendar_end") or today

    gtfs_data = load_gtfs()
    nodes, edges = build_travel_graph(gtfs_data, start, end)
    return filter_to_region(nodes, edges)
