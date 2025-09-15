#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download NAIP imagery for the U.S. Mid-Atlantic region using Planetary Computer STAC
and produce:
  - One 1 m mosaic GeoTIFF per state
  - One 1 m combined mosaic for the union of all states

States covered (configurable by CLI): DC, DE, MD, NJ, NY, PA, VA, WV

This script is self-contained and borrows robust download/mosaic logic from download.py:
- Finds most recent NAIP year per geometry
- Reuses already-downloaded tiles from --outdir
- Downloads remaining tiles in parallel (threaded)
- Mosaics with per-tile masks and nearest resampling to avoid artifacts

Usage example:
  python download_mid_atlantic.py \
    --outdir /work/nvme/bfbw/fperez2/datasets/naip_midatl \
    --tmpdir /work/nvme/bfbw/fperez2/tmp \
    --tile-workers 32 --overwrite
"""

import os
import argparse
import multiprocessing
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box as shapely_box

from pystac_client import Client
import planetary_computer as pc

import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling

import requests
from tqdm import tqdm


# ------------------------- Config -------------------------
MID_ATLANTIC_STATES = [
    "DC", "DE", "MD", "NJ", "NY", "PA", "VA", "WV",
]

args_global = None
tile_cache_index: Dict[str, str] = {}


# ----------------------- Utilities ------------------------
def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def mb(n: int) -> float:
    return n / 1048576.0


def stac_client() -> Client:
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


def year_from_item(item) -> Optional[int]:
    try:
        props = item.properties
        dt = props.get("datetime") or props.get("start_datetime")
        if dt:
            return int(dt[:4])
        y = props.get("naip:year")
        return int(y) if y is not None else None
    except Exception:
        return None


def most_recent_naip_year_for_geom(geom: BaseGeometry) -> Optional[int]:
    c = stac_client()
    search = c.search(
        collections=["naip"],
        intersects=geom,
        limit=1000,
    )
    years: List[int] = []
    for it in search.items():
        y = year_from_item(it)
        if y is not None:
            years.append(int(y))
    if not years:
        return None
    return max(years)


def _iter_naip_items_for_year(geom: BaseGeometry, year: int) -> List[object]:
    c = stac_client()
    search = c.search(
        collections=["naip"],
        intersects=geom,
        limit=1000,
        query={"naip:year": {"eq": year}},
    )
    return list(search.items())


def _asset_href_from_item(item) -> Optional[str]:
    try:
        signed = pc.sign(item)
    except Exception:
        signed = item
    asset = signed.assets.get("image") if hasattr(signed, "assets") else None
    if asset and getattr(asset, "href", None):
        return asset.href
    try:
        for a in signed.assets.values():
            media = getattr(a, "media_type", "") or ""
            if "tiff" in media or str(a.href).lower().endswith((".tif", ".tiff")):
                return a.href
    except Exception:
        pass
    return None


def _download_url(url: str, dest_path: str, overwrite: bool, timeout: int = 60) -> str:
    exists = os.path.exists(dest_path)
    size = os.path.getsize(dest_path) if exists else 0
    if (not overwrite) and exists and size > 0:
        return dest_path
    tmp_path = dest_path + ".part"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, dest_path)
            return dest_path
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if attempts >= 3:
                raise
    return dest_path


def _find_existing_tile(base_name: str) -> Optional[str]:
    if base_name in tile_cache_index:
        return tile_cache_index[base_name]
    root_dir = getattr(args_global, "outdir", None)
    if not root_dir:
        tile_cache_index[base_name] = None  # type: ignore
        return None
    candidate = os.path.join(root_dir, base_name)
    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
        tile_cache_index[base_name] = candidate
        return candidate
    try:
        for entry in os.scandir(root_dir):
            if entry.is_dir():
                candidate = os.path.join(entry.path, base_name)
                if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                    tile_cache_index[base_name] = candidate
                    return candidate
    except Exception:
        pass
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if base_name in filenames:
            candidate = os.path.join(dirpath, base_name)
            if os.path.getsize(candidate) > 0:
                tile_cache_index[base_name] = candidate
                return candidate
    tile_cache_index[base_name] = None  # type: ignore
    return None


def _asset_base_name_from_item(item) -> Optional[str]:
    try:
        # Use raw assets without signing to compute stable base name
        if hasattr(item, "assets") and item.assets:
            img = item.assets.get("image")
            if img and getattr(img, "href", None):
                return os.path.basename(str(img.href).split("?", 1)[0])
            for a in item.assets.values():
                href = getattr(a, "href", "") or ""
                if href and (href.lower().endswith((".tif", ".tiff")) or "tiff" in (getattr(a, "media_type", "") or "")):
                    return os.path.basename(str(href).split("?", 1)[0])
    except Exception:
        return None
    return None


def _download_item_fresh_signed(item, dest: str, overwrite: bool) -> str:
    # Generate a fresh signed href per attempt to avoid SAS expiry during long runs
    attempts = 0
    while attempts < 3:
        attempts += 1
        url = _asset_href_from_item(item)
        if not url:
            raise RuntimeError("Could not resolve asset href for item")
        try:
            return _download_url(url, dest, overwrite)
        except Exception:
            if attempts >= 3:
                raise
    return dest


def download_tiles_for_geom(geom: BaseGeometry, year: int, dst_dir: str, max_items: int, overwrite: bool) -> List[str]:
    ensure_dir(dst_dir)
    items = _iter_naip_items_for_year(geom, year)
    if not items:
        return []
    entries: List[Tuple[object, str]] = []  # (item, dest)
    for it in items:
        base = _asset_base_name_from_item(it) or f"{it.id}.tif"
        dest = os.path.join(dst_dir, base)
        entries.append((it, dest))
    if max_items and max_items > 0:
        entries = entries[: max_items]

    results: List[str] = []
    to_download: List[Tuple[str, str]] = []
    reused = 0
    for it, dest in entries:
        base = os.path.basename(dest)
        existing = _find_existing_tile(base)
        if existing and (not overwrite):
            # Reuse existing tile path directly (no copy) to enable resume
            results.append(existing)
            print(f"Reused existing tile: {base}")
            reused += 1
        else:
            to_download.append((it, dest))

    if to_download:
        workers = max(1, getattr(args_global, "tile_workers", 8))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_target = {ex.submit(_download_item_fresh_signed, it, dest, overwrite): dest for it, dest in to_download}
            for fut in tqdm(as_completed(future_to_target), total=len(future_to_target), desc="NAIP tiles", unit="tile"):
                dest = future_to_target[fut]
                fut.result()
                results.append(dest)

    if reused:
        try:
            print(f"Reused {reused} existing tiles from outdir")
        except Exception:
            pass

    return results


def mosaic_rgb_1m(tifs: List[str], bbox_geom: BaseGeometry, out_tif: str, target_crs: str = "EPSG:5070"):
    if not tifs:
        raise ValueError("No TIFFs to mosaic")
    srcs = [rasterio.open(p) for p in tifs]
    try:
        vrts = []
        for s in srcs:
            if s.crs != target_crs:
                try:
                    vrts.append(WarpedVRT(
                        s,
                        crs=target_crs,
                        resampling=Resampling.nearest,
                        add_alpha=True,
                        warp_extras={"NUM_THREADS": "ALL_CPUS"}
                    ))
                except TypeError:
                    vrts.append(WarpedVRT(s, crs=target_crs, resampling=Resampling.nearest, add_alpha=True))
            else:
                try:
                    try:
                        vrts.append(WarpedVRT(
                            s,
                            add_alpha=True,
                            resampling=Resampling.nearest,
                            warp_extras={"NUM_THREADS": "ALL_CPUS"}
                        ))
                    except TypeError:
                        vrts.append(WarpedVRT(s, add_alpha=True, resampling=Resampling.nearest))
                except Exception:
                    vrts.append(s)
        minx, miny, maxx, maxy = transform_bounds("EPSG:4326", target_crs, *bbox_geom.bounds, densify_pts=21)
        mosaic_arr, mosaic_transform = rio_merge(
            vrts,
            bounds=(minx, miny, maxx, maxy),
            res=(1.0, 1.0),
            method="first",
            indexes=[1, 2, 3],
        )
        meta = srcs[0].meta.copy()
        meta.update({
            "count": mosaic_arr.shape[0],
            "height": mosaic_arr.shape[1],
            "width": mosaic_arr.shape[2],
            "transform": mosaic_transform,
            "compress": "deflate",
            "BIGTIFF": "IF_NEEDED",
            "crs": target_crs,
        })
        if "nodata" in meta:
            try:
                del meta["nodata"]
            except Exception:
                pass
        ensure_dir(os.path.dirname(out_tif))
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(mosaic_arr)
    finally:
        for v in vrts:
            try:
                if isinstance(v, WarpedVRT):
                    v.close()
            except Exception:
                pass
        for s in srcs:
            try:
                s.close()
            except Exception:
                pass


def fetch_states(states: List[str]) -> gpd.GeoDataFrame:
    # Use Natural Earth admin 1 (states/provinces) via geopandas datasets
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        raise SystemExit("geopandas.datasets naturalearth_lowres not available; install geopandas with datasets.")
    usa = world[world["iso_a3"] == "USA"].to_crs(4326)
    # This dataset has only country, not states. Fallback: use cartographic boundary states from TIGER via simple HTTP
    # For a self-contained script, we fetch a lightweight states file from Census TIGER cartographic boundaries (1:20m)
    import io, zipfile, requests as rq
    url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"
    r = rq.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        members = [m for m in zf.namelist() if m.endswith('.shp') or m.endswith('.dbf') or m.endswith('.shx') or m.endswith('.prj')]
        tmpdir = os.path.join(getattr(args_global, "tmpdir", "/tmp"), "states20m")
        os.makedirs(tmpdir, exist_ok=True)
        for m in members:
            zf.extract(m, tmpdir)
        shp_path = [os.path.join(tmpdir, m) for m in members if m.endswith('.shp')][0]
    gdf = gpd.read_file(shp_path).to_crs(4326)
    gdf = gdf[gdf["STUSPS"].isin(states)]
    return gdf[["STUSPS", "geometry"]].reset_index(drop=True)


def main():
    # Favor multi-threaded warps/reads during mosaics
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    # Set a reasonable GDAL cache (in MB). Adjust if needed for your system.
    os.environ.setdefault("GDAL_CACHEMAX", "1024")

    # Dynamic default for tile download workers (network-bound, benefit from higher concurrency)
    try:
        tile_workers_default = max(8, min(64, multiprocessing.cpu_count() * 4))
    except Exception:
        tile_workers_default = 16

    ap = argparse.ArgumentParser(description="Download NAIP 1m for Mid-Atlantic (per-state and merged)")
    ap.add_argument("--states", nargs="*", default=MID_ATLANTIC_STATES, help="State USPS codes to include")
    ap.add_argument("--year", type=int, default=None, help="Force NAIP year; default: most recent per state")
    ap.add_argument("--max-per-year", type=int, default=0, help="Max tiles per state (0=all)")
    ap.add_argument("--tile-workers", type=int, default=tile_workers_default, help="Parallel tile downloads per state")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for mosaics and tiles reuse")
    ap.add_argument("--tmpdir", type=str, default=None, help="Temporary directory (defaults to parent of outdir)")
    ap.add_argument("--overwrite", action="store_true", help="Redownload/rebuild if exists")
    args = ap.parse_args()

    global args_global
    args_global = args

    ensure_dir(args.outdir)
    if args.tmpdir is None:
        try:
            args.tmpdir = os.path.dirname(os.path.abspath(args.outdir)) or "/tmp"
        except Exception:
            args.tmpdir = "/tmp"
    ensure_dir(args.tmpdir)

    states_gdf = fetch_states(args.states)

    # Per-state processing
    per_state_outputs: List[str] = []
    def process_state(row) -> str:
        st = row.STUSPS
        geom = row.geometry
        print(f"\n=== {st} ===")
        # year
        year = args.year or most_recent_naip_year_for_geom(geom)
        if year is None:
            print(f"[{st}] No NAIP available.")
            return ""
        print(f"[{st}] Year: {year}")
        # out paths
        state_dir = os.path.join(args.outdir, st, str(year))
        ensure_dir(state_dir)
        out_tif = os.path.join(args.outdir, f"naip_{st}_{year}_mosaic_1m.tif")
        if (not args.overwrite) and os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
            print(f"[{st}] Exists → {os.path.basename(out_tif)} ({mb(os.path.getsize(out_tif)):.1f} MB). Skipping.")
            return out_tif
        # temp
        import tempfile
        with tempfile.TemporaryDirectory(prefix=f"naip_{st}_", dir=args.tmpdir) as tmpdir:
            tiles = download_tiles_for_geom(geom, year, tmpdir, args.max_per_year, args.overwrite)
            if not tiles:
                print(f"[{st}] No tiles; skipping.")
                return ""
            print(f"[{st}] Mosaicking {len(tiles)} tiles → {out_tif}")
            mosaic_rgb_1m(tiles, geom, out_tif)
            try:
                print(f"[{st}] Done: {os.path.basename(out_tif)} ({mb(os.path.getsize(out_tif)):.1f} MB)")
            except OSError:
                print(f"[{st}] Done: {os.path.basename(out_tif)}")
            return out_tif

    # Process states sequentially to limit memory and output size pressure
    for _, row in states_gdf.iterrows():
        out = process_state(row)
        if out:
            per_state_outputs.append(out)

    # Combined mosaic across states
    if per_state_outputs:
        print("\n=== Combined mosaic ===")
        combined_tif = os.path.join(args.outdir, "naip_midatlantic_mosaic_1m.tif")
        if (not args.overwrite) and os.path.exists(combined_tif) and os.path.getsize(combined_tif) > 0:
            print(f"Exists → {os.path.basename(combined_tif)} ({mb(os.path.getsize(combined_tif)):.1f} MB). Skipping.")
            return
        total_geom = unary_union(list(states_gdf.geometry)).buffer(0)
        # Open per-state mosaics as inputs
        srcs = [rasterio.open(p) for p in per_state_outputs]
        try:
            # Assume all are already EPSG:5070 1 m RGB
            bounds = total_geom.bounds
            minx, miny, maxx, maxy = transform_bounds("EPSG:4326", srcs[0].crs, *bounds, densify_pts=21)
            mosaic_arr, mosaic_transform = rio_merge(
                srcs,
                bounds=(minx, miny, maxx, maxy),
                res=(1.0, 1.0),
                method="first",
                indexes=[1, 2, 3],
            )
            meta = srcs[0].meta.copy()
            meta.update({
                "count": mosaic_arr.shape[0],
                "height": mosaic_arr.shape[1],
                "width": mosaic_arr.shape[2],
                "transform": mosaic_transform,
                "compress": "deflate",
                "BIGTIFF": "IF_NEEDED",
            })
            if "nodata" in meta:
                try:
                    del meta["nodata"]
                except Exception:
                    pass
            ensure_dir(os.path.dirname(combined_tif))
            with rasterio.open(combined_tif, "w", **meta) as dst:
                dst.write(mosaic_arr)
            try:
                print(f"Done: {os.path.basename(combined_tif)} ({mb(os.path.getsize(combined_tif)):.1f} MB)")
            except OSError:
                print(f"Done: {os.path.basename(combined_tif)}")
        finally:
            for s in srcs:
                try:
                    s.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()


