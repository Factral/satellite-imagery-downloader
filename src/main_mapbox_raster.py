import os
import json
import cv2
import math
import argparse
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm
from glob import glob
import pyvips

from image_downloading import download_image


file_dir = os.path.dirname(__file__)
prefs_path = os.path.join(file_dir, 'mapbox_raster_preferences.json')

default_prefs = {
        'tilesets': 'mapbox.satellite',  # comma-separated tileset IDs username.id
        'mapbox_token': 'pk.eyJ1IjoiZmFjdHJhbCIsImEiOiJjbWV2aDJxZzYwaHB3MnNtdzNhbjdwbnB3In0.lkjBoSHDlskjHDt_xVUGzw',
        'tile_size': 512,  # 512 is default for Raster Tiles API
        'pixel_ratio': 1,  # 1 or 2 (for @2x)
        'raster_format': 'png',  # png, png32, jpg70, jpg80, jpg90
        'channels': 3,
        'dir': os.path.join(file_dir, 'images_mapbox_raster'),
        'output_format': 'vips',  # png | vips | tiff
        'headers': {
            'user-agent': 'satellite-imagery-downloader/1.0'
        },
        'zoom': 16,
        'bbox_km': 20,
        'exclude_usa': False,
        'max_workers': 24,
        'request_timeout': [8.0, 25.0]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download satellite images (Mapbox Raster Tiles API) for top-10 cities by country.')
    parser.add_argument('--exclude-usa', action='store_true', help='Exclude USA from the downloading process.')
    parser.add_argument('--out-dir', type=str, help='Output directory to save images (overrides preferences).')
    parser.add_argument('--format', dest='output_format', choices=['png', 'vips', 'tiff'], help='Output format: png, vips (.v), or tiled tiff (.tif).')
    parser.add_argument('--token', dest='mapbox_token', type=str, help='Mapbox access token.')
    parser.add_argument('--tilesets', type=str, default=None, help='Comma-separated tileset IDs (username.id). Default mapbox.satellite')
    parser.add_argument('--pixel-ratio', type=int, choices=[1, 2], default=None, help='Pixel ratio for @2x (1 or 2).')
    parser.add_argument('--raster-format', type=str, choices=['png', 'png32', 'jpg70', 'jpg80', 'jpg90'], default=None, help='Raster format')
    parser.add_argument('--max-workers', type=int, default=None, help='Max workers for tile downloads.')
    return parser.parse_args()


def ensure_directory(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    return ''.join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')


def compute_bbox_from_center(lat: float, lon: float, window_km: float) -> Tuple[float, float, float, float]:
    delta_meters = window_km * 1000.0
    dlat = delta_meters / 111320.0
    dlon = delta_meters / (111320.0 * max(0.00001, math.cos(math.radians(lat))))
    tl_lat = lat + dlat
    tl_lon = lon - dlon
    br_lat = lat - dlat
    br_lon = lon + dlon
    return tl_lat, tl_lon, br_lat, br_lon


def save_image_with_format(img, out_path: str, fmt: str, channels: int) -> None:
    fmt = fmt.lower()
    if fmt == 'png':
        cv2.imwrite(out_path, img)
        return
    if channels == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        vimage = pyvips.Image.new_from_memory(img_rgb.tobytes(), width, height, 3, 'uchar')
    else:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        height, width = img_rgba.shape[:2]
        vimage = pyvips.Image.new_from_memory(img_rgba.tobytes(), width, height, 4, 'uchar')

    if fmt == 'vips':
        vimage.write_to_file(out_path)
        return
    if fmt == 'tiff':
        vimage.tiffsave(out_path, tile=True, tile_width=256, tile_height=256,
                        bigtiff=True, compression='lzw', pyramid=False)
        return
    cv2.imwrite(out_path, img)


def get_target_country_iso2_codes(exclude_usa: bool) -> List[str]:
    import geonamescache

    gc = geonamescache.GeonamesCache()
    countries: Dict[str, Dict] = gc.get_countries()
    target_iso2: List[str] = []

    for _, country in countries.items():
        continent_code = country.get('continentcode')
        if continent_code in ('EU', 'NA', 'SA'):
            iso2 = country.get('iso')
            if not iso2:
                continue
            if exclude_usa and iso2.upper() == 'US':
                continue
            target_iso2.append(iso2.upper())
    return sorted(set(target_iso2))


def top_cities_by_country(iso2_codes: List[str], top_n: int = 10) -> Dict[str, List[Dict]]:
    import geonamescache

    gc = geonamescache.GeonamesCache()
    cities = gc.get_cities()

    country_to_cities: Dict[str, List[Dict]] = {code: [] for code in iso2_codes}
    for _, city in cities.items():
        code = str(city.get('countrycode', '')).upper()
        if code not in country_to_cities:
            continue
        try:
            population = int(city.get('population') or 0)
        except Exception:
            population = 0
        if population <= 0:
            continue
        lat = city.get('latitude')
        lon = city.get('longitude')
        if lat is None or lon is None:
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            continue
        country_to_cities[code].append({
            'name': city.get('name', 'unknown'),
            'lat': lat_f,
            'lon': lon_f,
            'population': population
        })

    for code in list(country_to_cities.keys()):
        ranked = sorted(country_to_cities[code], key=lambda c: c['population'], reverse=True)
        country_to_cities[code] = ranked[:top_n]

    return country_to_cities


def build_raster_tiles_url_template(tilesets: str, tile_size: int, pixel_ratio: int, raster_format: str, token: str) -> str:
    # Endpoint per Mapbox Raster Tiles API:
    # https://api.mapbox.com/v4/{tilesets}/{z}/{x}/{y}{@2x}.{format}?access_token=TOKEN
    # tilesets: comma-separated up to 15
    ratio_suffix = '@2x' if int(pixel_ratio) == 2 else ''
    # tile_size is 512 by default for Raster Tiles, @2x doubles pixel density, not geographic coverage
    return f'https://api.mapbox.com/v4/{tilesets}' + '/{z}/{x}/{y}' + f'{ratio_suffix}.{raster_format}?access_token={token}'


def run():
    if os.path.isfile(prefs_path):
        with open(prefs_path, 'r', encoding='utf-8') as f:
            prefs = json.loads(f.read())
    else:
        prefs = default_prefs
        with open(prefs_path, 'w', encoding='utf-8') as f:
            json.dump(default_prefs, f, indent=2, ensure_ascii=False)
        print(f'Preferences file created in {prefs_path}')

    args = parse_args()

    out_dir = args.out_dir if getattr(args, 'out_dir', None) else prefs['dir']
    ensure_directory(out_dir)

    zoom = int(prefs.get('zoom', 17))
    channels = int(prefs.get('channels', 3))
    tile_size = int(prefs.get('tile_size', 512))
    bbox_km = float(prefs.get('bbox_km', 20))

    exclude_usa = bool(prefs.get('exclude_usa', False) or args.exclude_usa)
    output_format = (args.output_format or prefs.get('output_format', 'vips')).lower()
    if output_format not in ('png', 'vips', 'tiff'):
        output_format = 'vips'
    ext_by_fmt = {'png': 'png', 'vips': 'v', 'tiff': 'tif'}
    out_ext = ext_by_fmt[output_format]

    tilesets = (args.tilesets or prefs.get('tilesets', 'mapbox.satellite'))
    token = args.mapbox_token or prefs.get('mapbox_token')
    if not token:
        raise RuntimeError('Mapbox token is required. Provide via --token or preferences.')
    pixel_ratio = int(args.pixel_ratio if args.pixel_ratio is not None else prefs.get('pixel_ratio', 1))
    raster_format = (args.raster_format or prefs.get('raster_format', 'png')).lower()
    if raster_format not in ('png', 'png32', 'jpg70', 'jpg80', 'jpg90'):
        raster_format = 'png'

    # Request settings
    max_workers = int(args.max_workers if args.max_workers is not None else prefs.get('max_workers', 24))
    rt = prefs.get('request_timeout', [8.0, 25.0])
    if isinstance(rt, list) and len(rt) == 2:
        request_timeout = (float(rt[0]), float(rt[1]))
    else:
        request_timeout = (8.0, 25.0)

    url_template = build_raster_tiles_url_template(tilesets, tile_size, pixel_ratio, raster_format, token)

    target_countries = get_target_country_iso2_codes(exclude_usa=exclude_usa)
    country_city_map = top_cities_by_country(target_countries, top_n=10)

    for country_code, cities in tqdm(country_city_map.items(), desc='Countries', unit='country'):
        if not cities:
            continue
        country_dir = os.path.join(out_dir, country_code)
        ensure_directory(country_dir)

        for city in tqdm(cities, desc=f'{country_code} cities', leave=False, unit='city'):
            name = sanitize_filename(city['name'])
            lat = city['lat']
            lon = city['lon']

            tl_lat, tl_lon, br_lat, br_lon = compute_bbox_from_center(lat, lon, bbox_km)

            km_str = ('{:.2f}'.format(bbox_km)).rstrip('0').rstrip('.').replace('.', 'p')
            base_prefix = f'{name}_z{zoom}_km{km_str}'
            existing_matches = glob(os.path.join(country_dir, f'{base_prefix}*.{out_ext}'))
            if existing_matches:
                continue

            try:
                img = download_image(tl_lat, tl_lon, br_lat, br_lon, zoom, url_template,
                    prefs['headers'], tile_size, channels, max_workers=max_workers, request_timeout=request_timeout)
            except Exception as e:
                print(f'Failed to download {country_code}/{name} at z{zoom}: {e}')
                continue

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f'{base_prefix}_{timestamp}.{out_ext}'
            out_path = os.path.join(country_dir, filename)
            try:
                save_image_with_format(img, out_path, output_format, channels)
                print(f'Saved {out_path}')
            except Exception as e:
                print(f'Failed to save {out_path}: {e}')


if __name__ == '__main__':
    run()

 