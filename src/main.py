import os
import json
import cv2
import math
import argparse
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm
from glob import glob

from image_downloading import download_image
import pyvips

file_dir = os.path.dirname(__file__)
prefs_path = os.path.join(file_dir, 'preferences.json')
default_prefs = {
        'url': 'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        'tile_size': 256,
        'channels': 3,
        'dir': os.path.join(file_dir, 'images'),
        'output_format': 'vips',  # png | vips | tiff
        'headers': {
            'cache-control': 'max-age=0',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'
        },
        'zoom': 17,
        'bbox_km': 20,
        'regions': ['Europe', 'Americas'],
        'exclude_usa': False
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download satellite images for top-10 cities per country in Europe and the Americas.')
    parser.add_argument('--exclude-usa', action='store_true', help='Exclude USA from the downloading process.')
    parser.add_argument('--out-dir', type=str, help='Output directory to save images (overrides preferences).')
    parser.add_argument('--format', dest='output_format', choices=['png', 'vips', 'tiff'], help='Output format: png, vips (.v), or tiled tiff (.tif).')
    return parser.parse_args()


def ensure_directory(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    return ''.join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')


def compute_bbox_from_center(lat: float, lon: float, window_km: float) -> Tuple[float, float, float, float]:
    # Interpret window_km as half-side distance around the center (Δm)
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
    # Convert OpenCV BGR/BGRA to RGB/RGBA as pyvips expects
    if channels == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        vimage = pyvips.Image.new_from_memory(img_rgb.tobytes(), width, height, 3, 'uchar')
    else:
        # Assume BGRA -> RGBA
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        height, width = img_rgba.shape[:2]
        vimage = pyvips.Image.new_from_memory(img_rgba.tobytes(), width, height, 4, 'uchar')

    if fmt == 'vips':
        # .v format, random-access friendly
        vimage.write_to_file(out_path)
        return
    if fmt == 'tiff':
        # Tiled, BigTIFF, with LZW compression for balance; pyramids optional
        # Use tile size 256 for compatibility with our input tiles
        vimage.tiffsave(out_path, tile=True, tile_width=256, tile_height=256,
                        bigtiff=True, compression='lzw', pyramid=False)
        return
    # Fallback to PNG
    cv2.imwrite(out_path, img)


def get_target_country_iso2_codes(exclude_usa: bool) -> List[str]:
    import geonamescache

    gc = geonamescache.GeonamesCache()
    countries: Dict[str, Dict] = gc.get_countries()
    target_iso2: List[str] = []

    for _, country in countries.items():
        continent_code = country.get('continentcode')
        # Americas: NA + SA; Europe: EU
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


def run():
    with open(os.path.join(file_dir, 'preferences.json'), 'r', encoding='utf-8') as f:
        prefs = json.loads(f.read())

    args = parse_args()

    out_dir = args.out_dir if getattr(args, 'out_dir', None) else prefs['dir']
    ensure_directory(out_dir)

    zoom = int(prefs.get('zoom', 17))
    channels = int(prefs.get('channels', 3))
    tile_size = int(prefs.get('tile_size', 256))
    bbox_km = float(prefs.get('bbox_km', 20))

    exclude_usa = bool(prefs.get('exclude_usa', False) or args.exclude_usa)
    output_format = (args.output_format or prefs.get('output_format', 'png')).lower()
    if output_format not in ('png', 'vips', 'tiff'):
        output_format = 'png'
    ext_by_fmt = {'png': 'png', 'vips': 'v', 'tiff': 'tif'}
    out_ext = ext_by_fmt[output_format]

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

            # Build deterministic prefix for this city/zoom/window and skip if already downloaded
            km_str = ('{:.2f}'.format(bbox_km)).rstrip('0').rstrip('.').replace('.', 'p')
            base_prefix = f'{name}_z{zoom}_km{km_str}'
            existing_matches = glob(os.path.join(country_dir, f'{base_prefix}*.{out_ext}'))
            if existing_matches:
                # Already downloaded for this config
                continue

            try:
                img = download_image(tl_lat, tl_lon, br_lat, br_lon, zoom, prefs['url'],
                    prefs['headers'], tile_size, channels)
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


if os.path.isfile(prefs_path):
    run()
else:
    with open(prefs_path, 'w', encoding='utf-8') as f:
        json.dump(default_prefs, f, indent=2, ensure_ascii=False)

    print(f'Preferences file created in {prefs_path}')
