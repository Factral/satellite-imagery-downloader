import os
import json
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import cv2


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
    parser = argparse.ArgumentParser(
        description='Create a PNG plotting one point per target city on a Mercator-projected world.'
    )
    parser.add_argument('--exclude-usa', action='store_true', help='Exclude USA from the city set.')
    parser.add_argument('--top-n', type=int, default=10, help='Top-N cities per country by population (default: 10).')
    parser.add_argument('--out', type=str, default=os.path.join(file_dir, 'cities_mercator.png'),
                        help='Output PNG path (default: ./cities_mercator.png).')
    parser.add_argument('--width', type=int, default=4096, help='Output image width in pixels (default: 4096).')
    parser.add_argument('--height', type=int, default=2048, help='Output image height in pixels (default: 2048).')
    parser.add_argument('--radius', type=int, default=8, help='City point radius in pixels (default: 8).')
    parser.add_argument('--graticule', action='store_true', help='Draw light graticule lines for context.')
    parser.add_argument('--background', choices=['none', 'cartopy', 'image'], default='cartopy',
                        help='Background type: none, cartopy (world land/water), or image (provide --background-image). Default: cartopy')
    parser.add_argument('--background-image', type=str, default='',
                        help='Path to a local background image already in Web Mercator. Used with --background image.')
    return parser.parse_args()


def ensure_prefs_file() -> Dict:
    if os.path.isfile(prefs_path):
        with open(prefs_path, 'r', encoding='utf-8') as f:
            return json.loads(f.read())
    with open(prefs_path, 'w', encoding='utf-8') as f:
        json.dump(default_prefs, f, indent=2, ensure_ascii=False)
    return default_prefs


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


WEB_MERCATOR_LAT_LIMIT = 85.05112878


def lonlat_to_webmercator_xy(lon: float, lat: float, width: int, height: int) -> Tuple[int, int]:
    lon = max(-180.0, min(180.0, lon))
    lat = max(-WEB_MERCATOR_LAT_LIMIT, min(WEB_MERCATOR_LAT_LIMIT, lat))

    x = (lon + 180.0) / 360.0
    siny = math.sin(math.radians(lat))
    y = 0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)

    px = int(round(x * (width - 1)))
    py = int(round(y * (height - 1)))
    return px, py


def draw_graticule(img: np.ndarray, width: int, height: int) -> None:
    line_color = (220, 220, 220)  # light gray in BGR
    thickness = 1

    # Longitude lines every 30 degrees
    for lon in range(-180, 181, 30):
        x, _ = lonlat_to_webmercator_xy(float(lon), 0.0, width, height)
        cv2.line(img, (x, 0), (x, height - 1), line_color, thickness, lineType=cv2.LINE_AA)

    # Latitude lines every 20 degrees (clamped to Mercator limits)
    for lat in range(-80, 81, 20):
        _, y = lonlat_to_webmercator_xy(0.0, float(lat), width, height)
        cv2.line(img, (0, y), (width - 1, y), line_color, thickness, lineType=cv2.LINE_AA)


def build_png(city_index: Dict[str, List[Dict]], out_path: str, width: int, height: int, radius: int, draw_grid: bool) -> None:
    background_color = (255, 255, 255)
    point_color = (31, 119, 180)  # matplotlib tab:blue in BGR approx

    img = np.full((height, width, 3), background_color, dtype=np.uint8)

    if draw_grid:
        draw_graticule(img, width, height)

    for country_code, cities in city_index.items():
        for c in cities:
            px, py = lonlat_to_webmercator_xy(c['lon'], c['lat'], width, height)
            cv2.circle(img, (px, py), radius, point_color, thickness=-1, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, img)


def build_png_with_background_image(city_index: Dict[str, List[Dict]], out_path: str, width: int, height: int, radius: int,
                                    draw_grid: bool, background_image_path: str) -> None:
    if not background_image_path or not os.path.isfile(background_image_path):
        raise FileNotFoundError(f'Background image not found: {background_image_path}')

    base = cv2.imread(background_image_path, cv2.IMREAD_COLOR)
    if base is None:
        raise RuntimeError(f'Failed to read background image: {background_image_path}')
    base = cv2.resize(base, (width, height), interpolation=cv2.INTER_AREA)

    if draw_grid:
        draw_graticule(base, width, height)

    point_color = (31, 119, 180)
    for country_code, cities in city_index.items():
        for c in cities:
            px, py = lonlat_to_webmercator_xy(c['lon'], c['lat'], width, height)
            cv2.circle(base, (px, py), radius, point_color, thickness=-1, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, base)


def build_png_with_cartopy(city_index: Dict[str, List[Dict]], out_path: str, width: int, height: int, radius: int, draw_grid: bool) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as e:
        raise RuntimeError('Cartopy and Matplotlib are required for --background cartopy. Install with: pip install cartopy matplotlib') from e

    dpi = 100
    fig_w_in = max(1.0, width / dpi)
    fig_h_in = max(1.0, height / dpi)

    proj = ccrs.Mercator()
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = plt.axes(projection=proj)
    ax.set_global()

    ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff')
    ax.add_feature(cfeature.LAND, facecolor='#f7f7f7')
    ax.add_feature(cfeature.COASTLINE, edgecolor='#7f7f7f', linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, edgecolor='#aaaaaa', linewidth=0.3)

    if draw_grid:
        gl = ax.gridlines(draw_labels=False, color='#dddddd', linewidth=0.5)

    lons: List[float] = []
    lats: List[float] = []
    for country_code, cities in city_index.items():
        for c in cities:
            lon = float(max(-180.0, min(180.0, c['lon'])))
            lat = float(max(-WEB_MERCATOR_LAT_LIMIT, min(WEB_MERCATOR_LAT_LIMIT, c['lat'])))
            lons.append(lon)
            lats.append(lat)

    p_points = (radius * 72.0 / dpi)
    size_pts2 = max(1.0, p_points * p_points)
    ax.scatter(lons, lats, s=size_pts2, color='#1f77b4', alpha=0.9, transform=ccrs.PlateCarree())

    plt.axis('off')
    fig.tight_layout(pad=0)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def run() -> None:
    prefs = ensure_prefs_file()
    args = parse_args()

    exclude_usa = bool(prefs.get('exclude_usa', False) or args.exclude_usa)
    top_n = int(args.top_n)
    out_png = args.out
    width = int(args.width)
    height = int(args.height)
    radius = int(args.radius)
    draw_grid = bool(args.graticule)
    background_mode = args.background
    background_image_path = args.background_image

    target_countries = get_target_country_iso2_codes(exclude_usa=exclude_usa)
    country_city_map = top_cities_by_country(target_countries, top_n=top_n)

    if background_mode == 'image':
        build_png_with_background_image(country_city_map, out_png, width, height, radius, draw_grid, background_image_path)
    elif background_mode == 'cartopy':
        try:
            build_png_with_cartopy(country_city_map, out_png, width, height, radius, draw_grid)
        except Exception as e:
            print(f'Cartopy background failed ({e}). Falling back to plain background.')
            build_png(country_city_map, out_png, width, height, radius, draw_grid)
    else:
        build_png(country_city_map, out_png, width, height, radius, draw_grid)
    print(f'PNG saved to {out_png}')


if __name__ == '__main__':
    run()


