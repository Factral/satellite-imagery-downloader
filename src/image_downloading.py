import cv2
import requests
import numpy as np
import threading
import time
import random
from urllib.parse import urlparse, urlunparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple


def _build_rotated_host_url(original_url: str, attempt_index: int) -> str:
    parsed = urlparse(original_url)
    host = parsed.hostname or ''
    # Rotate mt subdomains if applicable
    if host.endswith('google.com') and host.startswith('mt'):
        # Replace any mt*, or mt with mt{n}
        subdomain_index = attempt_index % 4
        new_host = f'mt{subdomain_index}.google.com'
        parsed = parsed._replace(netloc=new_host)
        return urlunparse(parsed)
    if host == 'mt.google.com':
        subdomain_index = attempt_index % 4
        new_host = f'mt{subdomain_index}.google.com'
        parsed = parsed._replace(netloc=new_host)
        return urlunparse(parsed)
    return original_url


def _create_session(max_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=64)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def download_tile(url: str, headers: dict, channels: int, session: requests.Session,
                  timeout: Tuple[float, float] = (5.0, 15.0), attempts: int = 4) -> Optional[np.ndarray]:
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        rotated_url = _build_rotated_host_url(url, attempt)
        try:
            response = session.get(rotated_url, headers=headers, timeout=timeout)
            if response.status_code != 200:
                last_exc = Exception(f'HTTP {response.status_code}')
                # brief pause before retry on non-200
                time.sleep(min(1.0, 0.25 * (attempt + 1)))
                continue
            arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            decoded = cv2.imdecode(arr, 1 if channels == 3 else -1)
            if decoded is None:
                last_exc = Exception('Failed to decode image')
                time.sleep(min(1.0, 0.25 * (attempt + 1)))
                continue
            return decoded
        except requests.exceptions.SSLError as e:
            last_exc = e
            time.sleep(0.5 + 0.25 * attempt)
            continue
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            last_exc = e
            time.sleep(0.5 + 0.25 * attempt)
            continue
        except Exception as e:
            last_exc = e
            time.sleep(0.25)
            continue
    # All attempts failed
    return None


# Mercator projection 
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def download_image(lat1: float, lon1: float, lat2: float, lon2: float,
    zoom: int, url: str, headers: dict, tile_size: int = 256, channels: int = 3,
    max_workers: int = 16, request_timeout: Tuple[float, float] = (5.0, 15.0)) -> np.ndarray:
    """
    Downloads a map region. Returns an image stored as a `numpy.ndarray` in BGR or BGRA, depending on the number
    of `channels`.

    Parameters
    ----------
    `(lat1, lon1)` - Coordinates (decimal degrees) of the top-left corner of a rectangular area

    `(lat2, lon2)` - Coordinates (decimal degrees) of the bottom-right corner of a rectangular area

    `zoom` - Zoom level

    `url` - Tile URL with {x}, {y} and {z} in place of its coordinate and zoom values

    `headers` - Dictionary of HTTP headers

    `tile_size` - Tile size in pixels

    `channels` - Number of channels in the output image. Also affects how the tiles are converted into numpy arrays.

    `max_workers` - Maximum number of threads used to download tile rows concurrently.

    `request_timeout` - (connect_timeout, read_timeout) for HTTP requests.
    """

    scale = 1 << zoom

    # Find the pixel coordinates and tile coordinates of the corners
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)


    session = _create_session()

    def build_row(tile_y):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tile_url = url.format(x=tile_x, y=tile_y, z=zoom)
            tile = download_tile(tile_url, headers, channels, session, timeout=request_timeout)

            if tile is not None:
                # Find the pixel coordinates of the new tile relative to the image
                tl_rel_x = tile_x * tile_size - tl_pixel_x
                tl_rel_y = tile_y * tile_size - tl_pixel_y
                br_rel_x = tl_rel_x + tile_size
                br_rel_y = tl_rel_y + tile_size

                # Define where the tile will be placed on the image
                img_x_l = max(0, tl_rel_x)
                img_x_r = min(img_w + 1, br_rel_x)
                img_y_l = max(0, tl_rel_y)
                img_y_r = min(img_h + 1, br_rel_y)

                # Define how border tiles will be cropped
                cr_x_l = max(0, -tl_rel_x)
                cr_x_r = tile_size + min(0, img_w - br_rel_x)
                cr_y_l = max(0, -tl_rel_y)
                cr_y_r = tile_size + min(0, img_h - br_rel_y)

                img[img_y_l:img_y_r, img_x_l:img_x_r] = tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]


    # Limit concurrency across rows to reduce SSL and provider throttling issues
    tile_rows = list(range(tl_tile_y, br_tile_y + 1))
    if max_workers < 1:
        max_workers = 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(build_row, tile_rows))
    
    return img


def image_size(lat1: float, lon1: float, lat2: float,
    lon2: float, zoom: int, tile_size: int = 256):
    """ Calculates the size of an image without downloading it. Returns the width and height in pixels as a tuple. """

    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    return abs(tl_pixel_x - br_pixel_x), br_pixel_y - tl_pixel_y
