import requests
import json
import re
import time
from geopy.distance import great_circle
from shapely.geometry import Polygon
from pydantic import BaseModel, Field
from langchain.tools import tool
from shapely.geometry import shape
from shapely.errors import ShapelyError
from shapely.ops import unary_union, transform
import pyproj
import geopandas as gpd
from functools import lru_cache
from src.construct_config import get_configured_path, get_setting


class GeonameInput(BaseModel):
    """Input for geoname queries."""
    geo_name: str = Field(description="指向一个现实地理位置或范围的地理对象名称")
    region: str = Field(description="在源文本中提及的该地理对象所在的行政区范围，范围等级最细到区县级")
    geo_type: str = Field(description="该地理对象的类型")


class GeonameInput1(BaseModel):
    """Input for geoname queries."""
    geo_name: str = Field(description="指向一个现实地理位置或范围的地理对象名称")


@tool("get_location", args_schema=GeonameInput1, description="获取具有现实地理位置或范围指向地理对象名称的具体地理坐标")
def get_location(geo_name: str):
    """
    Tool function to get the specific geographic coordinates associated with a
    real geographic location or range pointed to by the name of the geographic object.
    """
    result = get_geo_encoding_local(geo_name)
    if len(result) == 0:
        result = '找不到位置'
    return result


def get_geo_encoding_local(query_feature: str):
    """Return search results as a list, including an empty list when the request fails."""
    data = _request_geocoding_json(get_setting('geocoding.search_url'),
                                   {'q': query_feature, 'format': 'json'})
    if data is not None and not isinstance(data, list):
        print('[Geocoding] Search returned an unexpected response type; no results accepted.')
    return data if isinstance(data, list) else []


def _request_geocoding_json(url, params):
    """Request JSON with finite timeouts and retry only temporary HTTP or transport failures."""
    settings = get_setting('geocoding').get('http', {})
    timeout = (float(settings.get('connect_timeout_seconds', 5)),
               float(settings.get('read_timeout_seconds', 30)))
    attempts = int(settings.get('max_attempts', 3))
    delay = float(settings.get('retry_delay_seconds', 1))
    if min(timeout) <= 0 or attempts < 1 or delay < 0:
        raise ValueError('Geocoding timeouts and max_attempts must be positive; retry delay must be nonnegative.')
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            transient = (isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
                         or status == 429 or (status is not None and status >= 500))
            if transient and attempt < attempts:
                print(f'[Geocoding] Temporary request failure at {url}; '
                      f'retrying {attempt + 1}/{attempts}: {error}')
                time.sleep(delay * attempt)
                continue
            print(f'[Geocoding] Request failed at {url} after {attempt} attempt(s): {error}')
            return None
        try:
            data = response.json()
        except ValueError as error:
            print(f'[Geocoding] Invalid JSON response from {url}: {error}')
            return None
        if isinstance(data, dict) and 'error' in data:
            print(f"[Geocoding] API error from {url}: {data['error']}")
            return None
        return data


def normalize_osm_ids(value):
    """Normalize typed OSM IDs and reject empty or malformed identifiers."""
    if isinstance(value, str):
        tokens = re.split(r'[,;\s-]+', value.strip())
    elif isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value]
    else:
        return ''
    if not tokens or any(not re.fullmatch(r'[NWR][1-9][0-9]*', token, re.IGNORECASE) for token in tokens):
        return ''
    return ','.join(dict.fromkeys(token.upper() for token in tokens))


def get_address_lookup_local(osm_id):
    """Look up typed OSM IDs in bounded batches and return a list even on failure."""
    normalized = normalize_osm_ids(osm_id)
    if not normalized:
        print('[Geocoding] Lookup skipped: no valid typed OSM IDs were provided.')
        return []
    url = get_setting('geocoding.lookup_url')
    batch_size = int(get_setting('geocoding').get('http', {}).get('lookup_batch_size', 50))
    if not 1 <= batch_size <= 50:
        raise ValueError('Geocoding lookup_batch_size must be between 1 and 50.')
    identifiers = normalized.split(',')
    results = []
    for start in range(0, len(identifiers), batch_size):
        data = _request_geocoding_json(url, {
            'osm_ids': ','.join(identifiers[start:start + batch_size]),
            'format': 'json', 'polygon_geojson': 1,
        })
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            print('[Geocoding] Lookup returned no usable result list; partial batches discarded.')
            return []
        results.extend(data)
    return results


def get_geo_reverse_local(lat: float, lon: float):
    """Return reverse-geocoding JSON text, or an empty string when the request fails."""
    data = _request_geocoding_json(get_setting('geocoding.reverse_url'),
                                   {'lat': lat, 'lon': lon, 'format': 'json', 'polygon_geojson': 1})
    return json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else ''


def get_bounding_box_polygon(data: dict) -> Polygon:
    """
    Extracts bounding box coordinates from an API dictionary payload
    and converts them into a Shapely Polygon representation.
    """
    bbox_str = data.get('boundingbox')
    if not bbox_str or len(bbox_str) != 4:
        raise ValueError("Bounding box data is invalid.")

    min_lat, max_lat, min_lon, max_lon = map(float, bbox_str)

    polygon = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat)
    ])
    return polygon


def calculate_spatial_relations(r1_data: dict, r2_data: dict):
    """
    Calculates spatial distance and topological relations between two geographic entities
    using their coordinate and bounding box data.
    """
    print(f"--- Spatial entity information ---")
    print(f"R1: {r1_data['name']} (center: {r1_data['lon']}, {r1_data['lat']})")
    print(f"R2: {r2_data['name']} (center: {r2_data['lon']}, {r2_data['lat']})")

    r1_center = (float(r1_data['lat']), float(r1_data['lon']))
    r2_center = (float(r2_data['lat']), float(r2_data['lon']))

    distance_km = great_circle(r1_center, r2_center).km

    print("\n--- I. Spatial distance calculation ---")
    print(f"🌍 Geodesic distance (km): {distance_km:.2f} km")

    try:
        r1_polygon = get_bounding_box_polygon(r1_data)
        r2_polygon = get_bounding_box_polygon(r2_data)

        intersects = r1_polygon.intersects(r2_polygon)
        contains = r1_polygon.contains(r2_polygon) or r2_polygon.contains(r1_polygon)
        touches = r1_polygon.touches(r2_polygon)

        print("\n--- II. Spatial topology (bounding boxes) ---")

        if intersects:
            print("🟢 Topology: Intersects")
        elif contains:
            print("🟢 Topology: Contains/Within")
        elif touches:
            print("🟡 Topology: Touches")
        else:
            print("🔴 Topology: Disjoint")

        return [distance_km, [intersects, contains, touches]]
    except ValueError as e:
        print(f"\n❌ Topology calculation failed: {e}. Check the boundingbox data format.")


@lru_cache(maxsize=None)
def _get_projection(source_crs, metric_crs):
    """Cache coordinate transformers for configured source and metric reference systems."""
    return pyproj.Transformer.from_crs(source_crs, metric_crs, always_xy=True).transform


def get_metric_geometry(geom):
    """
    Transforms a geographic coordinate system geometry into a projected
    metric coordinate system geometry.
    """
    project = _get_projection(get_setting('geocoding.source_crs'), get_setting('geocoding.metric_crs'))
    return transform(project, geom)


def get_entity_combined_geometry(osm_ids_list):
    """Merge complete lookup geometries; reject failed, missing, or malformed components."""
    identifiers = normalize_osm_ids(osm_ids_list)
    if not identifiers:
        print('[Geocoding] Geometry skipped: invalid OSM IDs.')
        return None
    raw_data = get_address_lookup_local(identifiers)
    if not isinstance(raw_data, list) or not raw_data:
        print(f'[Geocoding] No lookup geometries available for {identifiers}.')
        return None
    expected = set(identifiers.split(','))
    geometries = {}
    prefixes = {'node': 'N', 'way': 'W', 'relation': 'R'}
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        identifier = prefixes.get(str(item.get('osm_type', '')).lower(), '') + str(item.get('osm_id', ''))
        if identifier not in expected:
            continue
        try:
            geometry = shape(item['geojson'])
            if geometry.is_empty or not geometry.is_valid:
                raise ValueError('Geometry is empty or invalid.')
            geometries[identifier] = geometry
        except (ShapelyError, ValueError, TypeError, KeyError, AttributeError) as error:
            print(f'[Geocoding] Invalid geometry for {identifier}: {error}')
    missing = expected - geometries.keys()
    if missing:
        print(f"[Geocoding] Incomplete geometry lookup; missing: {', '.join(sorted(missing))}.")
        return None
    try:
        return unary_union(list(geometries.values()))
    except ShapelyError as error:
        print(f'[Geocoding] Geometry union failed for {identifiers}: {error}')
        return None


def get_downstream_path(start_id, adjacency_map, max_depth=None):
    """
    Traces and returns all downstream basin IDs in a basin adjacency map
    starting from a specified starting node.
    """
    if max_depth is None:
        max_depth = get_setting('geocoding.downstream_max_depth')
    path = set()
    current = start_id
    while current != 0 and len(path) < max_depth:
        path.add(current)
        current = adjacency_map.get(current, 0)
    return path


def analyze_basin_relation(shape_a, shape_b, hybas_gdf):
    """
    Analyzes the overlap and up/downstream relations between two geometric shapes
    based on hydrological basin networking properties.
    """

    if shape_a.contains(shape_b):
        return {"overlap": "Basin Overlap", "up_down_stream": ""}
    if shape_b.contains(shape_a):
        return {"overlap": "Basin Overlap", "up_down_stream": ""}

    res_a = hybas_gdf[hybas_gdf.intersects(shape_a)]
    res_b = hybas_gdf[hybas_gdf.intersects(shape_b)]

    basins_a = set(res_a['HYBAS_ID'])
    basins_b = set(res_b['HYBAS_ID'])

    if not basins_a or not basins_b:
        return {"overlap": "", "up_down_stream": ""}

    adj_map = dict(zip(hybas_gdf['HYBAS_ID'], hybas_gdf['NEXT_DOWN']))

    exclusive_a = basins_a - basins_b
    exclusive_b = basins_b - basins_a

    is_up = False
    is_down = False

    for id_a in exclusive_a:
        path = get_downstream_path(id_a, adj_map)
        if not path.isdisjoint(basins_b):
            is_up = True
            break

    for id_b in exclusive_b:
        path = get_downstream_path(id_b, adj_map)
        if not path.isdisjoint(basins_a):
            is_down = True
            break

    overlap = "Basin Overlap" if not basins_a.isdisjoint(basins_b) else "No Basin Overlap"

    if is_up or is_up:
        up_down_stream = 'Upstream-Downstream'
    else:
        up_down_stream = ''

    return {
        "overlap": overlap,
        "up_down_stream": up_down_stream
    }


def analyze_spatial_relation(entity_a_ids, entity_b_ids):
    """
    Computes comprehensive spatial relations (topological, distance, and basin relations)
    between two geographic entities based on their OSM IDs.
    """
    geom_a_wgs84 = get_entity_combined_geometry(entity_a_ids)
    if geom_a_wgs84 is None:
        print(f'[Spatial] Pair skipped because lookup geometry is unavailable: {entity_a_ids} -> {entity_b_ids}')
        return {}
    geom_b_wgs84 = get_entity_combined_geometry(entity_b_ids)
    if geom_b_wgs84 is None:
        print(f'[Spatial] Pair skipped because lookup geometry is unavailable: {entity_a_ids} -> {entity_b_ids}')
        return {}

    hybas_lv12 = gpd.read_file(get_configured_path('geocoding.basin_file'))
    basin = analyze_basin_relation(geom_a_wgs84, geom_b_wgs84, hybas_lv12)

    relations = {
        "Intersect": geom_a_wgs84.intersects(geom_b_wgs84),
        "Touches": geom_a_wgs84.touches(geom_b_wgs84),
        "Contains": geom_a_wgs84.contains(geom_b_wgs84),
        "Contained By": geom_a_wgs84.within(geom_b_wgs84),
        "Disjoint": geom_a_wgs84.disjoint(geom_b_wgs84),
        "Overlap": geom_a_wgs84.overlaps(geom_b_wgs84)
    }

    geom_a_metric = get_metric_geometry(geom_a_wgs84)
    geom_b_metric = get_metric_geometry(geom_b_wgs84)

    distance_meters = geom_a_metric.distance(geom_b_metric)

    return {
        "topological_relations": relations,
        "distance_meters": round(distance_meters, get_setting('geocoding.distance_precision')),
        "basin_relations": basin
    }


def get_topological_relation(relations: dict):
    """
    Translates a dictionary of topological boolean results into a single
    descriptive string relationship label.
    """
    is_intersect = relations.get("Intersect", False)
    contains = relations.get("Contains", False)
    within = relations.get("Contained By", False)
    touches = relations.get("Touches", False)
    overlaps = relations.get("Overlap", False)

    if not is_intersect and not touches:
        return "Disjoint"

    if contains and within:
        return "Equal"
    if within:
        return "Contained By"
    if contains:
        return "Contains"
    if overlaps:
        return "Overlap"
    if touches:
        return "Adjacent"

    return "Intersect"


if __name__ == '__main__':
    print(get_geo_encoding_local('坂东镇'))
