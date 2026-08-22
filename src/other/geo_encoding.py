import requests
import json
from geopy.distance import great_circle
from shapely.geometry import Point, Polygon
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool
from langchain.agents import create_agent
from event_extraction import llm
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform
import pyproj
import geopandas as gpd


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
    """
    Fetches geographic encoding information for a specific feature by calling
    a local geocoding API.
    """
    url = "http://172.27.10.29:8181/search"
    params = {
        "q": query_feature
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"Request failed (status code: {response.status_code})")
            return json.loads(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")


def get_address_lookup_local(osm_id):
    """
    Looks up detailed address geometry and metadata for a given OSM ID
    by communicating with a local API service.
    """
    url = "http://172.27.10.29:8181/lookup"
    params = {
        "osm_ids": osm_id,
        "format": 'json',
        "polygon_geojson": 1
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Request failed (status code: {response.status_code})")
            return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")


def get_geo_reverse_local(lat: float, lon: float):
    """
    Performs a reverse geocoding lookup based on latitude and longitude
    to retrieve spatial location information from a local API.
    """
    url = "http://localhost:8180/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "polygon_geojson": 1
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.text
        else:
            print(f"Request failed (status code: {response.status_code})")
            return response.text

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")


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
    print(f"--- 空间实体信息 ---")
    print(f"R1: {r1_data['name']} (中心点: {r1_data['lon']}, {r1_data['lat']})")
    print(f"R2: {r2_data['name']} (中心点: {r2_data['lon']}, {r2_data['lat']})")

    r1_center = (float(r1_data['lat']), float(r1_data['lon']))
    r2_center = (float(r2_data['lat']), float(r2_data['lon']))

    distance_km = great_circle(r1_center, r2_center).km

    print("\n--- I. 空间距离计算 ---")
    print(f"🌍 大地距离 (km): {distance_km:.2f} km")

    try:
        r1_polygon = get_bounding_box_polygon(r1_data)
        r2_polygon = get_bounding_box_polygon(r2_data)

        intersects = r1_polygon.intersects(r2_polygon)
        contains = r1_polygon.contains(r2_polygon) or r2_polygon.contains(r1_polygon)
        touches = r1_polygon.touches(r2_polygon)

        print("\n--- II. 空间拓扑关系计算 (基于BoundingBox) ---")

        if intersects:
            print("🟢 拓扑关系: 相交 (Intersects)")
        elif contains:
            print("🟢 拓扑关系: 包含/被包含 (Contains/Within)")
        elif touches:
            print("🟡 拓扑关系: 接触/相邻 (Touches)")
        else:
            print("🔴 拓扑关系: 分离 (Disjoint)")

        return [distance_km, [intersects, contains, touches]]

    except ValueError as e:
        print(f"\n❌ 拓扑计算失败: {e}. 请检查 boundingbox 数据格式。")


project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True).transform


def get_metric_geometry(geom):
    """
    Transforms a geographic coordinate system geometry into a projected
    metric coordinate system geometry.
    """
    return transform(project, geom)


def get_entity_combined_geometry(osm_ids_list):
    """
    Retrieves GeoJSON representations for a list of OSM IDs and merges them
    into a single unified Shapely geometry object.
    """
    raw_data = get_address_lookup_local(osm_ids_list)

    geoms = []
    for item in raw_data:
        if 'geojson' in item:
            geoms.append(shape(item['geojson']))

    if not geoms:
        return None
    combined_geom = unary_union(geoms)
    return combined_geom


def get_downstream_path(start_id, adjacency_map, max_depth=500):
    """
    Traces and returns all downstream basin IDs in a basin adjacency map
    starting from a specified starting node.
    """
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
        return {"overlap": "流域重叠", "up_down_stream": ""}
    if shape_b.contains(shape_a):
        return {"overlap": "流域重叠", "up_down_stream": ""}

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

    overlap = "流域重叠" if not basins_a.isdisjoint(basins_b) else "流域不重叠"

    if is_up or is_up:
        up_down_stream = '上下游'
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
    geom_b_wgs84 = get_entity_combined_geometry(entity_b_ids)

    if not geom_a_wgs84 or not geom_b_wgs84:
        return {}

    hybas_lv12 = gpd.read_file("data/basin/henan_basin-1.shp")
    basin = analyze_basin_relation(geom_a_wgs84, geom_b_wgs84, hybas_lv12)

    relations = {
        "相交": geom_a_wgs84.intersects(geom_b_wgs84),
        "相接": geom_a_wgs84.touches(geom_b_wgs84),
        "包含": geom_a_wgs84.contains(geom_b_wgs84),
        "被包含": geom_a_wgs84.within(geom_b_wgs84),
        "相离": geom_a_wgs84.disjoint(geom_b_wgs84),
        "重叠": geom_a_wgs84.overlaps(geom_b_wgs84)
    }

    geom_a_metric = get_metric_geometry(geom_a_wgs84)
    geom_b_metric = get_metric_geometry(geom_b_wgs84)

    distance_meters = geom_a_metric.distance(geom_b_metric)

    return {
        "topological_relations": relations,
        "distance_meters": round(distance_meters, 2),
        "basin_relations": basin
    }


def get_topological_relation(relations: dict):
    """
    Translates a dictionary of topological boolean results into a single
    descriptive string relationship label.
    """
    is_intersect = relations.get("相交", False)
    contains = relations.get("包含", False)
    within = relations.get("被包含", False)
    touches = relations.get("相接", False)
    overlaps = relations.get("重叠", False)

    if not is_intersect and not touches:
        return "相离"

    if contains and within:
        return "相等"
    if within:
        return "被包含"
    if contains:
        return "包含"
    if overlaps:
        return "重叠"
    if touches:
        return "相邻"

    return "相交"


if __name__ == '__main__':
    print(get_geo_encoding_local('坂东镇'))