import os
import re
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry import shape, Polygon, MultiPolygon, Point
from shapely import wkt

from ontology.neo4jmodel import DisasterEvent, OSMFeature
from instantiation import get_cached_geojson, get_factor_to_disaster, \
    ins_graph_database

warnings.filterwarnings('ignore')

observation_importance = {
    "River": 1.5,
    "Lake": 1.0,
    "Basin": 1.0,
    "Islet": 1.0,
    "Shoal": 1.0,
    "Spring": 1.0,
    "Waterfall": 1.0,
    "Mountain": 1.0,
    "Cave": 1.0,
    "Sinkhole": 1.0,
    "Earth Bank": 1.0,
    "Valley": 1.0,
    "Sea": 1.0,
    "Coastline": 1.0,
    "Island": 1.0,
    "Farmland": 1.0,
    "Forest": 1.0,
    "Meadow": 1.0,
    "Sand": 1.0,
    "Wetland": 1.0,
    "Ditch": 1.0,
    "Reservoir": 2.5,
    "Pond": 1.0,
    "Water Well": 1.0,
    "Lock Gate": 1.0,
    "Dam": 1.0,
    "Dyke": 1.0,
    "Pumping Station": 1.0,
    "Flood Storage Area": 1.0,
    "Highway": 1.0,
    "Railway": 1.0,
    "Subway": 1.0,
    "Waterway": 1.0,
    "Bridge": 2.0,
    "Tunnel": 3.0,
    "Aerialway": 0.4,
    "Underpass": 2.0,
    "Traffic Intersection": 1.0,
    "Train Station": 1.0,
    "Metro Station": 3.0,
    "Tram Station": 1.0,
    "Dock": 1.0,
    "Port": 1.0,
    "Lighthouse": 1.0,
    "Aerodrome": 1.0,
    "Apron": 1.0,
    "Underground Parking": 3.0,
    "Ground Parking": 1.0,
    "In-Building Parking": 1.0,
    "Bus Station": 1.0,
    "Charging Station": 1.0,
    "Building": 1.0,
    "Underground Building": 1.0,
    "Industrial Structure": 1.0,
    "Public Service Structure": 1.0,
    "Observation Structure": 1.0,
    "Enclosure Structure": 3.0,
    "Cable": 1.0,
    "Tower": 1.0,
    "Utility Pole": 1.0,
    "Substation": 1.0,
    "Telecom Line": 1.0,
    "Telecom Base Station": 1.0,
    "Communications Tower": 1.0,
    "Pipeline": 1.0,
    "Storage Tank": 1.0,
    "Water Tower": 1.0,
    "Sewer": 1.0,
    "Sewer Vent": 1.0,
    "Residential Area": 1.0,
    "Public Management and Service Area": 1.0,
    "Commercial Area": 1.0,
    "Mining Area": 1.0,
    "Warehouse": 1.0,
    "Greenery": 1.0,
    "Farmyard": 1.0,
    "Infrastructure Area": 1.0,
    "Administrative Region": 1.0
}
importance_percentile = 0


def calculate_feature_importance_in_same_disaster(is_new: bool = False):
    """Calculate feature importance for instantiated disaster events."""
    has_calculate_disaster = []
    disasters = DisasterEvent.nodes.all()
    features = {}
    for disaster in disasters:
        geo_name = disaster.geo_argument.all()[0].geo_object_name
        features_geo = features.get(geo_name, {})

        event_des = disaster.event_description
        if event_des in has_calculate_disaster:
            continue
        if geo_name == 'Administrative Region':
            feature_nodes = (disaster.geo_argument.all()[0]).has_Feature.all()
            for feature_node in feature_nodes:
                osm_id = feature_node.osm_id
                if len(osm_id) < 20:
                    continue
            continue
        has_calculate_disaster.append(event_des)

        disasters_same = DisasterEvent.nodes.filter(event_description=event_des)
        disaster_prob = {}
        disaster_features = {}
        for d in disasters_same:
            disaster_prob[d.node_id] = d.probability
            geo_node = d.geo_argument.all()[0]
            feature_nodes = geo_node.has_Feature.all()
            for feature_node in feature_nodes:
                osm_id = feature_node.osm_id
                if not is_new:
                    importance = feature_node.importance
                else:
                    importance = feature_node.properties[d.node_id]['importance']
                prob_imp = disaster_features.get(osm_id, [])
                prob_imp.append(d.probability * importance)
                disaster_features[osm_id] = prob_imp

        for key, value in disaster_features.items():
            sum_importance = np.max(np.array(value)) * np.power(1.05, len(value) - 1)

            d_imp = features_geo.get(key, None)
            if not d_imp:
                features_geo[key] = [{event_des: sum_importance}]
            else:
                features_geo[key].append({event_des: sum_importance})
        features[geo_name] = features_geo
    return features


def aggregate_importance_by_factor(factor_to_disaster, features_output):
    """Aggregate event feature importance by observation factor."""
    factor_data_map = {}

    for factor_name, related_events in factor_to_disaster.items():
        if not related_events:
            continue

        if factor_name not in factor_data_map:
            factor_data_map[factor_name] = {}

        for geo_type, id_map in features_output.items():
            records = []

            for osm_id, event_list in id_map.items():
                relevant_scores = []

                for event_dict in event_list:
                    for event_name, score in event_dict.items():
                        if event_name in related_events:
                            val = float(score)
                            if val > 0:
                                relevant_scores.append(val)

                if relevant_scores:
                    max_score = max(relevant_scores)
                    records.append({
                        'osm_id': osm_id,
                        'importance': max_score
                    })

            if records:
                factor_data_map[factor_name][geo_type] = records

    return factor_data_map


def convert_features_to_dataframe(features_dict):
    """Convert nested feature importance records to a dataframe."""
    data_rows = []

    for geo_type, id_map in features_dict.items():
        for osm_id, event_list in id_map.items():
            row_base = {
                'geo_type': geo_type,
                'osm_id': osm_id,
                'total_importance': 0
            }

            for event_dict in event_list:
                for event_name, score in event_dict.items():
                    row_base[event_name] = score
                    row_base['total_importance'] += score

            data_rows.append(row_base)

    df = pd.DataFrame(data_rows)
    df = df.fillna(0)
    return df


def plot_event_specific_importance(features_output, region_coords, ori_coords,
                                   output_folder="./data/output_event_maps_beijing_2"):
    """Plot feature importance for each disaster event."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reg_lons = [c[0] for c in region_coords]
    reg_lats = [c[1] for c in region_coords]
    view_limit = [min(reg_lons), max(reg_lons), min(reg_lats), max(reg_lats)]

    events_map = {}
    for geo_type, id_map in features_output.items():
        for osm_id, event_list in id_map.items():
            for event_dict in event_list:
                for event_name, score in event_dict.items():
                    score_val = float(score)
                    if score_val <= 0: continue
                    if event_name not in events_map:
                        events_map[event_name] = {'geo_type': geo_type, 'records': []}
                    events_map[event_name]['records'].append({'osm_id': osm_id, 'importance': score_val})

    for event_name, event_info in events_map.items():
        geo_type = event_info['geo_type']
        df_event = pd.DataFrame(event_info['records'])
        df_event = df_event.groupby('osm_id', as_index=False).max(numeric_only=True)

        try:
            gdf_base = get_cached_geojson(geo_type)
            if gdf_base is None: continue
            if gdf_base.crs != "EPSG:4326": gdf_base = gdf_base.to_crs(epsg=4326)
        except Exception:
            continue

        merged_gdf = gdf_base.merge(df_event, on='osm_id', how='inner')
        if merged_gdf.empty: continue

        safe_filename = event_name.replace(",", "_").replace(" ", "")
        merged_gdf.to_file(f"{output_folder}/{safe_filename}.geojson", driver='GeoJSON')

        fig, ax = plt.subplots(figsize=(12, 12))

        ori_poly = patches.Polygon(ori_coords, closed=True, linewidth=1.2,
                                   edgecolor='#3498db', facecolor='#ebf5fb',
                                   alpha=0.5, zorder=1)
        ax.add_patch(ori_poly)

        gdf_base.plot(ax=ax, color='#eeeeee', edgecolor='#dcdcdc', linewidth=0.4, zorder=2)

        unique_vals = merged_gdf['importance'].nunique()
        plot_kwargs = {
            "column": 'importance',
            "ax": ax,
            "cmap": 'YlOrRd',
            "legend": True,
            "zorder": 4
        }

        is_point = (merged_gdf.geom_type == 'Point').any()
        if is_point:
            plot_kwargs["markersize"] = 100
            plot_kwargs["edgecolor"] = 'white'
            plot_kwargs["linewidth"] = 0.5
        else:
            plot_kwargs["linewidth"] = 2.0

        merged_gdf['importance'] = merged_gdf['importance'] + np.random.uniform(0, 1e-6, size=len(merged_gdf))
        if unique_vals > 1:
            k_val = min(5, unique_vals)
            plot_kwargs["scheme"] = 'fisher_jenks'
            plot_kwargs["k"] = k_val
            plot_kwargs["legend_kwds"] = {
                'loc': 'lower right',
                'title': 'Importance Level',
                'fontsize': 10
            }
        else:
            plot_kwargs["legend_kwds"] = {'label': 'Importance (Fixed)', 'orientation': "horizontal", 'shrink': 0.4}

        merged_gdf.plot(**plot_kwargs)

        ax.set_xlim(view_limit[0], view_limit[1])
        ax.set_ylim(view_limit[2], view_limit[3])

        ax.set_title(f"Event Specific Observation Importance: {event_name}", fontsize=15, pad=10)

        custom_legends = [
            patches.Patch(facecolor='#ebf5fb', edgecolor='#3498db', label='Original Disaster Zone'),
            Line2D([0], [0], marker='s' if not is_point else 'o', color='w',
                   label='High Importance Asset', markerfacecolor='#e31a1c', markersize=10)
        ]
        ax.legend(handles=custom_legends, loc='upper right', frameon=True)

        ax.set_axis_off()

        plt.savefig(f"{output_folder}/{safe_filename}.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_event_specific_importance_field_basemap(
        features_output,
        region_coords,
        ori_coords,
        output_folder="./data/output_event_maps_grid_beijing_2",
        grid_size=200,
        blur_sigma=5,
        aggregate_by_phenomenon=False,
        bg_image_path='background_map.jpg'
):
    """Plot and export event importance fields over a basemap."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    geotiff_folder = os.path.join(output_folder, "geotiff")
    if not os.path.exists(geotiff_folder):
        os.makedirs(geotiff_folder)

    all_valid_scores = []
    for geo_type, id_map in features_output.items():
        for osm_id, event_list in id_map.items():
            for event_dict in event_list:
                for event_full_name, score in event_dict.items():
                    score_val = float(score)
                    if score_val > 0:
                        all_valid_scores.append(score_val)

    if all_valid_scores:
        calculated_threshold = np.percentile(all_valid_scores, importance_percentile)
    else:
        calculated_threshold = 0.0

    temp_events_map = {}

    for geo_type, id_map in features_output.items():
        for osm_id, event_list in id_map.items():
            for event_dict in event_list:
                for event_full_name, score in event_dict.items():
                    score_val = float(score)
                    if score_val <= calculated_threshold: continue

                    target_key = event_full_name
                    if aggregate_by_phenomenon:
                        parts = re.split(r'[,，]', event_full_name)
                        if len(parts) > 1:
                            target_key = parts[-1].strip()

                    if target_key not in temp_events_map:
                        temp_events_map[target_key] = {}
                    if geo_type not in temp_events_map[target_key]:
                        temp_events_map[target_key][geo_type] = {}

                    current_val = temp_events_map[target_key][geo_type].get(osm_id, 0.0)
                    temp_events_map[target_key][geo_type][osm_id] = current_val + score_val

    events_map = {}
    for ev_name, geo_dict in temp_events_map.items():
        events_map[ev_name] = {}
        for g_type, id_score_map in geo_dict.items():
            record_list = []
            for oid, final_score in id_score_map.items():
                record_list.append({'osm_id': oid, 'importance': final_score})
            events_map[ev_name][g_type] = record_list

    reg_lons = [c[0] for c in region_coords]
    reg_lats = [c[1] for c in region_coords]
    xmin, xmax = min(reg_lons), max(reg_lons)
    ymin, ymax = min(reg_lats), max(reg_lats)

    pixel_width = (xmax - xmin) / grid_size
    pixel_height = (ymax - ymin) / grid_size
    transform = from_origin(xmin, ymax, pixel_width, pixel_height)

    region_poly = Polygon(region_coords)
    region_mask = rasterize(
        [(region_poly, 1)],
        out_shape=(grid_size, grid_size),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    for event_name, geo_data_map in events_map.items():
        importance_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        for geo_type, records in geo_data_map.items():
            try:
                gdf_base = get_cached_geojson(geo_type)
                if gdf_base is None or gdf_base.empty: continue
                if gdf_base.crs != "EPSG:4326":
                    gdf_base = gdf_base.to_crs(epsg=4326)
            except Exception as e:
                continue

            df_event = pd.DataFrame(records)
            merged_gdf = gdf_base.merge(df_event, on='osm_id', how='inner')
            extra_rows = []
            for record in records:
                if len(record['osm_id']) > 20:
                    feature_node = OSMFeature.nodes.first_or_none(osm_id=record['osm_id'])
                    if feature_node:
                        geometry_node = feature_node.has_geometry.all()
                        if geometry_node:
                            geom_val = geometry_node[0].geometry
                            try:
                                if isinstance(geom_val, str):
                                    record['geometry'] = wkt.loads(geom_val)
                                else:
                                    record['geometry'] = geom_val

                                extra_rows.append(record)
                            except Exception as e:
                                pass

            if extra_rows:
                df_extra = gpd.GeoDataFrame(extra_rows, geometry='geometry', crs="EPSG:4326")
                merged_gdf = pd.concat([merged_gdf, df_extra], ignore_index=True)
            if merged_gdf.empty: continue

            for _, row in merged_gdf.iterrows():
                geom = row.geometry
                final_value = row.importance

                if final_value <= 0: continue

                if geom.geom_type in ["Polygon", "MultiPolygon"]:
                    poly_raster = rasterize(
                        [(geom, final_value)],
                        out_shape=(grid_size, grid_size),
                        transform=transform,
                        fill=0,
                        dtype=np.float32
                    )
                    importance_grid = np.maximum(importance_grid, poly_raster)

                else:
                    points = []
                    if geom.geom_type == "Point":
                        points = [geom]
                    elif geom.geom_type == "LineString":
                        step = 2 * pixel_width
                        if geom.length > 0:
                            distances = np.arange(0, geom.length, step)
                            points = [geom.interpolate(d) for d in distances]
                        else:
                            points = [geom.interpolate(0)]

                    for p in points:
                        ix = int((p.x - xmin) / pixel_width)
                        iy = int((ymax - p.y) / pixel_height)

                        if 0 <= ix < grid_size and 0 <= iy < grid_size:
                            importance_grid[iy, ix] = max(
                                float(importance_grid[iy, ix]),
                                final_value
                            )

        safe_filename = event_name.replace(",", "_").replace(" ", "").replace("/", "_")
        tif_path = f"{geotiff_folder}/{safe_filename}_importance.tif"
        raw_png_path = f"{output_folder}/{safe_filename}_RAW_UNBLURRED.png"

        plt.imsave(raw_png_path, importance_grid, cmap="YlOrRd")

        if blur_sigma > 0:
            importance_grid = gaussian_filter(importance_grid, sigma=blur_sigma)

        NODATA_VALUE = -9999.0
        valid_area = (region_mask == 1)

        importance_grid[valid_area] = np.nan_to_num(importance_grid[valid_area], nan=0.0)
        importance_grid[region_mask == 0] = NODATA_VALUE

        with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=grid_size,
                width=grid_size,
                count=1,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
                nodata=NODATA_VALUE
        ) as dst:
            dst.write(importance_grid, 1)

        fig, ax = plt.subplots(figsize=(12, 12))

        try:
            bg_img = mpimg.imread(bg_image_path)
            ax.imshow(bg_img, extent=[xmin, xmax, ymin, ymax], origin='upper', zorder=1)
        except FileNotFoundError:
            pass

        masked_grid = np.ma.masked_where(importance_grid == NODATA_VALUE, importance_grid)

        im = ax.imshow(
            masked_grid,
            extent=[xmin, xmax, ymin, ymax],
            origin="upper",
            cmap="YlOrRd",
            alpha=0.65,
            zorder=2
        )

        valid_values = importance_grid[importance_grid != NODATA_VALUE]
        if len(valid_values) > 0 and np.max(valid_values) > np.min(valid_values):
            threshold = np.min(valid_values) + 0.28 * (np.max(valid_values) - np.min(valid_values))
            if not threshold:
                threshold = 100

            lons = np.linspace(xmin, xmax, grid_size)
            lats = np.linspace(ymax, ymin, grid_size)
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            DEEP_ORANGE = '#E65100'
            SHADOW_COLOR = '#333333'

            dx, dy = (xmax - xmin) * 0.003, (ymax - ymin) * 0.003
            shadow_contour = ax.contour(
                lon_grid + dx, lat_grid - dy,
                importance_grid,
                levels=[threshold],
                colors=[SHADOW_COLOR],
                linewidths=4,
                alpha=0.4,
                linestyles='-',
                zorder=3.5
            )

            main_contour = ax.contour(
                lon_grid, lat_grid,
                importance_grid,
                levels=[threshold],
                colors=[DEEP_ORANGE],
                linewidths=2.5,
                linestyles='-',
                zorder=4
            )

        ori_poly = patches.Polygon(
            ori_coords,
            closed=True,
            edgecolor="#3498db",
            facecolor="none",
            linewidth=2.0,
            linestyle='--',
            zorder=3,
            label='Initial Disaster Zone'
        )
        ax.add_patch(ori_poly)

        reg_poly = patches.Polygon(
            region_coords,
            closed=True,
            edgecolor="black",
            facecolor="none",
            linewidth=1.5,
            zorder=3
        )
        ax.add_patch(reg_poly)

        ax.set_axis_off()

        png_path = f"{output_folder}/{safe_filename}_field.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_factor_specific_importance(factor_data_map, factor_to_disaster_map, region_coords, ori_coords,
                                    output_folder="./data/output_factor_maps_beijing_2"):
    """Plot feature importance for each observation factor."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reg_lons = [c[0] for c in region_coords]
    reg_lats = [c[1] for c in region_coords]
    view_limit = [min(reg_lons), max(reg_lons), min(reg_lats), max(reg_lats)]

    ori_polygon = Polygon(ori_coords)
    gdf_ori_area = gpd.GeoDataFrame(
        {'importance': [10.0], 'geometry': [ori_polygon]},
        crs="EPSG:4326"
    )

    for factor_name, geo_type_data in factor_data_map.items():
        safe_filename = factor_name.replace("/", "_").replace(" ", "")

        related_events = factor_to_disaster_map.get(factor_name, [])
        events_str = ", ".join(related_events)
        if len(events_str) > 30: events_str = events_str[:30] + "..."

        is_precip_related = "Administrative Region,Extreme Precipitation" in related_events

        gdf_list = []
        for geo_type, records in geo_type_data.items():
            try:
                gdf_base = get_cached_geojson(geo_type)
                if gdf_base is None: continue
                if gdf_base.crs != "EPSG:4326": gdf_base = gdf_base.to_crs(epsg=4326)

                df_scores = pd.DataFrame(records)
                merged_part = gdf_base.merge(df_scores, on='osm_id', how='inner')
                if not merged_part.empty:
                    weight = observation_importance.get(geo_type, 1.0)
                    merged_part['importance'] = merged_part['importance'] * weight
                    gdf_list.append(merged_part)
            except Exception as e:
                pass

        if not gdf_list and not is_precip_related:
            continue

        gdf_features = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True)) if gdf_list else None

        scores = []
        if is_precip_related:
            scores.extend(gdf_ori_area['importance'].tolist())
        if gdf_features is not None and not gdf_features.empty:
            scores.extend(gdf_features['importance'].tolist())

        if not scores: continue

        vmin, vmax = min(scores), max(scores)
        if vmin == vmax: vmin, vmax = vmin - 1, vmax + 1

        fig, ax = plt.subplots(figsize=(12, 12))
        cmap = 'YlOrRd'

        if is_precip_related:
            gdf_ori_area.plot(
                column='importance', ax=ax, cmap=cmap,
                vmin=vmin, vmax=vmax,
                alpha=0.6,
                zorder=1,
                edgecolor='none',
                linewidth=0
            )

        if gdf_features is not None and not gdf_features.empty:
            gdf_features = gdf_features.sort_values(by='importance', ascending=True)

            low_thresh = gdf_features['importance'].quantile(0.4)
            high_thresh = gdf_features['importance'].quantile(0.8)

            layers = [
                (gdf_features[gdf_features['importance'] <= low_thresh], 5),
                (gdf_features[(gdf_features['importance'] > low_thresh) & (gdf_features['importance'] <= high_thresh)],
                 6),
                (gdf_features[gdf_features['importance'] > high_thresh], 8)
            ]

            for layer_df, zo in layers:
                if layer_df.empty: continue

                has_point = (gdf_features.geom_type == 'Point').any()

                feat_kwargs = {
                    "column": 'importance', "ax": ax, "cmap": cmap,
                    "vmin": vmin, "vmax": vmax,
                    "zorder": zo,
                    "alpha": 1.0
                }

                if has_point and len(gdf_features) == len(gdf_features[gdf_features.geom_type == 'Point']):
                    feat_kwargs.update({"markersize": 100, "edgecolor": '#bd0026', "linewidth": 0})
                    gdf_features.plot(**feat_kwargs)
                else:
                    current_lw = 3 if zo < 8 else 4.5
                    feat_kwargs.update({"linewidth": current_lw})
                    gdf_features.plot(**feat_kwargs)

        ori_border = patches.Polygon(ori_coords, closed=True, linewidth=2,
                                     edgecolor='red', facecolor='none',
                                     linestyle='--', zorder=10, label='Initial Event Extent')
        ax.add_patch(ori_border)

        ax.set_xlim(view_limit[0], view_limit[1])
        ax.set_ylim(view_limit[2], view_limit[3])
        ax.set_title(f"Observation Importance Mapping: [{factor_name}]\nRelated Events: {events_str}", fontsize=15,
                     pad=15)
        ax.set_axis_off()

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
        cbar.set_label('Importance Rating')

        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', lw=2, label='Initial Area'),
            Line2D([0], [0], color='#bd0026', lw=3, label='High Priority Target')
        ]

        leg = ax.legend(handles=legend_elements, loc='upper left', frameon=True)
        leg.set_zorder(100)

        plt.savefig(f"{output_folder}/{safe_filename}_red.png", dpi=300, bbox_inches='tight')
        plt.close()


def sample_line_geometry(geom, step):
    """Sample points along a line at the given interval."""
    length = geom.length
    distances = np.arange(0, length, step)
    return [geom.interpolate(d) for d in distances]


def sample_polygon_geometry(polygon, step):
    """Sample points within a polygon at the given interval."""
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)

    points = []
    for x in xs:
        for y in ys:
            p = Point(x, y)
            if polygon.contains(p):
                points.append(p)
    return points


def calculate_observation_coverage_ratio(
        importance_grid,
        threshold,
        transform,
        disaster_shp_path,
        grid_crs="EPSG:4326",
        projected_crs="EPSG:3857"
):
    """Calculate observation coverage within the study region."""
    key_obs_mask = (importance_grid > threshold).astype(np.uint8)

    shapes = rasterio.features.shapes(key_obs_mask, transform=transform)
    key_obs_polygons = []
    for geom, val in shapes:
        if val == 1:
            key_obs_polygons.append(shape(geom))

    if not key_obs_polygons:
        return 0.0, 0.0, 0.0

    gdf_obs = gpd.GeoDataFrame(geometry=key_obs_polygons, crs=grid_crs)

    gdf_disaster = gpd.read_file(disaster_shp_path)
    if gdf_disaster.crs is None:
        gdf_disaster.set_crs(grid_crs, inplace=True)
    elif gdf_disaster.crs != grid_crs:
        gdf_disaster = gdf_disaster.to_crs(grid_crs)

    gdf_disaster['geometry'] = gdf_disaster['geometry'].buffer(0)
    gdf_obs['geometry'] = gdf_obs['geometry'].buffer(0)

    gdf_obs_proj = gdf_obs.to_crs(projected_crs)
    gdf_disaster_proj = gdf_disaster.to_crs(projected_crs)

    obs_geom_union = gdf_obs_proj.unary_union
    disaster_geom_union = gdf_disaster_proj.unary_union

    if disaster_geom_union.is_empty:
        return 0.0, 0.0, 0.0

    total_disaster_area = disaster_geom_union.area
    intersection_geom = disaster_geom_union.intersection(obs_geom_union)
    intersection_area = intersection_geom.area

    ratio = intersection_area / total_disaster_area if total_disaster_area > 0 else 0.0

    return ratio, intersection_area, total_disaster_area


def plot_factor_specific_importance_field_basemap(
        factor_data_map,
        factor_to_disaster_map,
        region_coords,
        ori_coords,
        output_folder="./data/output_factor_maps_grid_beijing_2",
        grid_size=200,
        blur_sigma=5,
        bg_image_path='background_map.jpg'
):
    """Plot and export factor importance fields over a basemap."""
    os.makedirs(output_folder, exist_ok=True)
    geotiff_folder = f"{output_folder}_geotiff"
    os.makedirs(geotiff_folder, exist_ok=True)

    reg_lons = [c[0] for c in region_coords]
    reg_lats = [c[1] for c in region_coords]
    xmin, xmax = min(reg_lons), max(reg_lons)
    ymin, ymax = min(reg_lats), max(reg_lats)

    pixel_width = (xmax - xmin) / grid_size
    pixel_height = (ymax - ymin) / grid_size
    transform = from_origin(xmin, ymax, pixel_width, pixel_height)

    region_poly = Polygon(region_coords)
    region_mask = rasterize(
        [(region_poly, 1)],
        out_shape=(grid_size, grid_size),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    for factor_name, geo_map in factor_data_map.items():
        importance_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        for geo_type, records in geo_map.items():
            gdf = get_cached_geojson(geo_type)
            if gdf is None or gdf.empty: continue
            if gdf.crs != "EPSG:4326": gdf = gdf.to_crs(epsg=4326)

            df = pd.DataFrame(records)
            merged = gdf.merge(df, on="osm_id", how="inner")

            extra_rows = []
            for record in records:
                if len(record['osm_id']) > 20:
                    feature_node = OSMFeature.nodes.first_or_none(osm_id=record['osm_id'])
                    if feature_node:
                        geometry_node = feature_node.has_geometry.all()
                        if geometry_node:
                            geom_val = geometry_node[0].geometry
                            try:
                                if isinstance(geom_val, str):
                                    record['geometry'] = wkt.loads(geom_val)
                                else:
                                    record['geometry'] = geom_val

                                extra_rows.append(record)
                            except Exception as e:
                                pass

            if extra_rows:
                df_extra = gpd.GeoDataFrame(extra_rows, geometry='geometry', crs="EPSG:4326")
                merged = pd.concat([merged, df_extra], ignore_index=True)

            if merged.empty: continue

            for _, row in merged.iterrows():
                geom = row.geometry
                base_importance = row.importance
                if base_importance <= 0: continue

                feature_type = str(row.get("osm_id", "")).split("_")[0]
                type_weight = observation_importance.get(feature_type, 1.0)
                
                final_value = base_importance * type_weight

                if geom.geom_type in ["Polygon", "MultiPolygon"]:
                    poly_raster = rasterize([(geom, final_value)], out_shape=(grid_size, grid_size),
                                            transform=transform, fill=0, dtype=np.float32)
                    importance_grid = np.maximum(importance_grid, poly_raster)
                elif geom.geom_type in ["Point", "LineString"]:
                    if geom.geom_type == "Point":
                        points = [geom]
                    else:
                        step = 2 * pixel_width
                        distances = np.arange(0, geom.length, step)
                        points = [geom.interpolate(d) for d in distances]
                    for p in points:
                        ix = int((p.x - xmin) / pixel_width)
                        iy = int((ymax - p.y) / pixel_height)
                        if 0 <= ix < grid_size and 0 <= iy < grid_size:
                            importance_grid[iy, ix] = max(
                                float(importance_grid[iy, ix]),
                                final_value
                            )

        importance_grid = gaussian_filter(importance_grid, sigma=blur_sigma)
        NODATA_VALUE = -9999.0
        valid_area = (region_mask == 1)
        importance_grid[valid_area] = np.nan_to_num(importance_grid[valid_area], nan=0.0)
        importance_grid[region_mask == 0] = NODATA_VALUE

        safe_factor_name = re.sub(r'[\\/:*?"<>|]', '_', factor_name).strip()
        tif_path = f"{geotiff_folder}/{safe_factor_name}_importance_field.tif"

        with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=grid_size,
                width=grid_size,
                count=1,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
                nodata=NODATA_VALUE
        ) as dst:
            dst.write(importance_grid, 1)

        fig, ax = plt.subplots(figsize=(12, 12))

        try:
            bg_img = mpimg.imread(bg_image_path)
            ax.imshow(bg_img, extent=[xmin, xmax, ymin, ymax], origin='upper', zorder=1)
        except FileNotFoundError:
            pass

        masked_grid = np.ma.masked_where(importance_grid == NODATA_VALUE, importance_grid)
        im = ax.imshow(
            masked_grid,
            extent=[xmin, xmax, ymin, ymax],
            origin="upper",
            cmap="YlOrRd",
            alpha=0.65,
            zorder=2
        )

        valid_values = importance_grid[importance_grid != NODATA_VALUE]
        threshold = 0
        if len(valid_values) > 0 and np.max(valid_values) > np.min(valid_values):
            threshold = np.min(valid_values) + 0.18 * (np.max(valid_values) - np.min(valid_values))

            lons = np.linspace(xmin, xmax, grid_size)
            lats = np.linspace(ymax, ymin, grid_size)
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            DEEP_ORANGE = '#FF8C00'
            SHADOW_COLOR = '#333333'

            dx, dy = (xmax - xmin) * 0.003, (ymax - ymin) * 0.003
            shadow_contour = ax.contour(
                lon_grid + dx, lat_grid - dy,
                importance_grid,
                levels=[threshold],
                colors=[SHADOW_COLOR],
                linewidths=4,
                alpha=0.4,
                linestyles='-',
                zorder=3.5
            )

            main_contour = ax.contour(
                lon_grid, lat_grid,
                importance_grid,
                levels=[threshold],
                colors=[DEEP_ORANGE],
                linewidths=2.5,
                linestyles='-',
                zorder=4
            )

        ax.plot(reg_lons + [reg_lons[0]], reg_lats + [reg_lats[0]],
                color="black", linewidth=1.5, zorder=3)

        ax.set_axis_off()

        png_path = f"{output_folder}/{safe_factor_name}_field.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()


def run_distance_fields(config, database=None, snapshot=None, output_dir=None, no_plots=False):
    """Generate distance-decay fields from the instance graph or a snapshot."""
    from distance_fields import run
    return run(config, snapshot, output_dir, no_plots, database=database)


def main_init():
    """Run the legacy mapping workflow for the configured study region."""
    zhengzhou_coords = [
        [113.549609, 34.791410],
        [113.657406, 34.791410],
        [113.657406, 34.702835],
        [113.549609, 34.702835],
        [113.549609, 34.791410]
    ]
    region_coords = zhengzhou_coords
    ori_coords = zhengzhou_coords

    mission = 'zhengzhou_all_frame1_o_sensor_t'
    event_specific_importance_folder = f"./data/ins_figures/{mission}/output_event_maps"
    event_specific_importance_field_folder = f"./data/ins_figures/{mission}/output_event_maps_grid"
    factor_specific_importance_folder = f"./data/ins_figures/{mission}/output_factor_maps"
    factor_specific_importance_field_folder = f"./data/ins_figures/{mission}/output_factor_maps_grid"

    from ontology.neo4jmodel import Neo4jDatabaseConnector
    Neo4jDatabaseConnector(ins_graph_database)

    features_output = calculate_feature_importance_in_same_disaster(True)
    factor_to_disaster = get_factor_to_disaster()

    plot_event_specific_importance(features_output, region_coords, ori_coords, event_specific_importance_folder)

    plot_event_specific_importance_field_basemap(features_output, region_coords, ori_coords,
                                                 event_specific_importance_field_folder,
                                                 aggregate_by_phenomenon=True,
                                                 bg_image_path="./data/geo_dataset/geos_zhengzhou/bitmap.png")

    factors_output = aggregate_importance_by_factor(factor_to_disaster, features_output)

    plot_factor_specific_importance(factors_output, factor_to_disaster, region_coords, ori_coords,
                                    factor_specific_importance_folder)

    plot_factor_specific_importance_field_basemap(factors_output, factor_to_disaster, region_coords, ori_coords,
                                                  output_folder=factor_specific_importance_field_folder,
                                                  bg_image_path="./data/geo_dataset/geos_zhengzhou/bitmap.png")


if __name__ == "__main__":
    main_init()
