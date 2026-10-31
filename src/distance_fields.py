"""Read an instantiated graph and map feature-distance-decay importance fields.

python plot_instantiation_distance_fields.py --config instantiation_config.json
python plot_instantiation_distance_fields.py --snapshot PATH --output-dir PATH

Read-only Neo4j access; no importing instantiation_en or any LLM/database models.
Events retain max(probability * per-event importance) * 1.05**(n_paths-1).
Each feature contributes its score times a geometry-specific compact kernel,
zero outside its cutoff. Distance is to the full geometry in a local metric CRS.
Spatial overlap uses a logarithmic bonus above the strongest contribution
by default (optional sum/max); factor weights are applied exactly once to the
associated full event description, not to unrelated geography/event classes.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import urllib.parse

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pyproj import CRS, Transformer
import shapely
from shapely.geometry import shape, mapping, box
from shapely.ops import transform, unary_union

from standard_observation_factors import factor_key


# Keep the OF numbering used by instantiation.py and the manuscript figure.
OBSERVATION_FACTOR_LABELS = {
    factor_key(name):f'OF{number}'
    for number,names in enumerate((
        ('沉降深度','Subsidence Depth'),
        ('地表覆盖情况','Land Cover'),
        ('地表温度','Surface Temperature'),
        ('地表形变速率、形变量','Surface Deformation'),
        ('风速','Wind Speed'),
        ('风向','Wind Direction'),
        ('降水量','Precipitation'),
        ('结构破坏特征','Structural Characteristics'),
        ('空气湿度','Air Humidity'),
        ('空气温度','Air Temperature'),
        ('气压','Atmospheric Pressure'),
        ('裂缝宽度','Crack Width'),
        ('倾斜倾角','Tilt Angle'),
        ('设施形变速率、形变量','Infrastructure Deformation'),
        ('渗透（孔隙）水压力','Seepage Pressure'),
        ('水流速度','Water Flow Velocity'),
        ('水流量','Water Flow Rate'),
        ('水体面积','Water Body Area'),
        ('水位','Water Level'),
        ('水质','Water Quality'),
        ('土壤湿度','Soil Moisture'),
        ('云顶高度','Cloud Top Height'),
        ('云顶温度','Cloud Top Temperature'),
        ('云覆盖率','Cloud Fraction'),
        ('植被覆盖率','Vegetation Fraction'),
    ),1)
    for name in names
}


def cell_histogram_settings(config):
    """WGS84 target coordinates; null disables the additional chart."""
    coord = config.get('target_coord',[113.60,34.75])
    if coord is not None:
        if not isinstance(coord,(list,tuple)) or len(coord)!=2:
            raise ValueError('target_coord must be [longitude, latitude] or null.')
        try:
            coord = [float(value) for value in coord]
        except (TypeError,ValueError) as exc:
            raise ValueError('target_coord must contain numeric longitude and latitude.') from exc
        if not all(math.isfinite(value) for value in coord) or not (-180<=coord[0]<=180 and -90<=coord[1]<=90):
            raise ValueError('target_coord must be finite WGS84 longitude/latitude.')
    filename = config.get('histogram_filename','cell_importance_histogram.png')
    if (not isinstance(filename,str) or not filename.strip()
            or re.search(r'[\\/:*?"<>|]',filename) or not filename.lower().endswith('.png')):
        raise ValueError('histogram_filename must be a PNG filename without directories.')
    config['target_coord'] = coord
    config['histogram_filename'] = filename
    return coord,filename


def sample_target_cell(factor_fields, grid, target_coord):
    """Read the containing pixel from compact valid-cell vectors, in raster CRS."""
    result = dict(status='disabled',target_coord=target_coord,row=None,col=None,factors=[],image=None)
    if target_coord is None:
        return result
    x,y = Transformer.from_crs(4326,grid['crs'],always_xy=True).transform(*target_coord)
    if not math.isfinite(x) or not math.isfinite(y):
        result['status'] = 'outside_grid'
        return result
    col_float,row_float = ~grid['transform']*(x,y)
    row,col = math.floor(row_float),math.floor(col_float)
    if not (0<=row<grid['height'] and 0<=col<grid['width']):
        result['status'] = 'outside_grid'
        return result
    result.update(row=row,col=col)
    if not grid['valid'][row,col]:
        result['status'] = 'invalid_cell'
        return result
    # Fields are packed by grid['indices']; a raster flat index is not a vector index.
    flat_index = row*grid['width']+col
    matches = np.flatnonzero(grid['indices']==flat_index)
    if len(matches)!=1:
        raise ValueError('Target cell is missing or duplicated in valid grid indices.')
    index = int(matches[0])
    scores = []
    for name,vector in factor_fields.items():
        # Match the exported float32 GeoTIFF value; no display-scale normalization.
        value = float(np.float32(vector[index]))
        if not math.isfinite(value) or value<0:
            raise ValueError(f'Invalid target-cell factor score: {name}')
        scores.append(dict(factor=name,label=OBSERVATION_FACTOR_LABELS.get(factor_key(name),name),score=value))
    result['factors'] = sorted(scores,key=lambda item:(-item['score'],item['factor']))
    result['status'] = 'sampled' if any(item['score']>0 for item in scores) else 'no_positive_values'
    return result


def export_cell_importance_histogram(factor_fields, grid, config, out):
    """Export the target cell's positive factor scores in descending OF order."""
    coord,filename = cell_histogram_settings(config)
    result = sample_target_cell(factor_fields,grid,coord)
    if result['status']!='sampled':
        if result['status']!='disabled':
            print(f'Skipped cell importance histogram at {coord}: {result["status"]}',flush=True)
        return result
    if not config.get('render_png',True):
        result['status'] = 'render_png_disabled'
        return result
    import matplotlib.pyplot as plt
    items = [item for item in result['factors'] if item['score']>0]
    labels = [item['label'] for item in items]
    # Local style avoids changing the map figures or the caller's plotting defaults.
    with plt.rc_context({'font.family':['Times New Roman','DejaVu Sans'], 'axes.unicode_minus':False}):
        fig,ax = plt.subplots(figsize=(12,6))
        try:
            x_pos = np.arange(len(items))*.5
            ax.bar(x_pos,[item['score'] for item in items],width=.25,
                   color='#1A365D',edgecolor='#000000',linewidth=1.,alpha=.9)
            if config.get('show_title',True):
                ax.set_title(f'Target Cell Location ({coord[0]:.4f}, {coord[1]:.4f}) - Factor Importance',
                             fontsize=16,pad=15)
            ax.set_xlabel('Observable Factors',fontsize=12)
            ax.set_ylabel('Observation Priority',fontsize=12)
            ax.set_xticks(x_pos,labels,rotation=45,ha='right',fontsize=14)
            ax.tick_params(axis='y',labelsize=14)
            ax.grid(axis='y',linestyle='--',alpha=.5)
            ax.set_xlim(x_pos[0]-.25,x_pos[-1]+.25)
            fig.tight_layout()
            path = Path(out)/'output_factor_maps_grid'/filename
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                raise FileExistsError(f'Cell histogram would overwrite an existing image: {path}')
            fig.savefig(path,dpi=config.get('dpi',180),bbox_inches='tight')
        finally:
            plt.close(fig)
    result.update(status='rendered',image=str(path.resolve()))
    print(f'Exported cell importance histogram: {path}',flush=True)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def literal_setting(path, name, class_name=None):
    """Read literal configuration without importing side-effect-heavy modules."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    scope = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name) if class_name else tree
    values = []
    for node in ast.walk(scope):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            try:
                values.append((node.lineno,ast.literal_eval(node.value)))
            except (ValueError,TypeError):
                pass
    if not values:
        raise ValueError(f'No literal {name} in {path}')
    return sorted(values)[-1][1]


def event_name(description):
    return re.split('[,，]',description)[-1].strip()


def safe_name(name, factor=False):
    return re.sub(r'[\\/:*?"<>|]', '_', name).strip() if factor else name.replace(' ','').replace(',','_').replace('/','_')


def normalize_weights(weights):
    result = {}
    for key,value in weights.items():
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f'Invalid hazard weight: {key}')
        result[key.strip().casefold()] = number
    return result


def geo_type_weight_settings(config):
    """Resolve Chinese/English geographic-type weights to the graph's English types."""
    enabled = config.get('apply_geo_type_importance',True)
    if not isinstance(enabled,bool):
        raise ValueError('apply_geo_type_importance must be true or false.')
    config['apply_geo_type_importance'] = enabled
    if not enabled:
        config['geo_type_weight_by_type'] = {}
        return dict(enabled=False,source='disabled by configuration',dictionary=None,
                    configured_types=0,resolved_types=0,weights={})
    raw_weights = config.get('geo_type_importance') or {}
    if not isinstance(raw_weights,dict):
        raise ValueError('geo_type_importance must be a dictionary or null.')
    dictionary_path = config.get('geo_type_dictionary')
    translations = read_json(dictionary_path) if dictionary_path else {}
    if not isinstance(translations,dict):
        raise ValueError('geo_type_dictionary must contain a JSON object.')
    resolved = {}
    for source_name,value in raw_weights.items():
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f'Invalid geographic-type importance: {source_name}')
        geo_type = str(translations.get(str(source_name).strip(),source_name)).strip()
        if not geo_type:
            raise ValueError('Geographic-type importance contains an empty name.')
        key = geo_type.casefold()
        if key in resolved and not math.isclose(resolved[key],weight):
            raise ValueError(f'Conflicting geographic-type importance for {geo_type}.')
        resolved[key] = weight
    config['geo_type_weight_by_type'] = resolved
    return dict(enabled=True,source=config.get('geo_type_importance_source','configuration'),
                dictionary=str(dictionary_path) if dictionary_path else None,
                configured_types=len(raw_weights),resolved_types=len(resolved),weights=resolved)


def hazard_weight(description, weights):
    return weights.get(description.strip().casefold(),weights.get(event_name(description).casefold(),1.))


def cutoff_for(description, geo_type, config, geometry_kind=None):
    specific = config.get('event_geo_cutoff_m',{}).get(description)
    event = config.get('event_cutoff_m',{}).get(event_name(description))
    geo = config.get('geo_type_cutoff_m',{}).get(geo_type)
    geometry_cutoff = config.get('geometry_cutoff_m',{}).get(geometry_kind)
    value = float(next(v for v in (specific,event,geo,geometry_cutoff,config.get('default_cutoff_m',500.)) if v is not None))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Invalid cutoff for {description}: {value}')
    return value


def compact_decay(distances, cutoff):
    """Biweight kernel: K(0)=1, K(h)=0; finite support and no Gaussian blur."""
    d = np.asarray(distances,dtype=float)
    if not math.isfinite(cutoff) or cutoff<=0 or np.any(d<0) or not np.isfinite(d).all():
        raise ValueError('Distances must be finite and nonnegative; cutoff must be positive.')
    return np.maximum(0.,1.-(d/cutoff)**2)**2


DEFAULT_GEOMETRY_KERNELS = {'point':'compact_biweight','line':'compact_exponential','polygon':'linear'}


def geometry_decay_settings(config):
    overrides = config.get('geometry_kernels',{})
    cutoffs = config.get('geometry_cutoff_m',{})
    if not isinstance(overrides,dict) or not isinstance(cutoffs,dict):
        raise ValueError('geometry_kernels and geometry_cutoff_m must be dictionaries.')
    if (set(overrides)|set(cutoffs))-set(DEFAULT_GEOMETRY_KERNELS):
        raise ValueError('Geometry keys must be point, line or polygon.')
    kernels = {**DEFAULT_GEOMETRY_KERNELS,**overrides}
    if any(v not in ('compact_biweight','compact_exponential','linear') for v in kernels.values()):
        raise ValueError('Unsupported geometry decay kernel.')
    for value in cutoffs.values():
        if not math.isfinite(float(value)) or float(value)<=0:
            raise ValueError('Geometry cutoffs must be finite and positive.')
    rate = float(config.get('exponential_decay_rate',3.))
    if not math.isfinite(rate) or rate<=0:
        raise ValueError('exponential_decay_rate must be finite and positive.')
    return kernels,rate


def geometry_parts_by_kind(geometry):
    """Keep multipart geometry whole; group mixed collections without double counting."""
    groups = defaultdict(list)
    def visit(part):
        if part.is_empty:
            return
        types = {'Point':'point','MultiPoint':'point','LineString':'line',
                 'MultiLineString':'line','LinearRing':'line','Polygon':'polygon','MultiPolygon':'polygon'}
        if part.geom_type=='GeometryCollection':
            for child in part.geoms:
                visit(child)
        elif part.geom_type in types:
            groups[types[part.geom_type]].append(part)
        else:
            raise ValueError(f'Unsupported geometry: {part.geom_type}')
    visit(geometry)
    if not groups:
        raise ValueError('Feature geometry has no nonempty point/line/polygon components.')
    return {kind:unary_union(parts) for kind,parts in groups.items()}


def distance_decay(distances, cutoff, kernel, exponential_rate=3.):
    if kernel=='compact_biweight':
        return compact_decay(distances,cutoff)
    d = np.asarray(distances,dtype=float)
    if not math.isfinite(cutoff) or cutoff<=0 or np.any(d<0) or not np.isfinite(d).all():
        raise ValueError('Distances must be finite and nonnegative; cutoff must be positive.')
    ratio = np.minimum(d/cutoff,1.)
    if kernel=='linear':
        return 1.-ratio
    if kernel=='compact_exponential':
        if not math.isfinite(exponential_rate) or exponential_rate<=0:
            raise ValueError('Exponential decay rate must be finite and positive.')
        # Rescaled exponential: continuous, K(0)=1 and K(h)=0; stable for small rates.
        return np.maximum(0.,np.exp(-exponential_rate*ratio)*
                          np.expm1(-exponential_rate*(1.-ratio))/np.expm1(-exponential_rate))
    raise ValueError(f'Unsupported geometry decay kernel: {kernel}')


def aggregation_mode(config):
    mode = config.get('spatial_aggregation','log_overlap')
    if mode not in ('sum','max','log_overlap'):
        raise ValueError('spatial_aggregation must be log_overlap, sum or max.')
    return mode


def overlap_strength(config):
    value = float(config.get('overlap_strength',.25))
    if not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('overlap_strength must be finite and between 0 and 1.')
    return value


def logarithmic_overlap(total, peak, strength):
    """M * (1 + alpha * ln(S/M)); zero without support, unchanged for one source.

    For 0<=alpha<=1 this is monotone in each contribution and lies in [M,S].
    Compute once after collecting S and M, never recursively on partial results.
    """
    ratio = np.divide(total,peak,out=np.ones_like(total),where=peak>0)
    return peak*(1.+strength*np.log(np.maximum(ratio,1.)))


def spatial_formula(config):
    mode = aggregation_mode(config)
    contribution = 'c_i = score_i * max_over_geometry_parts K_kind(distance_to_full_geometry, cutoff_i); kernels=' + json.dumps(geometry_decay_settings(config)[0],sort_keys=True)
    if mode=='log_overlap':
        return 'M * (1 + overlap_strength * ln(S/M)); S=sum_i c_i; M=max_i c_i; zero if M=0; '+contribution
    return mode+'_i c_i; '+contribution


def display_scale_settings(config):
    """Set a shared display range for every map produced by one method."""
    for key in ('apply_display_scale_maximum','show_title','show_colorbar','show_scale_bar'):
        value = config.get(key, True)
        if not isinstance(value, bool):
            raise ValueError(f'{key} must be true or false.')
        config[key] = value
    scope = config.get('display_scale_scope', 'by_kind_per_method')
    if scope != 'by_kind_per_method':
        raise ValueError('display_scale_scope must be by_kind_per_method.')
    scale_bar_km = float(config.get('scale_bar_km', 2.0))
    if not math.isfinite(scale_bar_km) or scale_bar_km <= 0:
        raise ValueError('scale_bar_km must be finite and positive.')
    config['display_scale_scope'] = scope
    config['scale_bar_km'] = scale_bar_km
    return scope, scale_bar_km


def configured_display_maxima(config):
    """Return optional fixed colour-scale maxima for event and factor maps."""
    values = config.get('display_scale_maximum_by_kind')
    if values is None:
        return {}
    if not isinstance(values,dict) or set(values)-{'event','factor'}:
        raise ValueError('display_scale_maximum_by_kind must contain only event and/or factor.')
    checked = {}
    for kind,value in values.items():
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'display_scale_maximum_by_kind.{kind} must be finite and positive, or null.')
        checked[kind] = value
    config['display_scale_maximum_by_kind'] = {kind:checked.get(kind) for kind in ('event','factor')}
    return checked


def set_display_scale(event_fields, factor_fields, config):
    """Use shared kind scales when enabled, otherwise scale each field itself."""
    scope, scale_bar_km = display_scale_settings(config)
    configured = configured_display_maxima(config)
    if not config['apply_display_scale_maximum']:
        configured = {}
    groups = {'event':event_fields, 'factor':factor_fields}
    maxima, sources, field_maxima = {},{},{}
    for kind,group in groups.items():
        if not config['apply_display_scale_maximum']:
            field_maxima[kind] = {name:display_maximum_for_field(values,kind,config)
                                  for name,values in group.items()}
            sources[kind] = 'automatic_per_field'
            continue
        automatic = max((float(np.max(values)) for values in group.values()), default=0.)
        maximum = configured.get(kind,automatic)
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError('Display-scale maximum must be finite and nonnegative.')
        maxima[kind] = maximum if maximum > 0 else 1.
        sources[kind] = 'configured' if kind in configured else 'automatic'
        field_maxima[kind] = {name:maxima[kind] for name in group}
    config['plot_vmax_by_kind'] = maxima
    config['plot_vmax_by_field'] = field_maxima
    effective_scope = scope if config['apply_display_scale_maximum'] else 'per_field'
    return dict(scope=effective_scope, maximum_by_kind=maxima, maximum_by_field=field_maxima,
                maximum_source_by_kind=sources, scale_bar_km=scale_bar_km,
                apply_configured_maximum=config['apply_display_scale_maximum'],
                show_title=config['show_title'], show_colorbar=config['show_colorbar'],
                show_scale_bar=config['show_scale_bar'],
                value_semantics='method-specific importance score; not a probability')


def display_maximum_for_field(values, kind, config):
    """Resolve the actual colour-scale upper bound, ignoring masked cells."""
    valid_values = np.ma.asarray(values).compressed()
    automatic = float(valid_values.max()) if len(valid_values) else 0.
    maximum = (config.get('plot_vmax_by_kind',{}).get(kind,automatic)
               if config.get('apply_display_scale_maximum',True) else automatic)
    maximum = float(maximum)
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError('Display-scale maximum must be finite and nonnegative.')
    return maximum if maximum > 0 else 1.


def aggregate_graph_feature(row, path_boost, exclude_types):
    properties = row.get('properties') or {}
    if isinstance(properties,str):
        properties = json.loads(properties)
    if not isinstance(properties,dict):
        raise ValueError(f'Invalid per-event properties for {row["osm_id"]}')
    groups = defaultdict(list)
    seen = set()
    for item in row['events']:
        geo,description,node_id = item['geo_type'],item['description'],item['node_id']
        if geo in exclude_types:
            continue
        key = (geo,description,node_id)
        if key in seen:
            continue
        seen.add(key)
        details = properties.get(node_id)
        if not isinstance(details,dict) or 'importance' not in details:
            raise ValueError(f'Missing per-event importance: {row["osm_id"]}, {node_id}')
        importance,probability = float(details['importance']),float(item['probability'])
        if not np.isfinite([importance,probability]).all() or importance<0 or probability<0:
            raise ValueError(f'Invalid graph score: {row["osm_id"]}, {node_id}')
        groups[(geo,description)].append(probability*importance)
    output = []
    for (geo,description),values in sorted(groups.items()):
        score = max(values)*path_boost**(len(values)-1)
        if not math.isfinite(score):
            raise ValueError(f'Importance overflow: {row["osm_id"]}')
        if score>0:
            output.append(dict(geo_type=geo,osm_id=row['osm_id'],event_description=description,
                               score=score,n_paths=len(values)))
    return output


def open_driver(config):
    from neo4j import GraphDatabase
    url = os.environ.get('NEO4J_DATABASE_URL') or literal_setting(config['connection_module'],'DATABASE_URL','Neo4jDatabaseConnector')
    parsed = urllib.parse.urlsplit(url)
    username = os.environ.get('NEO4J_USERNAME') or urllib.parse.unquote(parsed.username or '')
    password = os.environ.get('NEO4J_PASSWORD') or urllib.parse.unquote(parsed.password or '')
    host = parsed.hostname or 'localhost'
    if ':' in host:
        host = '['+host+']'
    uri = os.environ.get('NEO4J_URI') or f'{parsed.scheme}://{host}:{parsed.port or 7687}'
    return GraphDatabase.driver(uri,auth=(username,password),connection_timeout=15,
                               max_transaction_retry_time=0)


def initial_event_record(row, fallback_importance=90.):
    """Represent one tree root by its disPhen and complete impact geometry."""
    geo = str(row['geo_type']).strip()
    phenomena = sorted({str(p).strip() for p in row.get('phenomena',[]) if p})
    if len(phenomena)>1:
        raise ValueError(f'Multiple disPhen arguments on initial event {row["node_id"]}')
    phenomenon = phenomena[0] if phenomena else event_name(row['description'])
    if (geo.casefold() in ('administrative region','行政区','行政区域') and
            phenomenon.casefold() in ('extreme precipitation','极端降水')):
        return None
    if not geo or not phenomenon:
        raise ValueError(f'Missing geoType/disPhen on initial event {row["node_id"]}')
    node_id = str(row['node_id'])
    features = row.get('features',[]) or []
    importance_values = []
    for feature in features:
        properties = feature.get('properties') or {}
        if isinstance(properties,str):
            properties = json.loads(properties)
        details = properties.get(node_id)
        if isinstance(details,dict) and details.get('importance') is not None:
            importance_values.append(float(details['importance']))
    if importance_values and (not np.isfinite(importance_values).all() or min(importance_values)<0):
        raise ValueError(f'Invalid initial event importance: {node_id}')
    importance = max(importance_values) if importance_values else float(fallback_importance)
    probability = 1. if row.get('probability') is None else float(row['probability'])
    if not np.isfinite([importance,probability]).all() or importance<0 or probability<0:
        raise ValueError(f'Invalid initial event score: {node_id}')
    raw_geometries = [v for v in row.get('geometries',[]) if v]
    if not raw_geometries:
        raw_geometries = [f['geometry'] for f in features if f.get('geometry')]
    geometries = []
    for value in raw_geometries:
        if isinstance(value,dict):
            geom = shape(value)
        elif str(value).lstrip().startswith('{'):
            geom = shape(json.loads(value))
        else:
            geom = shapely.from_wkt(value)
        geometries.append(shapely.make_valid(geom))
    if not geometries:
        raise ValueError(f'Missing impact geometry for initial event {node_id}')
    geometry = shapely.make_valid(unary_union(geometries))
    if geometry.is_empty or not np.isfinite(geometry.bounds).all():
        raise ValueError(f'Invalid initial event geometry: {node_id}')
    west,south,east,north = geometry.bounds
    if west < -180 or east > 180 or south < -90 or north > 90:
        raise ValueError(f'Initial event geometry is not WGS84 longitude/latitude: {node_id}')
    record = dict(geo_type=geo,osm_id=f'initial_event_{node_id}_{safe_name(geo)}',
                  event_description=f'{geo},{phenomenon}',score=probability*importance,n_paths=1,
                  source_kind='initial_event',node_id=node_id,disPhen=phenomenon,
                  original_event_description=row['description'],probability=probability,
                  importance=importance,
                  importance_source='graph_per_event' if importance_values else 'initial_event_importance')
    return record,geometry


def fetch_graph(config):
    """Stream each feature once; properties are not repeated per event path."""
    from neo4j import Query
    query = '''MATCH (d:DisasterEvent)-[:ARGUMENT]->(g:GeoObject)-[:HAS_FEATURE]->(f:OSMFeature)
    WHERE NOT g.geo_object_name IN $excluded AND coalesce(d.depth,-1) <> 0
    WITH DISTINCT d,g,f
    WITH f, collect({node_id:d.node_id,description:d.event_description,
                    probability:d.probability,geo_type:g.geo_object_name}) AS events
    RETURN f.osm_id AS osm_id, f.properties AS properties, events'''
    records,initial_geometries,root_records = [],{},{}
    initial_importance = float(config.get('initial_event_importance',90.))
    if not math.isfinite(initial_importance) or initial_importance<0:
        raise ValueError('initial_event_importance must be finite and nonnegative.')
    with open_driver(config) as driver:
        with driver.session(database=config['database'],default_access_mode='READ',fetch_size=500) as session:
            for i,row in enumerate(session.run(Query(query,timeout=300),excluded=config['exclude_geo_types']),1):
                records.extend(aggregate_graph_feature(dict(row),config['path_boost'],config['exclude_geo_types']))
                if i%5000==0:
                    print(f'Read {i} graph features; {len(records)} positive feature-event scores',flush=True)
            root_query = '''MATCH (d:DisasterEvent)-[:ARGUMENT]->(g:GeoObject)
                WHERE d.depth = 0
                OPTIONAL MATCH (d)-[:ARGUMENT]->(p)
                WHERE 'Impact' IN labels(p) OR 'DisPhen' IN labels(p)
                WITH d,g,collect(DISTINCT coalesce(p.impact_name,p.impact_type)) AS phenomena
                OPTIONAL MATCH (g)-[:HASGEOMETRY]->(extent:Geometry)
                WITH d,g,phenomena,collect(DISTINCT extent.geometry) AS geometries
                OPTIONAL MATCH (g)-[:HAS_FEATURE]->(f:OSMFeature)
                OPTIONAL MATCH (f)-[:HASGEOMETRY|HAS_GEOMETRY]->(fg:Geometry)
                RETURN d.node_id AS node_id,d.event_description AS description,
                    coalesce(g.geo_type,g.geo_object_name) AS geo_type,d.probability AS probability,
                    phenomena,geometries,
                    collect(DISTINCT {properties:f.properties,geometry:fg.geometry}) AS features'''
            for row in session.run(Query(root_query,timeout=120)):
                item = initial_event_record(dict(row),initial_importance)
                if item is None:
                    continue
                record,geometry = item
                records.append(record)
                root_records.setdefault(record['node_id'],[]).append(record)
                initial_geometries[record['osm_id']] = mapping(geometry)
            factors = defaultdict(set)
            for row in session.run(Query('''MATCH (d:DisasterEvent)-[:TRIGGER]->(:ObservationEvent)-[:OBSERVED]->(f)
                WHERE f:Factor OR f:StandardFactor
                RETURN DISTINCT d.node_id AS node_id,d.event_description AS event, f.factor_name AS factor''',timeout=60)):
                if row['factor'] and row['event']:
                    factors[row['factor']].add(row['event'])
                    for record in root_records.get(row['node_id'],[]):
                        factors[row['factor']].add(record['event_description'])
                        record.setdefault('factor_scores',{})[row['factor']] = record['score']
            for group in root_records.values():
                for record in group:
                    record.setdefault('factor_scores',{})
    if not records or not factors:
        raise ValueError('Instance graph has no positive feature-event scores or factor associations.')
    return dict(schema_version=1,source='neo4j',database=config['database'],
                captured_at=datetime.now(timezone.utc).isoformat(),path_boost=config['path_boost'],
                records=records,factor_to_disaster={k:sorted(v) for k,v in sorted(factors.items())},
                initial_events_included=True,initial_event_count=len(root_records),
                initial_event_feature_count=sum(map(len,root_records.values())),
                initial_event_geometries=initial_geometries)


def region_geometry(path):
    data = read_json(path)
    geoms = ([f['geometry'] for f in data['features']] if data['type']=='FeatureCollection'
             else [data['geometry']] if data['type']=='Feature' else [data])
    region = shapely.make_valid(unary_union([shape(g) for g in geoms if g]))
    source_crs = CRS.from_user_input(data.get('crs',{}).get('properties',{}).get('name','EPSG:4326'))
    if not source_crs.equals(CRS.from_epsg(4326),ignore_axis_order=True):
        region = transform(Transformer.from_crs(source_crs,4326,always_xy=True).transform,region)
    if region.is_empty or region.geom_type not in ('Polygon','MultiPolygon'):
        raise ValueError('Study region must be a nonempty Polygon/MultiPolygon.')
    return region


def resolve_geometries(snapshot, config, live_graph):
    """Resolve original GeoType_row IDs, with graph WKT only for unresolved IDs."""
    if 'geometries' in snapshot:
        geometry_map = {key:shape(value) for key,value in snapshot['geometries'].items()}
        return geometry_map,snapshot.get('geometry_inputs',{})
    ids = defaultdict(set)
    for record in snapshot['records']:
        ids[record['geo_type']].add(record['osm_id'])
    geometry_map = {key:shape(value) for key,value in snapshot.get('initial_event_geometries',{}).items()}
    inputs = {}
    for geo,needed in sorted(ids.items()):
        needed = needed-geometry_map.keys()
        if not needed:
            continue
        filename = Path(config['features_folder'])/(geo+'.json')
        if filename.exists():
            collection = read_json(filename)
            inputs[str(filename.resolve())] = file_hash(filename)
            crs = collection.get('crs',{}).get('properties',{}).get('name','EPSG:4326')
            project = Transformer.from_crs(crs,4326,always_xy=True).transform
            for index,feature in enumerate(collection['features']):
                key = f'{geo}_{index}'
                if key in needed and feature.get('geometry'):
                    geom = shapely.make_valid(transform(project,shape(feature['geometry'])))
                    if not geom.is_empty:
                        geometry_map[key] = geom
        print(f'Geometry layer {geo}: {sum(key in geometry_map for key in needed)}/{len(needed)}',flush=True)
    missing = sorted({r['osm_id'] for r in snapshot['records']}-geometry_map.keys())
    if missing and live_graph:
        from neo4j import Query
        with open_driver(config) as driver:
            with driver.session(database=config['database'],default_access_mode='READ') as session:
                for start in range(0,len(missing),500):
                    result = session.run(Query('''MATCH (f:OSMFeature)-[:HASGEOMETRY|HAS_GEOMETRY]->(g:Geometry)
                        WHERE f.osm_id IN $ids RETURN f.osm_id AS id,collect(DISTINCT g.geometry) AS geometries''',timeout=120),ids=missing[start:start+500])
                    for row in result:
                        values = [shapely.from_wkt(value) for value in row['geometries'] if value]
                        if values:
                            geometry_map[row['id']] = shapely.make_valid(unary_union(values))
    missing = sorted({r['osm_id'] for r in snapshot['records']}-geometry_map.keys())
    if missing:
        raise ValueError(f'{len(missing)} unresolved geometries (not silently skipped): {missing[:12]}')
    for key,geom in geometry_map.items():
        if geom.is_empty or not np.isfinite(geom.bounds).all():
            raise ValueError(f'Invalid geometry: {key}')
        west,south,east,north = geom.bounds
        if west < -180 or east > 180 or south < -90 or north > 90:
            raise ValueError(f'Geometry is not WGS84 longitude/latitude: {key}')
    return geometry_map,inputs


def setup_grid(config, region):
    if config.get('reference_raster'):
        with rasterio.open(config['reference_raster']) as src:
            if src.crs is None or src.count!=1:
                raise ValueError('Reference raster must be single-band and have a CRS.')
            affine,crs,height,width = src.transform,CRS.from_user_input(src.crs),src.height,src.width
            reference_valid = src.read_masks(1)>0
            reference_valid &= np.isfinite(src.read(1))
    else:
        height = width = int(config.get('grid_size',200))
        crs = CRS.from_epsg(4326)
        affine = from_bounds(*region.bounds,width,height)
        reference_valid = np.ones((height,width),dtype=bool)
    if width*height>4_000_000 or min(height,width)<2:
        raise ValueError('Grid must contain 2..4 million cells per configured dimensions.')
    if affine.b!=0 or affine.d!=0 or affine.a<=0 or affine.e>=0:
        raise ValueError('A north-up raster grid is required.')
    grid_region = transform(Transformer.from_crs(4326,crs,always_xy=True).transform,region)
    valid = rasterize([(grid_region,1)],out_shape=(height,width),transform=affine).astype(bool)&reference_valid
    if not valid.any():
        raise ValueError('Region and reference grid do not overlap.')
    metric_crs = CRS.from_user_input(config['distance_crs'])
    if not metric_crs.is_projected or any(abs(axis.unit_conversion_factor-1)>1e-8 for axis in metric_crs.axis_info):
        raise ValueError('distance_crs must be a locally appropriate projected CRS in metres.')
    rr,cc = np.indices((height,width))
    xx = affine.c+(cc+.5)*affine.a
    yy = affine.f+(rr+.5)*affine.e
    xx,yy = Transformer.from_crs(crs,metric_crs,always_xy=True).transform(xx,yy)
    valid_indices = np.flatnonzero(valid)
    centres = shapely.points(np.asarray(xx).ravel()[valid_indices],np.asarray(yy).ravel()[valid_indices])
    return dict(transform=affine,crs=crs,height=height,width=width,valid=valid,
                indices=valid_indices,centres=centres,tree=shapely.STRtree(centres),metric_crs=metric_crs)


def construct_fields(snapshot, geometries, grid, config):
    """One exact geometry-distance calculation per feature and cutoff group."""
    mode = aggregation_mode(config)
    strength = overlap_strength(config)
    geometry_kernels,exponential_rate = geometry_decay_settings(config)
    event_to_factors = defaultdict(set)
    for factor,events in snapshot['factor_to_disaster'].items():
        for event in events:
            event_to_factors[event].add(factor)
    weights = normalize_weights(config['event_hazard_weights'])
    geo_type_weights = config.get('geo_type_weight_by_type')
    if geo_type_weights is None:
        geo_type_weights = geo_type_weight_settings(config)['weights']
    n = len(grid['indices'])
    event_fields = {event_name(r['event_description']):np.zeros(n,dtype='float64') for r in snapshot['records']}
    factor_fields = {factor:np.zeros(n,dtype='float64') for factor in snapshot['factor_to_disaster']}
    event_peaks = {k:np.zeros(n) for k in event_fields} if mode=='log_overlap' else {}
    factor_peaks = {k:np.zeros(n) for k in factor_fields} if mode=='log_overlap' else {}
    by_feature = defaultdict(list)
    seen = set()
    for record in snapshot['records']:
        identity = (record['osm_id'],record['event_description'])
        if identity in seen:
            raise ValueError(f'Duplicate feature-event score in snapshot: {identity}')
        seen.add(identity)
        value = float(record['score'])
        if not math.isfinite(value) or value<=0:
            raise ValueError('Snapshot must contain strictly positive finite scores.')
        by_feature[record['osm_id']].append(record)
    project = Transformer.from_crs(4326,grid['metric_crs'],always_xy=True).transform
    active_features = 0
    distance_evaluations = 0
    geometry_kind_counts = defaultdict(int)
    geo_type_weight_counts = defaultdict(int)
    for i,(key,records) in enumerate(sorted(by_feature.items()),1):
        # Never replace lines/polygons with centroids or sampled points.
        geometry = shapely.make_valid(transform(project,geometries[key]))
        parts = geometry_parts_by_kind(geometry)
        for kind in parts:
            geometry_kind_counts[kind] += 1
        maximum_cutoff = max(cutoff_for(r['event_description'],r['geo_type'],config,kind)
                             for r in records for kind in parts)
        xmin,ymin,xmax,ymax = geometry.bounds
        candidates = grid['tree'].query(box(xmin-maximum_cutoff,ymin-maximum_cutoff,xmax+maximum_cutoff,ymax+maximum_cutoff))
        if not len(candidates):
            continue
        distances = {kind:shapely.distance(grid['centres'][candidates],part) for kind,part in parts.items()}
        inside = np.minimum.reduce(list(distances.values())) < maximum_cutoff
        candidates = candidates[inside]
        distances = {kind:d[inside] for kind,d in distances.items()}
        if not len(candidates):
            continue
        active_features += 1
        distance_evaluations += len(candidates)*len(parts)
        kernels = {}
        for record in records:
            desc,score = record['event_description'],record['score']
            geo_type = str(record['geo_type']).strip()
            geo_weight = geo_type_weights.get(geo_type.casefold(),1.)
            geo_type_weight_counts[geo_type] += 1
            component_kernels = []
            for kind,d in distances.items():
                cutoff = cutoff_for(desc,record['geo_type'],config,kind)
                cache_key = (kind,cutoff)
                if cache_key not in kernels:
                    kernels[cache_key] = distance_decay(d,cutoff,geometry_kernels[kind],exponential_rate)
                component_kernels.append(kernels[cache_key])
            kernel_values = component_kernels[0] if len(component_kernels)==1 else np.maximum.reduce(component_kernels)
            values = score*geo_weight*kernel_values
            event = event_fields[event_name(desc)]
            event[candidates] = np.maximum(event[candidates],values) if mode=='max' else event[candidates]+values
            if mode=='log_overlap':
                peak = event_peaks[event_name(desc)]
                peak[candidates] = np.maximum(peak[candidates],values)
            # Optional instance-specific scores preserve causal-path/task eligibility.
            factor_scores = record.get('factor_scores')
            if factor_scores is None:
                factor_scores = {factor:score for factor in event_to_factors.get(desc,())}
            for factor,factor_score in factor_scores.items():
                if factor not in factor_fields or not math.isfinite(factor_score) or not 0<=factor_score<=score:
                    raise ValueError(f'Invalid instance-specific factor score: {key}, {factor}')
                weighted = factor_score*geo_weight*kernel_values*hazard_weight(desc,weights)
                field = factor_fields[factor]
                field[candidates] = np.maximum(field[candidates],weighted) if mode=='max' else field[candidates]+weighted
                if mode=='log_overlap':
                    peak = factor_peaks[factor]
                    peak[candidates] = np.maximum(peak[candidates],weighted)
        if i%2000==0:
            print(f'Distance fields: {i}/{len(by_feature)} features',flush=True)
    if mode=='log_overlap':
        event_fields = {k:logarithmic_overlap(v,event_peaks[k],strength) for k,v in event_fields.items()}
        factor_fields = {k:logarithmic_overlap(v,factor_peaks[k],strength) for k,v in factor_fields.items()}
    return event_fields,factor_fields,dict(features=len(by_feature),active_features=active_features,
                                          distance_evaluations=distance_evaluations,geometry_kind_counts=dict(geometry_kind_counts),
                                          geo_type_weight_counts=dict(geo_type_weight_counts))


def add_scale_bar(ax, grid, config):
    """Draw an accurate metric scale bar at lower left for any grid CRS."""
    if not config.get('show_scale_bar', True):
        return
    _, length_km = display_scale_settings(config)
    west,south,east,north = rasterio.transform.array_bounds(grid['height'],grid['width'],grid['transform'])
    to_metric = Transformer.from_crs(grid['crs'],grid['metric_crs'],always_xy=True)
    from_metric = Transformer.from_crs(grid['metric_crs'],grid['crs'],always_xy=True)
    anchor_x = west + (east-west)*.065
    anchor_y = south + (north-south)*.075
    origin_x,origin_y = to_metric.transform(anchor_x,anchor_y)
    length_m = length_km*1000.
    tick_m = max(length_m*.045, 40.)

    def coordinate(dx=0.,dy=0.):
        return from_metric.transform(origin_x+dx,origin_y+dy)

    end_x,end_y = coordinate(length_m)
    start_x,start_y = coordinate()
    ax.plot([start_x,end_x],[start_y,end_y],color='white',lw=6.5,zorder=8,solid_capstyle='butt')
    ax.plot([start_x,end_x],[start_y,end_y],color='black',lw=2.2,zorder=9,solid_capstyle='butt')
    for fraction,label in ((0.,'0'),(.5,f'{length_km/2:g}'),(1.,f'{length_km:g} km')):
        upper_x,upper_y = coordinate(length_m*fraction,tick_m)
        lower_x,lower_y = coordinate(length_m*fraction,-tick_m)
        ax.plot([lower_x,upper_x],[lower_y,upper_y],color='white',lw=4,zorder=8)
        ax.plot([lower_x,upper_x],[lower_y,upper_y],color='black',lw=1.5,zorder=9)
        label_x,label_y = coordinate(length_m*fraction,-2.1*tick_m)
        ax.text(label_x,label_y,label,ha='center',va='top',fontsize=8.5,color='black',zorder=10,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.4))


def plot_field(values,grid,name,kind,path,config):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    fig,ax = plt.subplots(figsize=(8,8),layout='constrained')
    west,south,east,north = rasterio.transform.array_bounds(grid['height'],grid['width'],grid['transform'])
    extent = [west,east,south,north]
    if config.get('basemap'):
        if not grid['crs'].equals(CRS.from_epsg(4326),ignore_axis_order=True):
            raise ValueError('Unreferenced basemap overlay is supported only on a WGS84 grid.')
        ax.imshow(mpimg.imread(config['basemap']),extent=extent,origin='upper')
    masked = np.ma.array(values,mask=~grid['valid'])
    top = display_maximum_for_field(masked,kind,config)
    artist = ax.imshow(masked,extent=extent,origin='upper',cmap='YlOrRd',alpha=.68,
                       interpolation='nearest',vmin=0,vmax=top if top>0 else 1)
    if config.get('show_colorbar', True):
        fig.colorbar(artist,ax=ax,shrink=.75,label='Importance score (not probability)')
    if config.get('show_title', True):
        ax.set_title(f'{name}\nFeature-distance decay | {kind}',fontsize=13)
    if grid['crs'].is_geographic:
        ax.set_aspect(1/math.cos(math.radians((north+south)/2)))
    add_scale_bar(ax,grid,config)
    ax.set_axis_off()
    fig.savefig(path,dpi=config.get('dpi',180))
    plt.close(fig)


def export_fields(event_fields,factor_fields,grid,config,out):
    mode = aggregation_mode(config)
    geometry_kernels,exponential_rate = geometry_decay_settings(config)
    _,scale_bar_km = display_scale_settings(config)
    rows = []
    for kind,fields,folder in [('event',event_fields,out/'output_event_maps_grid'),
                               ('factor',factor_fields,out/'output_factor_maps_grid')]:
        tifs = folder/'geotiff' if kind=='event' else Path(str(folder)+'_geotiff')
        tifs.mkdir(parents=True,exist_ok=True)
        folder.mkdir(parents=True,exist_ok=True)
        names = set()
        for name,vector in sorted(fields.items()):
            if not np.isfinite(vector).all() or np.any(vector>np.finfo('float32').max):
                raise ValueError(f'Field cannot be represented as finite float32: {name}')
            key = safe_name(name,kind=='factor')
            if not key or key.casefold() in names:
                raise ValueError(f'Ambiguous output filename: {name}')
            names.add(key.casefold())
            values = np.full((grid['height'],grid['width']),-9999.,dtype='float32')
            if not np.isfinite(vector).all():
                raise ValueError(f'Nonfinite field: {name}')
            values.ravel()[grid['indices']] = vector
            display_maximum = display_maximum_for_field(np.ma.array(values,mask=~grid['valid']),kind,config)
            display_scope = config['display_scale_scope'] if config['apply_display_scale_maximum'] else 'per_field'
            suffix = '_importance.tif' if kind=='event' else '_importance_field.tif'
            target = tifs/(key+suffix)
            with rasterio.open(target,'w',driver='GTiff',height=grid['height'],width=grid['width'],count=1,
                               dtype='float32',crs=grid['crs'],transform=grid['transform'],nodata=-9999.,compress='deflate') as dst:
                dst.write(values,1)
                dst.update_tags(field_name=name,field_kind=kind,method='feature_distance_decay',
                                kernel='geometry_specific',geometry_kernels=json.dumps(geometry_kernels,sort_keys=True),
                                geometry_cutoff_m=json.dumps(config.get('geometry_cutoff_m',{}),sort_keys=True),
                                exponential_decay_rate=str(exponential_rate),distance_crs=grid['metric_crs'].to_string(),
                                default_cutoff_m=str(config['default_cutoff_m']),aggregation=mode,
                                overlap_strength=str(overlap_strength(config)),spatial_formula=spatial_formula(config),
                                value_semantics='importance score; not a probability',
                                geo_type_weighting='enabled' if config.get('geo_type_weight_by_type') else 'disabled',
                                geo_type_weight_count=str(len(config.get('geo_type_weight_by_type',{}))),
                                display_scale_scope=display_scope,
                                display_maximum=str(display_maximum),
                                scale_bar_km=str(scale_bar_km),
                                cutoff_overrides=json.dumps({k:config.get(k,{}) for k in ('event_cutoff_m','geo_type_cutoff_m','event_geo_cutoff_m')},ensure_ascii=False))
            png = folder/(key+'_field.png')
            if config.get('render_png',True):
                plot_field(values,grid,name,kind,png,config)
            rows.append(dict(kind=kind,field=name,geotiff=str(target.resolve()),
                             image=str(png.resolve()) if config.get('render_png',True) else None,
                             min_score=float(vector.min()),max_score=float(vector.max()),
                             display_maximum=display_maximum,display_scale_scope=display_scope,
                             positive_cells=int((vector>0).sum()),valid_cells=len(vector)))
            print(f'Exported {kind}: {name}',flush=True)
    return rows


def run(config_path, snapshot_path=None, output_dir=None, no_plots=False, database=None):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    if database:
        config['database'] = database
    for key in ('source_module','connection_module','features_folder','region','reference_raster','basemap','output_dir','geo_type_dictionary'):
        if config.get(key):
            config[key] = str((config_path.parent/config[key]).resolve())
    if output_dir:
        config['output_dir'] = str(Path(output_dir).resolve())
    if not config.get('database'):
        config['database'] = literal_setting(config['source_module'],'ins_graph_database')
    if config.get('event_hazard_weights') is None:
        config['event_hazard_weights'] = literal_setting(config['source_module'],'DISASTER_PHENOMENON_HAZARD_WEIGHTS')
    if not config.get('apply_geo_type_importance',True):
        config['geo_type_importance_source'] = 'disabled by configuration'
    elif config.get('geo_type_importance') is None:
        if config.get('source_module'):
            config['geo_type_importance'] = literal_setting(config['source_module'],'observation_importance')
            config['geo_type_importance_source'] = 'instantiation_en.py:observation_importance'
        else:
            config['geo_type_importance'] = {}
            config['geo_type_importance_source'] = 'not configured for offline snapshot'
    else:
        config['geo_type_importance_source'] = 'configuration'
    normalize_weights(config['event_hazard_weights'])
    geo_type_importance = geo_type_weight_settings(config)
    config['spatial_aggregation'] = aggregation_mode(config)
    config['overlap_strength'] = overlap_strength(config)
    geometry_decay_settings(config)
    display_scale_settings(config)
    cell_histogram_settings(config)
    config.setdefault('exclude_geo_types',['Administrative Region'])
    config.setdefault('path_boost',1.05)
    if not math.isfinite(config['path_boost']) or config['path_boost']<=0:
        raise ValueError('path_boost must be finite and positive.')
    cutoff_for('', '', config)
    config['render_png'] = config.get('render_png',True) and not no_plots
    out = Path(config['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty; select --output-dir for a new run: {out}')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'run_status.json',{'status':'running','started_at':datetime.now(timezone.utc).isoformat()})
    try:
        snapshot = read_json(snapshot_path) if snapshot_path else fetch_graph(config)
        if snapshot.get('schema_version')!=1:
            raise ValueError('Snapshot schema_version must be 1.')
        if snapshot_path and not snapshot.get('initial_events_included'):
            print('Legacy snapshot: initial event inclusion is unverified. Read Neo4j again to include all eligible roots.',flush=True)
        print(f'Loaded {len(snapshot["records"])} scored feature-event records',flush=True)
        if not snapshot_path:
            write_json(out/'graph_scores.json',snapshot)
        geometries,geometry_inputs = resolve_geometries(snapshot,config,live_graph=not snapshot_path)
        snapshot['geometries'] = {key:mapping(geom) for key,geom in geometries.items()}
        snapshot['geometry_inputs'] = geometry_inputs
        snapshot_file = out/'source_snapshot.json'
        write_json(snapshot_file,snapshot)
        region = region_geometry(config['region'])
        grid = setup_grid(config,region)
        events,factors,statistics = construct_fields(snapshot,geometries,grid,config)
        display_scale = set_display_scale(events,factors,config)
        if config['render_png']:
            from plot_style import configure_plot_style
            configure_plot_style()
        outputs = export_fields(events,factors,grid,config,out)
        cell_histogram = export_cell_importance_histogram(factors,grid,config,out)
        manifest = dict(config=config,source_database=snapshot.get('database'),display_scale=display_scale,
                        cell_importance_histogram=cell_histogram,
                        initial_events_included=snapshot.get('initial_events_included',False),
                        initial_event_count=snapshot.get('initial_event_count'),
                        geo_type_importance=geo_type_importance,
                        captured_at=snapshot.get('captured_at'),statistics=statistics,
                        score_formula='max(probability * importance) * path_boost ** (n_paths - 1)',
                        spatial_formula=spatial_formula(config),
                        cutoff_status='initial configurable scale, not calibrated on instantiation_validation points',
                        source_snapshot=str(snapshot_file.resolve()),snapshot_sha256=file_hash(snapshot_file),
                        region_sha256=file_hash(config['region']),
                        reference_sha256=file_hash(config['reference_raster']) if config.get('reference_raster') else None,
                        raster_crs=grid['crs'].to_string(),distance_crs=grid['metric_crs'].to_string(),
                        grid_shape=[grid['height'],grid['width']],outputs=outputs,
                        limitations=['Euclidean geometry distance, no hydrodynamic routing or barriers',
                                     'Source scores and configured hazard weights are not probabilities',
                                     'Overlap enhancement depends on feature segmentation and correlated event contributions',
                                     'Raster values are evaluated at pixel centres; very narrow features may lie between centres'])
        write_json(out/'manifest.json',manifest)
        write_json(out/'run_status.json',{'status':'complete','events':len(events),'factors':len(factors)})
        print(f'Complete: {len(events)} event fields, {len(factors)} factor fields -> {out}',flush=True)
        return manifest
    except Exception as exc:
        # Never save a connection URI/password in reports.
        write_json(out/'run_status.json',{'status':'failed','error_type':type(exc).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('instantiation_config.json'))
    parser.add_argument('--snapshot',type=Path,help='Offline self-contained source_snapshot.json or graph_scores.json plus original local geometry files')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--no-plots',action='store_true',help='Export GeoTIFFs without PNGs')
    args = parser.parse_args()
    run(args.config,args.snapshot,args.output_dir,args.no_plots)


if __name__=='__main__':
    main()
