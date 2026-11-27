"""Map OELG Neo4j instances with depth-only scores and geometry distance decay.

python plot_oelg_distance_fields.py --config oelg_distance_config.json
The instance graph is authoritative; local exports are never input fallbacks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import mapping

import distance_fields as fields
from standard_observation_factors import canonical_factors
from oelg_graph_store import load_graph

SCORE_FORMULA = '100 * depth_decay ** (depth - 1) for eligible depth >= 1; no temporal or path-count weighting'


def validate_depth_decay(value):
    value = float(value)
    if not 0<value<1:
        raise ValueError('depth_decay must be finite and strictly between 0 and 1.')
    return value


def score_events(events, depth_decay=.5):
    """Discount only causal depth; immediate/delayed labels do not affect scores."""
    depth_decay = validate_depth_decay(depth_decay)
    nodes = {}
    for feature in events['features']:
        node = feature['properties']
        key = node['id']
        if key in nodes:
            raise ValueError(f'Duplicate event ID: {key}')
        depth = node['current_depth']
        if isinstance(depth,bool) or not isinstance(depth,int) or depth<0:
            raise ValueError(f'Invalid event depth: {key}')
        nodes[key] = node
    if not nodes or sum(n['parent_id'] is None for n in nodes.values())!=1:
        raise ValueError('Require one root and a nonempty OELG event tree.')
    scored = {}
    for key,node in sorted(nodes.items(),key=lambda pair:(pair[1]['current_depth'],pair[0])):
        depth = node['current_depth']
        parent_id = node['parent_id']
        if parent_id is None:
            if depth!=0 or node['relation_type'] is not None:
                raise ValueError(f'Invalid root event: {key}')
            delayed,immediate = 0,0
            path = [node['phenomenon']]
        else:
            if parent_id not in scored or depth!=scored[parent_id]['depth']+1:
                raise ValueError(f'Missing parent, cycle or inconsistent depth: {key}')
            relation = node['relation_type']
            relation = {'immediate': 'Immediate', 'altering': 'Altering'}.get(str(relation).casefold(), relation)
            if relation not in ('Immediate','Altering'):
                raise ValueError(f'Unknown causal relation: {relation}')
            parent = scored[parent_id]
            delayed = parent['delayed_edges']+int(relation=='Altering')
            immediate = parent['immediate_edges']+int(relation=='Immediate')
            path = parent['path']+[node['phenomenon']]
        if node.get('path') is not None and node['path']!=path:
            raise ValueError(f'Event path disagrees with parent chain: {key}')
        depth_weight = depth_decay**max(0,depth-1)
        if depth_weight<=0:
            raise ValueError(f'Depth score underflow: {key}')
        scored[key] = dict(event_id=key,phenomenon=node['phenomenon'],parent_id=parent_id,
                           depth=depth,relation_type=None if parent_id is None else relation,path=path,
                           immediate_edges=immediate,delayed_edges=delayed,
                           depth_weight=depth_weight,score=100.*depth_weight)
    return scored


def build_snapshot(scored, affected_rows, tasks, exclude_types=()):
    """Deduplicate paths per feature/event; keep factor scores path-specific."""
    tasks_by_event = defaultdict(list)
    for task in tasks:
        key = task['disaster_id']
        if key not in scored:
            raise ValueError(f'Observation refers to unknown event: {key}')
        tasks_by_event[key].append(dict(task,factors=canonical_factors(task.get('factors',[]))))
    factor_cache = {}
    records,paths = {},defaultdict(set)
    factor_to_events = defaultdict(set)
    seen = set()
    total_rows = 0
    for row in affected_rows:
        total_rows += 1
        key,geo,feature_id = row['event_id'],row['geo_type'],row['osm_id']
        if key not in scored:
            raise ValueError(f'Feature refers to unknown event: {key}')
        event = scored[key]
        if int(row['depth'])!=event['depth'] or row['phenomenon']!=event['phenomenon']:
            raise ValueError(f'Feature/event depth or phenomenon mismatch: {key}')
        eligible = int(row['eligible'])
        if eligible not in (0,1):
            raise ValueError(f'Invalid eligibility: {key}')
        if not eligible or geo in exclude_types or event['depth']==0:
            continue
        if not feature_id.startswith(geo+'_'):
            raise ValueError(f'Feature ID does not match geography type: {feature_id}')
        row_key = (key,geo,feature_id)
        if row_key in seen:
            continue
        seen.add(row_key)
        desc = f'{geo},{event["phenomenon"]}'
        identity = (feature_id,desc)
        score = event['score']
        if identity not in records:
            records[identity] = dict(osm_id=feature_id,geo_type=geo,event_description=desc,
                                     score=score,best_event_id=key,best_depth=event['depth'],factor_scores={})
        record = records[identity]
        paths[identity].add(key)
        if (score,key)>(record['score'],record['best_event_id']):
            record.update(score=score,best_event_id=key,best_depth=event['depth'])
        cache_key = (key,geo)
        if cache_key not in factor_cache:
            factor_cache[cache_key] = {factor for task in tasks_by_event[key]
                                      if not task.get('geo_types') or geo in task['geo_types']
                                      for factor in task.get('factors',[]) if factor}
        for factor in factor_cache[cache_key]:
            record['factor_scores'][factor] = max(record['factor_scores'].get(factor,0.),score)
            factor_to_events[factor].add(desc)
    if not records:
        raise ValueError('No eligible non-root feature-event records.')
    for identity,record in records.items():
        record['n_paths'] = len(paths[identity])
    return dict(schema_version=1,source='neo4j_oelg_instances',
                captured_at=datetime.now(timezone.utc).isoformat(),
                records=[records[k] for k in sorted(records)],
                factor_to_disaster={k:sorted(v) for k,v in sorted(factor_to_events.items())},
                statistics=dict(affected_rows=total_rows,unique_instance_feature_rows=len(seen),
                                feature_event_records=len(records),event_instances=len(scored)))


def load_config(path):
    path = Path(path).resolve()
    config = fields.read_json(path)
    config['depth_decay'] = validate_depth_decay(config.get('depth_decay',.5))
    for key in ('oelg_config','map_config','output_dir'):
        config[key] = str((path.parent/config[key]).resolve())
    map_path = Path(config['map_config'])
    maps = fields.read_json(map_path)
    for key in ('source_module','connection_module','features_folder','region','reference_raster','basemap','geo_type_dictionary'):
        if maps.get(key):
            maps[key] = str((map_path.parent/maps[key]).resolve())
    # Depth is the only source-score weight; no additional hazard/time weighting.
    # This top-level setting intentionally overrides the shared DOELG map config.
    # It lets the two methods keep independent colour-scale maxima.
    if 'display_scale_maximum_by_kind' in config:
        maps['display_scale_maximum_by_kind'] = config['display_scale_maximum_by_kind']
    # OELG uses the same geographic-type dictionary as DOELG, but its source
    # weights can be independently overridden in oelg_distance_config.json.
    geo_weighting_enabled = config.get('apply_geo_type_importance',True)
    if not isinstance(geo_weighting_enabled,bool):
        raise ValueError('apply_geo_type_importance must be true or false.')
    maps['apply_geo_type_importance'] = geo_weighting_enabled
    if 'geo_type_importance' in config:
        maps['geo_type_importance'] = config['geo_type_importance']
    if not geo_weighting_enabled:
        maps['geo_type_importance_source'] = 'disabled by configuration'
    elif maps.get('geo_type_importance') is None and maps.get('source_module'):
        maps['geo_type_importance'] = fields.literal_setting(maps['source_module'],'observation_importance')
        maps['geo_type_importance_source'] = 'instantiation_en.py:observation_importance'
    elif maps.get('geo_type_importance') is None:
        maps['geo_type_importance'] = {}
        maps['geo_type_importance_source'] = 'not configured for offline snapshot'
    else:
        maps['geo_type_importance_source'] = 'configuration'
    fields.geo_type_weight_settings(maps)
    maps['event_hazard_weights'] = {}
    maps['spatial_aggregation'] = fields.aggregation_mode(maps)
    maps['overlap_strength'] = fields.overlap_strength(maps)
    fields.geometry_decay_settings(maps)
    maps['output_dir'] = config['output_dir']
    fields.normalize_weights(maps['event_hazard_weights'])
    fields.cutoff_for('','',maps)
    oelg_path = Path(config['oelg_config'])
    oelg = fields.read_json(oelg_path)
    for key in ('abstract_database', 'observation_database', 'instance_database'):
        if key in oelg:
            oelg[key] = fields.resolve_database_name(oelg[key])
    for key in ('region',):
        oelg[key] = str((oelg_path.parent/oelg[key]).resolve())
    maps['region'] = oelg['region']
    maps['exclude_geo_types'] = oelg.get('exclude_geo_types',[])
    return config,maps,oelg


def write_csv(path, rows, columns):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def run(config_path, output_dir=None, no_plots=False, *, oelg_override=None, only_kind=None, grid_overrides=None):
    config,maps,oelg = load_config(config_path)
    if oelg_override is not None:
        oelg = oelg_override
        maps['region'] = oelg['region']
        maps['exclude_geo_types'] = oelg.get('exclude_geo_types', [])
    if grid_overrides is not None:
        if not set(grid_overrides) <= {'reference_raster', 'region', 'distance_crs'}:
            raise ValueError('Grid overrides may change only reference_raster, region and distance_crs.')
        maps.update(grid_overrides)
    if only_kind not in (None, 'event', 'factor'):
        raise ValueError('only_kind must be event or factor')
    if output_dir:
        config['output_dir'] = maps['output_dir'] = str(Path(output_dir).resolve())
    maps['render_png'] = maps.get('render_png',True) and not no_plots
    out = Path(config['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty; choose --output-dir: {out}')
    out.mkdir(parents=True,exist_ok=True)
    fields.write_json(out/'run_status.json',{'status':'running'})
    try:
        graph_events, affected_rows, tasks, geometries, provenance = load_graph(oelg, maps)
        scored = score_events(graph_events, config['depth_decay'])
        snapshot = build_snapshot(scored, affected_rows, tasks, maps['exclude_geo_types'])
        snapshot['graph_source'] = provenance
        geometry_inputs = dict(source='neo4j', crs='EPSG:4326', **{k: provenance[k] for k in ('database','snapshot','run_signature')})
        print(f'OELG: {len(scored)} instances -> {len(snapshot["records"])} depth-weighted feature-event records',flush=True)
        fields.write_json(out/'event_scores.json',scored)
        write_csv(out/'event_scores.csv',scored.values(),['event_id','phenomenon','parent_id','depth',
                  'relation_type','immediate_edges','delayed_edges','depth_weight','score'])
        write_csv(out/'feature_scores.csv',snapshot['records'],['osm_id','geo_type','event_description',
                  'score','best_event_id','best_depth','n_paths'])
        snapshot.update(geometries={k:mapping(v) for k,v in geometries.items()},geometry_inputs=geometry_inputs,
                        scoring_config=config,score_formula=SCORE_FORMULA)
        fields.write_json(out/'source_snapshot.json',snapshot)
        grid = fields.setup_grid(maps,fields.region_geometry(maps['region']))
        events,factors,stats = fields.construct_fields(snapshot,geometries,grid,maps)
        if only_kind == 'event':
            factors = {}
        elif only_kind == 'factor':
            events = {}
        display_scale = fields.set_display_scale(events,factors,maps)
        if maps['render_png']:
            from evaluate_importance_fields import configure_plot_style
            configure_plot_style()
        outputs = fields.export_fields(events,factors,grid,maps,out)
        import rasterio
        for item in outputs:
            with rasterio.open(item['geotiff'],'r+') as dst:
                dst.update_tags(source_method='oelg_depth_only',score_formula=SCORE_FORMULA,
                                secondary_feature_score='100',depth_decay=str(config['depth_decay']),
                                temporal_weighting='none', source_database=provenance['database'],
                                source_snapshot=provenance['snapshot'], source_run_signature=provenance['run_signature'])
        manifest = dict(config=config,map_config=maps,graph_source=provenance,display_scale=display_scale,
                        geometry_inputs=geometry_inputs,statistics={**snapshot['statistics'],**stats},
                        score_formula=SCORE_FORMULA,spatial_formula=fields.spatial_formula(maps),
                        path_aggregation='maximum per feature/event and per eligible factor; no path-count boost',
                        source_snapshot=str((out/'source_snapshot.json').resolve()),
                        region_sha256=fields.file_hash(maps['region']),
                        reference_sha256=fields.file_hash(maps['reference_raster']) if maps.get('reference_raster') else None,
                        grid_shape=[grid['height'],grid['width']],raster_crs=grid['crs'].to_string(),outputs=outputs,
                        limitations=['Configured depth priority applied to original eligibility score 100; not calibrated probability',
                                     'Immediate/delayed relations do not affect scores; temporal metadata is retained only for auditing',
                                     'Depth ordering applies to source scores; spatial overlap can change field ordering',
                                     'Euclidean full-geometry distance; original OELG eligibility retained, no hydrodynamic routing',
                                     'Existing OELG environment constraints may use proxy mode; not corrected by this mapping step'])
        fields.write_json(out/'manifest.json',manifest)
        fields.write_json(out/'run_status.json',{'status':'complete','events':len(events),'factors':len(factors)})
        print(f'Complete: {len(events)} event maps, {len(factors)} factor maps -> {out}',flush=True)
        return manifest
    except Exception as exc:
        fields.write_json(out/'run_status.json',{'status':'failed','error_type':type(exc).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('oelg_distance_config.json'))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--no-plots',action='store_true')
    args = parser.parse_args()
    run(args.config,args.output_dir,args.no_plots)


if __name__=='__main__':
    main()
