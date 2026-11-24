"""Evaluate exported importance GeoTIFFs against observed disaster points.

Run: python evaluate_importance_fields.py [--config importance_evaluation_config.json]
Requires requirements_importance_evaluation.txt; does not import instantiation_en
or connect to Neo4j. Configuration paths are relative to the configuration file.

H(a) is a hit-area / prediction-rate curve, NOT precision-recall. Equal scores
are grouped; within a group the curve uses area-proportional random selection
in expectation (linear interpolation). Full-group threshold metrics are also
reported, so a fractional tie hit is never represented as an observed integer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import html
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
import shapefile
from pyproj import CRS, Transformer


@dataclass
class Curve:
    scores: np.ndarray  # descending, distinct score levels
    area: np.ndarray  # includes origin
    hit: np.ndarray  # includes origin
    cells: np.ndarray
    point_count: np.ndarray
    auc: float
    n_points: int


def canonical_point_set(name):
    """Resolve historical point-set identifiers to the canonical Subsidence name."""
    text = str(name).strip()
    return 'Subsidence' if text.casefold() in ('collapse', 'subsidence') else text


def normalize_point_set_config(config):
    """Use canonical point-set names consistently in selections and event references."""
    normalized = deepcopy(config)
    for key in ('point_sets', 'factor_fields_by_point_set'):
        if key not in normalized:
            continue
        values = {}
        for name, value in normalized[key].items():
            canonical = canonical_point_set(name)
            if canonical in values:
                raise ValueError(f'Duplicate point-set aliases for {canonical}.')
            values[canonical] = value
        normalized[key] = values
    for pair in normalized.get('event_pairs', []):
        pair['points'] = canonical_point_set(pair['points'])
    return normalized


def field_point_label(field, point_set, separator=' | '):
    """Label a field and validation set without repeating an identical hazard name."""
    point_set = canonical_point_set(point_set)
    key = lambda value: ''.join(character for character in value if character.isalnum()).casefold()
    return field if key(field) == key(point_set) else f'{field}{separator}{point_set}'


def build_curve(scores, areas, point_scores) -> Curve:
    """Exact score groups, weighted area, individual point counts (no dedup)."""
    scores = np.asarray(scores, dtype=float).ravel()
    areas = np.asarray(areas, dtype=float).ravel()
    point_scores = np.asarray(point_scores, dtype=float).ravel()
    if not len(scores) or not len(point_scores) or scores.shape != areas.shape:
        raise ValueError('Need valid cells, matching area weights, and >=1 point.')
    if not all(np.isfinite(x).all() for x in (scores, areas, point_scores)) or np.any(areas <= 0):
        raise ValueError('Scores must be finite and cell areas strictly positive.')
    levels, inverse = np.unique(scores, return_inverse=True)
    bins = np.searchsorted(levels, point_scores)
    if np.any(bins == len(levels)) or not np.array_equal(levels[bins], point_scores):
        raise ValueError('Point scores must be sampled from valid raster cells.')
    weights = np.bincount(inverse, weights=areas, minlength=len(levels))[::-1]
    cells = np.bincount(inverse, minlength=len(levels))[::-1]
    counts = np.bincount(bins, minlength=len(levels))[::-1]
    area = np.r_[0., np.cumsum(weights) / weights.sum()]
    hit = np.r_[0., np.cumsum(counts) / len(point_scores)]
    area[-1] = hit[-1] = 1.
    auc = float(np.sum(np.diff(area) * (hit[1:] + hit[:-1]) / 2))
    return Curve(levels[::-1], area, hit, cells, counts, auc, len(point_scores))


def budget_metrics(curve: Curve, fraction: float) -> dict:
    if not 0 < fraction <= 1:
        raise ValueError('Area fraction must be in (0, 1].')
    j = max(1, int(np.searchsorted(curve.area, fraction, side='left')))
    partial = (fraction-curve.area[j-1])/(curve.area[j]-curve.area[j-1])
    hr = float(np.interp(fraction, curve.area, curve.hit))
    return dict(target_AF=fraction, HR=hr, Lift=hr/fraction,
                expected_hits=hr*curve.n_points, score_threshold=float(curve.scores[j-1]),
                boundary_selection_fraction=float(partial),
                strict_AF=float(curve.area[j-1]), strict_HR=float(curve.hit[j-1]),
                inclusive_AF=float(curve.area[j]), inclusive_HR=float(curve.hit[j]))


def area_at_hit(curve: Curve, target: float) -> tuple[float, float]:
    """Smallest interpolated area and smallest whole-score-group area."""
    if not 0 < target <= 1:
        raise ValueError('Hit target must be in (0, 1].')
    j = int(np.searchsorted(curve.hit, target, side='left'))
    fraction = (target-curve.hit[j-1])/(curve.hit[j]-curve.hit[j-1])
    return (float(curve.area[j-1]+fraction*(curve.area[j]-curve.area[j-1])),
            float(curve.area[j]))


def threshold_metrics(curve: Curve, threshold: float) -> dict:
    if not math.isfinite(threshold):
        raise ValueError('Score threshold must be finite.')
    n = int(np.count_nonzero(curve.scores >= threshold))
    af, hr = float(curve.area[n]), float(curve.hit[n])
    return dict(threshold=threshold, AF=af, HR=hr,
                Lift=hr/af if af else None, N_hit=int(curve.point_count[:n].sum()),
                N_all=curve.n_points)


def read_points(path: Path, exclude_status=()) -> dict:
    prj = path.with_suffix('.prj')
    if not prj.exists():
        raise ValueError(f'Missing CRS: {prj}')
    crs = CRS.from_wkt(prj.read_text(encoding='utf-8-sig'))
    cpg = path.with_suffix('.cpg')
    encoding = cpg.read_text(encoding='ascii').strip() if cpg.exists() else 'utf-8'
    if encoding == '65001':
        encoding = 'utf-8'
    points = []
    with shapefile.Reader(str(path), encoding=encoding) as reader:
        fields = [f[0] for f in reader.fields[1:]]
        for i, record in enumerate(reader.iterShapeRecords(), 1):
            if record.shape.shapeType not in (1, 11, 21) or len(record.shape.points) != 1:
                raise ValueError(f'{path}: record {i} is not a single Point.')
            x, y = record.shape.points[0]
            if not np.isfinite([x, y]).all():
                raise ValueError(f'{path}: nonfinite coordinate at record {i}.')
            if crs.is_geographic and not (-180 <= x <= 180 and -90 <= y <= 90):
                raise ValueError(f'{path}: coordinates are not longitude/latitude despite the CRS; use the converted WGS84 SHP.')
            attributes = {key.upper(): value for key, value in zip(fields, record.record)}
            status = str(attributes.get('STATUS', ''))
            points.append(dict(record_no=i, source_id=attributes.get('POINT_ID', attributes.get('FID')),
                               x=x, y=y, digitization_status=status,
                               excluded_by_filter=status in exclude_status))
    if not points:
        raise ValueError(f'Empty point layer: {path}')
    return dict(path=str(path.resolve()), crs=crs, points=points)


@lru_cache(maxsize=8)
def _areas(crs_wkt: str, affine_values: tuple, height: int, width: int, mode: str):
    from affine import Affine
    transform = Affine(*affine_values)
    if mode == 'pixel':
        return np.ones((height, width), dtype=float)
    if mode != 'geodesic':
        raise ValueError('area_mode must be geodesic or pixel.')
    crs = CRS.from_wkt(crs_wkt)
    to_wgs84 = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    geod = CRS.from_epsg(4326).get_geod()
    weights = np.empty((height, width), dtype=float)
    # In WGS84 north-up grids each cell in one row has equal ellipsoidal area.
    row_constant = crs.equals(CRS.from_epsg(4326), ignore_axis_order=True) and transform.b == transform.d == 0
    for row in range(height):
        for col in range(1 if row_constant else width):
            corners = [transform*(col,row), transform*(col+1,row),
                       transform*(col+1,row+1), transform*(col,row+1)]
            lon, lat = to_wgs84.transform(*zip(*corners))
            area = abs(geod.polygon_area_perimeter(lon, lat)[0])
            if not math.isfinite(area) or area <= 0:
                raise ValueError('Invalid physical cell area; verify the raster CRS and transform.')
            if row_constant:
                weights[row, :] = area
            else:
                weights[row, col] = area
    return weights


def evaluate_raster(path: Path, point_data: dict, area_mode='geodesic', outside_policy='error'):
    if outside_policy not in ('error', 'exclude'):
        raise ValueError('outside_policy must be error or exclude.')
    with rasterio.open(path) as src:
        if src.crs is None or src.count != 1:
            raise ValueError(f'{path}: a single-band, georeferenced raster is required.')
        grid = src.read(1, masked=True)
        data = np.asarray(grid.data, dtype=float)
        valid = ~np.ma.getmaskarray(grid) & np.isfinite(data)
        if not valid.any():
            raise ValueError(f'{path}: no valid raster cells.')
        # Zero is valid unless the file explicitly declares it NoData.
        areas = _areas(src.crs.to_wkt(), tuple(src.transform)[:6], src.height, src.width, area_mode)
        transformer = Transformer.from_crs(point_data['crs'], src.crs, always_xy=True)
        sampled = []
        audit = []
        inverse = ~src.transform
        for point in point_data['points']:
            entry = dict(point)
            entry.update(raster_x=None, raster_y=None, row=None, col=None, score=None)
            if point['excluded_by_filter']:
                entry['evaluation_status'] = 'filtered'
            else:
                x, y = transformer.transform(point['x'], point['y'])
                entry.update(raster_x=x, raster_y=y)
                if not np.isfinite([x, y]).all():
                    entry['evaluation_status'] = 'transform_failed'
                else:
                    col_float, row_float = inverse*(x, y)
                    # Include exact outer edges in the adjacent border cell.
                    tol = 1e-8
                    if not (-tol <= col_float <= src.width+tol and -tol <= row_float <= src.height+tol):
                        entry['evaluation_status'] = 'outside'
                    else:
                        col = min(src.width-1, max(0, int(math.floor(col_float))))
                        row = min(src.height-1, max(0, int(math.floor(row_float))))
                        entry.update(row=row, col=col)
                        if not valid[row,col]:
                            entry['evaluation_status'] = 'nodata'
                        else:
                            entry.update(score=float(data[row,col]), evaluation_status='valid')
                            sampled.append(data[row,col])
            audit.append(entry)
        invalid = [p for p in audit if p['evaluation_status'] not in ('valid','filtered')]
        if invalid and outside_policy == 'error':
            counts = {k:sum(p['evaluation_status']==k for p in invalid) for k in ('outside','nodata','transform_failed')}
            raise ValueError(f'{path.name}: {counts}; fix alignment or explicitly set outside_policy="exclude". No silent point removal.')
        curve = build_curve(data[valid], areas[valid], sampled)
        signature = hashlib.sha256(src.crs.to_wkt().encode()+str(tuple(src.transform)).encode()
                                   +str(valid.shape).encode()+valid.tobytes()).hexdigest()
        metadata = dict(raster=str(path.resolve()), point_file=point_data['path'],
                        raster_crs=src.crs.to_string(), bounds=list(src.bounds),
                        width=src.width, height=src.height, nodata=src.nodata,
                        area_mode=area_mode, domain_hash=signature,
                        sample_hash=hashlib.sha256(json.dumps(
                            [(p['record_no'],p['x'],p['y']) for p in audit if p['evaluation_status']=='valid'],
                            separators=(',', ':')).encode()).hexdigest(),
                        valid_cells=int(valid.sum()), zero_cells=int((data[valid]==0).sum()),
                        distinct_scores=len(curve.scores), constant_field=len(curve.scores)==1,
                        min_score=float(curve.scores[-1]), max_score=float(curve.scores[0]),
                        total_area=float(areas[valid].sum()), area_unit='m2' if area_mode=='geodesic' else 'cells',
                        source_points=len(audit), used_points=len(sampled),
                        filtered_points=sum(p['evaluation_status']=='filtered' for p in audit),
                        outside_points=sum(p['evaluation_status']=='outside' for p in audit),
                        nodata_points=sum(p['evaluation_status']=='nodata' for p in audit),
                        transform_failed_points=sum(p['evaluation_status']=='transform_failed' for p in audit),
                        occupied_cells=len({(p['row'],p['col']) for p in audit if p['evaluation_status']=='valid'}))
    return curve, audit, metadata


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slug(text: str):
    return re.sub(r'[^\w.-]+', '_', text).strip('_')


def configure_plot_style():
    """Register CJK fonts explicitly for Chinese factor labels and PDF export."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    for filename in ('msyh.ttc', 'simhei.ttf'):
        path = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/filename
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
    matplotlib.rcParams['font.sans-serif'] = [name for name in preferred if name in available]+['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['pdf.fonttype'] = 42


def plot_curve(curve, metrics, title, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 5.3), layout='constrained')
    ax.plot(curve.area*100, curve.hit*100, color='#246ca6', lw=2,
            label=f'Hit-area curve (AUC = {curve.auc:.3f})')
    ax.plot([0,100], [0,100], '--', color='#888888', lw=1.2, label='Random selection (AUC = 0.5)')
    hr10 = metrics['HR_at_10pct']
    area80 = metrics['Area_at_80pct_hit']
    ax.plot([10,10,0], [0,hr10*100,hr10*100], ':', color='#c77713')
    ax.scatter([10,area80*100], [hr10*100,80], color=['#c77713','#663d89'], s=28, zorder=3)
    ax.text(.97,.05, f"HR@10% = {hr10:.1%}\nLift@10% = {metrics['Lift_at_10pct']:.2f}\nArea@80%Hit = {area80:.1%}\nN = {metrics['used_points']}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='#dddddd', alpha=.95))
    ax.set(xlim=(0,100), ylim=(0,100), xlabel='Study area selected (%)',
           ylabel='Observed points captured (%)', title=title)
    ax.grid(alpha=.18)
    ax.legend(loc='upper left', fontsize=9)
    fig.savefig(path.with_suffix('.png'), dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def plot_summary(rows, point_name, path):
    import matplotlib.pyplot as plt
    selected = sorted([r for r in rows if r['point_set']==point_name], key=lambda r:r['AUC_PA'])
    labels = [f"{r['kind']}: {r['field']}" for r in selected]
    if len({r['run'] for r in selected}) > 1:
        labels = [f"{r['run']} | {label}" for r,label in zip(selected,labels)]
    fig, axes = plt.subplots(1,4,figsize=(16,max(5,len(selected)*.3)),sharey=True,layout='constrained')
    for ax,key,title,base in zip(axes,
            ['AUC_PA','HR_at_10pct','Lift_at_10pct','Area_at_80pct_hit'],
            ['AUC_PA (higher better)','HR@10% (higher better)','Lift@10% (higher better)','Area@80%Hit (lower better)'],
            [.5,.1,1,.8]):
        values = [r[key] for r in selected]
        ax.barh(range(len(selected)),values,color=['#c77713' if r['kind']=='event' else '#246ca6' for r in selected])
        ax.axvline(base,ls='--',color='#888888',lw=1)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='x',alpha=.2)
        if key!='Lift_at_10pct':
            ax.set_xlim(0,1)
    axes[0].set_yticks(range(len(selected)),labels,fontsize=8)
    fig.suptitle(f'{point_name}: importance fields vs observed points\nDashed lines: random selection',fontsize=12)
    fig.savefig(path.with_suffix('.png'),dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def render_report(rows, config, out, warnings):
    table_rows = []
    for r in rows:
        name = html.escape(f"{r['run']} / {r['kind']} / {r['field']}")
        if r['constant_field']:
            name += ' <strong>（常数场，无空间区分能力）</strong>'
        table_rows.append(f'<tr><td><a href="{r["result_dir"]}/curve.png">{name}</a></td>'
                          f'<td>{html.escape(canonical_point_set(r["point_set"]))}</td><td>{r["used_points"]}</td>'
                          f'<td>{r["AUC_PA"]:.4f}</td><td>{r["HR_at_10pct"]:.2%}</td>'
                          f'<td>{r["Lift_at_10pct"]:.3f}</td><td>{r["Area_at_80pct_hit"]:.2%}</td>'
                          f'<td>{r["outside_points"]+r["nodata_points"]+r["transform_failed_points"]}</td></tr>')
    independence = ('用户配置声明点位独立于建模，可以称为 prediction-rate curve。' if config.get('validation_independent')
                    else '尚未确认点位是否独立于建模：当前报告称为 hit-area curve。确认独立后可称 prediction-rate curve；若点位用于建模，应称 success-rate curve。')
    figures = ''.join(f'<h2>{html.escape(name)}</h2><a href="summary_{slug(name)}.pdf">矢量 PDF</a><img src="summary_{slug(name)}.png">' for name in config['point_sets'])
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>重要性场空间命中评估</title>
<style>body{font:15px system-ui,sans-serif;margin:32px;color:#203047;line-height:1.6}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3f9;position:sticky;top:0}input{padding:9px;width:400px;max-width:90%;margin:12px 0}img{max-width:100%}.note{background:#f2f6fa;padding:16px}a{color:#1766a4}</style>
<h1>重要性场空间命中评估</h1>'''
    content += f'<p>{html.escape(independence)}</p><div class="note">'
    content += 'AUC_PA = ∫H(a)da；HR@10% = H(0.1)；Lift@10% = H(0.1)/0.1；Area@80%Hit 为达到 H(a)=0.8 的最小面积比例。随机基线分别为 0.5、10%、1、80%。<br>'
    content += '相同分值整组统计，组内采用线性期望插值；前10%的命中数可能是期望值。每组目录的 area_budgets.csv 同时提供严格大于与大于等于阈值的实际面积/命中率，curve.csv 保存所有整组端点。<br>'
    content += f'面积口径：{html.escape(config.get("area_mode","geodesic"))}；NoData/NaN/Inf 排除，有效零值保留；同一像元中的多个点分别计数。<br>'
    content += '因子场衡量观测重要性与事件位置的空间集中程度，不解释为灾害概率。截图数字化点存在位置不确定性；默认塌陷点含遮挡推测点，不含3个边缘候选点。</div>'
    content += ''.join(f'<p><b>注意：</b>{html.escape(w)}</p>' for w in warnings)
    content += '<p><a href="summary.csv">下载汇总 CSV</a> · <a href="manifest.json">输入与方法记录</a></p><input id="filter" placeholder="筛选实验、因子或点位类型"><table><thead><tr><th>实验 / 场（点击查看曲线）</th><th>点位</th><th>N</th><th>AUC_PA ↑</th><th>HR@10% ↑</th><th>Lift@10% ↑</th><th>Area@80%Hit ↓</th><th>无效点</th></tr></thead><tbody>'
    content += ''.join(table_rows)+'</tbody></table>'+figures
    content += '<script>document.getElementById("filter").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>'
    (out/'report.html').write_text(content,encoding='utf-8')


def run(config_path: Path, overwrite=False, config_overrides=None):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if config_overrides:
        config.update(config_overrides)
    config = normalize_point_set_config(config)
    base = config_path.parent
    resolve = lambda p:(base/Path(p)).resolve()
    fractions = sorted(set([.1]+list(config.get('area_fractions',[.1]))))
    target = config.get('hit_target',.8)
    if not all(0<float(a)<=1 for a in fractions) or not 0<target<=1:
        raise ValueError('Area budgets and hit_target must be in (0,1].')
    if len({r['name'] for r in config['runs']}) != len(config['runs']):
        raise ValueError('Run names must be unique.')
    points = {name:read_points(resolve(v['path']),v.get('exclude_status',[])) for name,v in config['point_sets'].items()}
    jobs = []
    selection = config.get('factor_fields_by_point_set')
    selection_warnings = []
    from standard_observation_factors import factor_key
    if selection is not None and set(selection)!=set(points):
        raise ValueError('factor_fields_by_point_set must specify every configured point set.')
    for experiment in config['runs']:
        for pair in config['event_pairs']:
            if pair['points'] not in points:
                raise ValueError(f'Unknown point set: {pair["points"]}')
            jobs.append((experiment['name'],'event',resolve(experiment['event_dir'])/pair['file'],pair['points']))
        factors = sorted(resolve(experiment['factor_dir']).glob(config.get('factor_glob','*_importance_field.tif')))
        if not factors:
            raise ValueError(f'No factor GeoTIFFs: {experiment["factor_dir"]}')
        if selection is not None:
            available = {factor_key(p.stem.removesuffix('_importance_field')) for p in factors}
            for name,requested in selection.items():
                missing = [field for field in requested if factor_key(field) not in available]
                if missing:
                    selection_warnings.append(f'{experiment["name"]} / {name}: 缺少选定因子 {missing}，不补零。')
        for raster in factors:
            for name in points:
                if selection is not None and factor_key(raster.stem.removesuffix('_importance_field')) not in {
                        factor_key(value) for value in selection[name]}:
                    continue
                jobs.append((experiment['name'],'factor',raster,name))
    for _,_,raster,_ in jobs:
        if not raster.is_file():
            raise FileNotFoundError(raster)
    if not jobs:
        raise ValueError('No evaluation jobs configured.')
    out = resolve(config['output_dir'])
    # Keep previous exports unless explicit overwrite is requested.
    out.mkdir(parents=True,exist_ok=True)
    for artifact in ('report.html','summary.csv','manifest.json'):
        if (out/artifact).exists() and not overwrite:
            raise FileExistsError(f'{out}: result files already exist. Choose a new output_dir, or pass --overwrite to update these generated results.')
    configure_plot_style()
    summaries = []
    details = []
    for i,(name,kind,raster,point_name) in enumerate(jobs,1):
        field = raster.stem.removesuffix('_importance_field').removesuffix('_importance')
        job_key = hashlib.sha256(f'{name}|{kind}|{raster}|{point_name}'.encode()).hexdigest()[:10]
        folder = out/f'{kind}_{slug(field)}_{slug(point_name)}_{job_key}'
        folder.mkdir(exist_ok=True)
        curve,audit,metadata = evaluate_raster(raster,points[point_name],config.get('area_mode','geodesic'),config.get('outside_policy','error'))
        hr10 = budget_metrics(curve,.1)
        area80,area80_group = area_at_hit(curve,.8)
        area_custom,area_custom_group = area_at_hit(curve,target)
        summary = dict(run=name,kind=kind,field=field,point_set=point_name,
                       AUC_PA=curve.auc,HR_at_10pct=hr10['HR'],Lift_at_10pct=hr10['Lift'],
                       Area_at_80pct_hit=area80,Area_at_80pct_hit_full_group=area80_group,
                       hit_target=target,Area_at_target_hit=area_custom,Area_at_target_hit_full_group=area_custom_group,
                       **metadata,result_dir=folder.name)
        summaries.append(summary)
        write_csv(folder/'curve.csv', [dict(score_threshold=None,AF=0.,HR=0.,Lift=None,cumulative_hits=0,group_cells=0,group_hits=0)]+
                  [dict(score_threshold=float(score),AF=float(curve.area[j+1]),HR=float(curve.hit[j+1]),
                        Lift=float(curve.hit[j+1]/curve.area[j+1]),
                        cumulative_hits=int(round(curve.hit[j+1]*curve.n_points)),group_cells=int(curve.cells[j]),group_hits=int(curve.point_count[j]))
                   for j,score in enumerate(curve.scores)])
        write_csv(folder/'area_budgets.csv',[budget_metrics(curve,float(a)) for a in fractions])
        write_csv(folder/'point_audit.csv',audit)
        write_csv(folder/'score_thresholds.csv',[threshold_metrics(curve,float(t)) for t in config.get('score_thresholds',[])])
        plot_curve(curve,summary,field_point_label(field,point_name),folder/'curve')
        details.append(summary)
        print(f'[{i}/{len(jobs)}] {kind} / {field} / {point_name}: N={curve.n_points}, AUC={curve.auc:.4f}, HR10={hr10["HR"]:.3f}',flush=True)
    warnings = list(selection_warnings)
    for name in points:
        group = [r for r in summaries if r['point_set']==name]
        if len({r['domain_hash'] for r in group})>1:
            warnings.append(f'{name}: 栅格有效区域或网格不同，各结果使用各自有效区域，不能直接认定为同一空间域上的公平模型比较。')
        if len({r['used_points'] for r in group})>1 or any(r['outside_points']+r['nodata_points']+r['transform_failed_points'] for r in group):
            warnings.append(f'{name}: 存在被排除点；请查看 point_audit.csv，各场使用的验证样本可能不同。')
        plot_summary(summaries,name,out/f'summary_{slug(name)}')
    write_csv(out/'summary.csv',summaries)
    manifest = dict(config=config,config_path=str(config_path),method='Grouped scores, area-proportional expected interpolation within ties',
                    points={k:dict(path=v['path'],crs=v['crs'].to_string(),n=len(v['points'])) for k,v in points.items()},
                    warnings=warnings,results=details)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    render_report(summaries,config,out,warnings)
    print(f'Report: {out/"report.html"}')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('importance_evaluation_config_alter.json'))
    parser.add_argument('--overwrite',action='store_true',help='Update generated outputs in the configured output_dir; input files are never overwritten.')
    args = parser.parse_args()
    run(args.config,overwrite=args.overwrite)


if __name__=='__main__':
    main()
