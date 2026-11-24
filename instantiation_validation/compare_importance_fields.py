"""Evaluate OELG and instantiation_en exports, then create paired comparisons.

python compare_importance_fields.py [--config importance_comparison_config.json]
Input GeoTIFF directories are explicit in importance_comparison_config.json.
Only the configured hazard-specific observation factors are evaluated.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from evaluate_importance_fields import (canonical_point_set, field_point_label,
                                        normalize_point_set_config, run, slug, write_csv)

METRICS = ('AUC_PA','HR_at_10pct','Lift_at_10pct','Area_at_80pct_hit')
DEFAULT_FACTOR_SELECTION = {
    'Subsidence':['Subsidence Depth','Crack Width','Surface Deformation','Tilt Angle','Structural Characteristics'],
    'waterlogging':['Water Body Area','Water Level','Precipitation','Soil Moisture'],
}


def match_key(value):
    """Ignore filename spacing/punctuation, never guess translated aliases."""
    return ''.join(c for c in value if c.isalnum()).casefold()


def selected_factor_fields(config):
    """Return hazard-specific selections using canonical validation point-set names."""
    config = normalize_point_set_config(config)
    selection = config.get('factor_fields_by_point_set',DEFAULT_FACTOR_SELECTION)
    if set(selection)!=set(config['point_sets']):
        raise ValueError('Specify factor_fields_by_point_set for every point set.')
    return selection


def filter_results(results,config):
    """Normalize legacy point-set identifiers and retain the selected factor fields."""
    selection = {name:{match_key(v) for v in values} for name,values in selected_factor_fields(config).items()}
    results = [dict(row, point_set=canonical_point_set(row['point_set'])) for row in results]
    return [r for r in results if r['kind']!='factor' or match_key(r['field']) in selection[r['point_set']]]


def pair_results(results, reference, candidate):
    indices = {reference:{}, candidate:{}}
    if reference == candidate:
        raise ValueError('Reference and candidate runs must be different.')
    for row in results:
        if row['run'] not in indices:
            continue
        key = (row['kind'],match_key(row['field']),canonical_point_set(row['point_set']))
        if key in indices[row['run']]:
            raise ValueError(f'Ambiguous normalized field name: {row["field"]}')
        indices[row['run']][key] = row
    if not all(indices.values()):
        raise ValueError('Both configured comparison runs must have evaluation results.')
    paired = []
    for key in sorted(set(indices[reference]) | set(indices[candidate])):
        left,right = indices[reference].get(key),indices[candidate].get(key)
        label = (left or right)['field']
        if left is None:
            status = 'candidate_only'
        elif right is None:
            status = 'reference_only'
        elif (not left.get('sample_hash') or not right.get('sample_hash') or
              left['sample_hash'] != right['sample_hash'] or
              left['domain_hash'] != right['domain_hash'] or left['area_mode'] != right['area_mode']):
            status = 'incompatible_domain_or_samples'
        else:
            status = 'matched'
        row = dict(kind=key[0],field=label,point_set=key[2],status=status,
                   reference=reference,candidate=candidate,
                   reference_N=left['used_points'] if left else None,
                   candidate_N=right['used_points'] if right else None,
                   reference_result_dir=left['result_dir'] if left else '',
                   candidate_result_dir=right['result_dir'] if right else '')
        for metric in METRICS:
            row[f'reference_{metric}'] = float(left[metric]) if left else None
            row[f'candidate_{metric}'] = float(right[metric]) if right else None
            row[f'delta_{metric}'] = float(right[metric])-float(left[metric]) if status=='matched' else None
        paired.append(row)
    return paired


def read_curve(path):
    with path.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r['AF']) for r in rows]),np.array([float(r['HR']) for r in rows])


def plot_pair(row,out):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(6.8,5.3),layout='constrained')
    for side,color in [('reference','#246ca6'),('candidate','#c77713')]:
        area,hit = read_curve(out/row[f'{side}_result_dir']/'curve.csv')
        ax.plot(area*100,hit*100,color=color,lw=2,
                label=f'{row[side]} (AUC={row[f"{side}_AUC_PA"]:.3f})')
    ax.plot([0,100],[0,100],'--',color='#888888',lw=1,label='Random selection')
    ax.set(xlim=(0,100),ylim=(0,100),xlabel='Study area selected (%)',
           ylabel='Observed points captured (%)',title=field_point_label(row['field'],row['point_set']))
    ax.grid(alpha=.2)
    ax.legend(loc='lower right')
    target = out/'paired_curves'/f'{row["kind"]}_{slug(field_point_label(row["field"],row["point_set"],"_"))}'
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target.with_suffix('.png'),dpi=180)
    fig.savefig(target.with_suffix('.pdf'))
    plt.close(fig)
    return target.with_suffix('.png').relative_to(out).as_posix()


def plot_paired_metrics(paired,point_set,out):
    import matplotlib.pyplot as plt
    rows = [r for r in paired if r['status']=='matched' and r['point_set']==point_set]
    if not rows:
        return
    rows.sort(key=lambda r:(r['kind'],r['field']),reverse=True)
    fig,axes = plt.subplots(1,4,figsize=(16,max(4.5,len(rows)*.55)),sharey=True,layout='constrained')
    positions = np.arange(len(rows))
    for ax,metric,title,baseline in zip(axes,METRICS,
            ['AUC_PA (higher better)','HR@10% (higher better)','Lift@10% (higher better)','Area@80%Hit (lower better)'],[.5,.1,1,.8]):
        ax.barh(positions+.18,[r[f'reference_{metric}'] for r in rows],height=.34,color='#246ca6',label=rows[0]['reference'])
        ax.barh(positions-.18,[r[f'candidate_{metric}'] for r in rows],height=.34,color='#c77713',label=rows[0]['candidate'])
        ax.axvline(baseline,color='#888888',ls='--',lw=1)
        ax.set_title(title,fontsize=10)
        ax.grid(axis='x',alpha=.2)
        if metric!='Lift_at_10pct':
            ax.set_xlim(0,1)
    axes[0].set_yticks(positions,[f'{r["kind"]}: {r["field"]}' for r in rows],fontsize=9)
    axes[-1].legend(loc='best',fontsize=9)
    fig.suptitle(f'{point_set}: paired comparison on the same spatial domain and validation points')
    target = out/f'paired_metrics_{slug(point_set)}'
    fig.savefig(target.with_suffix('.png'),dpi=180)
    fig.savefig(target.with_suffix('.pdf'))
    plt.close(fig)


def make_comparison(results,config,out):
    """Write paired reports and figures using canonical hazard names."""
    config = normalize_point_set_config(config)
    results = filter_results(results,config)
    reference = config['comparison']['reference']
    candidate = config['comparison']['candidate']
    paired = pair_results(results,reference,candidate)
    for row in paired:
        row['curve_image'] = plot_pair(row,out) if row['status']=='matched' else ''
    write_csv(out/'paired_comparison.csv',paired)
    for name in config['point_sets']:
        plot_paired_metrics(paired,name,out)
    status_names = {'matched':'可配对','candidate_only':f'仅 {candidate}',
                    'reference_only':f'仅 {reference}','incompatible_domain_or_samples':'空间域或样本不一致，不计算差值'}
    matched = sum(r['status']=='matched' for r in paired)
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>OELG 与 DOELG 指标对比</title>
<style>body{font:15px system-ui,sans-serif;margin:32px;color:#203047;line-height:1.6}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3f9;position:sticky;top:0}img{max-width:100%}.note{padding:16px;background:#f2f6fa}a{color:#1766a4}input{padding:9px;width:420px;max-width:90%;margin:12px 0}</style>
<h1>OELG 与 instantiation_en 指标对比</h1>'''
    content += f'<p>共 {len(results)} 组独立评价；{matched} 组具有同名场、相同空间域及验证样本的配对结果。</p>'
    content += f'<div class="note">差值定义为 {html.escape(candidate)} − {html.escape(reference)}。AUC、HR 和 Lift 越高越好；Area@80%Hit 越低越好。缺失场不补零。同分值采用组内期望插值，实际阈值统计见各场 CSV。'
    content += ''.join(f'<p>{html.escape(note)}</p>' for note in config.get('method_notes',[]))+'</div>'
    content += '<p><a href="paired_comparison.csv">配对指标与差值 CSV</a> · <a href="report.html">全部场的详细评价报告</a> · <a href="summary.csv">全部指标 CSV</a></p>'
    content += '<input id="filter" placeholder="筛选事件、因子、点位或配对状态"><table><thead><tr><th>场 / 点位</th><th>状态</th><th>方法</th><th>AUC_PA</th><th>HR@10%</th><th>Lift@10%</th><th>Area@80%Hit</th></tr></thead><tbody>'
    for r in paired:
        label = html.escape(f'{r["kind"]}: {field_point_label(r["field"],r["point_set"]," / ")}')
        if r['curve_image']:
            label = f'<a href="{r["curve_image"]}">{label}</a>'
        for side in ('reference','candidate','delta'):
            method = r[side] if side!='delta' else '差值'
            cells = []
            for metric in METRICS:
                value = r[f'{side}_{metric}']
                if value is None:
                    cells.append('—')
                elif metric in ('HR_at_10pct','Area_at_80pct_hit'):
                    cells.append(f'{value*100:+.2f} pp' if side=='delta' else f'{value:.2%}')
                else:
                    cells.append(f'{value:+.4f}' if side=='delta' else f'{value:.4f}')
            content += f'<tr><td>{label}</td><td>{html.escape(status_names[r["status"]])}</td><td>{html.escape(method)}</td>'+''.join(f'<td>{v}</td>' for v in cells)+'</tr>'
    content += '</tbody></table>'
    for name in config['point_sets']:
        if (out/f'paired_metrics_{slug(name)}.png').exists():
            content += f'<h2>{html.escape(name)}</h2><a href="paired_metrics_{slug(name)}.pdf">矢量 PDF</a><img src="paired_metrics_{slug(name)}.png">'
    content += '<script>document.getElementById("filter").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>'
    (out/'comparison_report.html').write_text(content,encoding='utf-8')
    print(f'Paired report: {out/"comparison_report.html"}',flush=True)
    return paired


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('importance_comparison_config.json'))
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = normalize_point_set_config(json.loads(config_path.read_text(encoding='utf-8-sig')))
    names = {r['name'] for r in config['runs']}
    if not {config['comparison']['reference'],config['comparison']['candidate']} <= names:
        raise ValueError('Comparison names must be present in runs.')
    results = run(config_path,overwrite=args.overwrite,
                  config_overrides={'factor_fields_by_point_set':selected_factor_fields(config)})
    out = (config_path.parent/config['output_dir']).resolve()
    make_comparison(results,config,out)


if __name__=='__main__':
    main()
