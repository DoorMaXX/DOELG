import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from extract_validation.validation import is_fuzzy_match


def extract_geo_info(json_str):
    """Parse geographic entities and spatial relations from a prepared prediction."""
    if not isinstance(json_str, str) or not json_str.strip():
        return ([], [])
    try:
        data = json.loads(json_str.replace("'", '"'))
    except:
        try:
            data = ast.literal_eval(json_str)
        except:
            return ([], [])
    features = [{'name': f.get('name'), 'type': f.get('type')} for f in data.get('features', [])]
    relations = [{'sub': r.get('subject'), 'obj': r.get('object'), 'rel': r.get('relation_type') or r.get('relation')} for r in data.get('spatial_relations', [])]
    return (features, relations)


def calculate_geo_separated_metrics(file_path):
    """Return entity, relation, and combined micro precision, recall, and F1."""
    df = pd.read_csv(file_path) if str(file_path).endswith('.csv') else pd.read_excel(file_path)
    tp_f, fp_f, fn_f = (0, 0, 0)
    tp_r, fp_r, fn_r = (0, 0, 0)
    for _, row in df.iterrows():
        pred_f, pred_r = extract_geo_info(row['raw_json'])
        true_f, true_r = extract_geo_info(row['true_json'])
        temp_true_f = true_f[:]
        for pf in pred_f:
            match_idx = -1
            for i, tf in enumerate(temp_true_f):
                if is_fuzzy_match(pf['name'], tf['name']):
                    match_idx = i
                    break
            if match_idx != -1:
                tp_f += 1
                del temp_true_f[match_idx]
            else:
                fp_f += 1
        fn_f += len(temp_true_f)
        temp_true_r = true_r[:]
        for pr in pred_r:
            match_idx = -1
            for i, tr in enumerate(temp_true_r):
                forward = is_fuzzy_match(pr['sub'], tr['sub']) and is_fuzzy_match(pr['obj'], tr['obj'])
                backward = is_fuzzy_match(pr['sub'], tr['obj']) and is_fuzzy_match(pr['obj'], tr['sub'])
                rel_match = is_fuzzy_match(pr['rel'], tr['rel'], threshold=0.6)
                if (forward or backward) and rel_match:
                    match_idx = i
                    break
            if match_idx != -1:
                tp_r += 1
                del temp_true_r[match_idx]
            else:
                fp_r += 1
        fn_r += len(temp_true_r)

    def get_metrics(tp, fp, fn):
        """Calculate precision, recall, and F1 from accumulated match counts."""
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (prec * recall) / (prec + recall) if prec + recall > 0 else 0
        return (prec, recall, f1)
    tp_total = tp_f + tp_r
    fp_total = fp_f + fp_r
    fn_total = fn_f + fn_r
    return (get_metrics(tp_f, fp_f, fn_f), get_metrics(tp_r, fp_r, fn_r), get_metrics(tp_total, fp_total, fn_total))


def calculate_geo_metrics(file_path):
    """Return entity, relation, and combined geographic micro-F1 scores."""
    return tuple(metrics[2] for metrics in calculate_geo_separated_metrics(file_path))


def get_method_scores(shot_list, base_dir, method_prefix):
    """Collect the original geographic metrics for available few-shot workbooks."""
    shots, entities, relations, combined = [], [], [], []
    for shot in shot_list:
        path = Path(base_dir) / f'{method_prefix}-{shot}shot.xlsx'
        if not path.exists():
            path = path.with_suffix('.csv')
        if not path.exists():
            raise FileNotFoundError(path)
        entity, relation, micro = calculate_geo_metrics(path)
        shots.append(shot)
        entities.append(entity)
        relations.append(relation)
        combined.append(micro)
    return shots, entities, relations, combined


def plot_geo_comparison_results(shot_list, base_dir, method1='default', method2='code', output_dir=None):
    """Plot entity, relation, and combined micro-F1 with relation-word matching enabled."""
    output = Path(output_dir or base_dir)
    output.mkdir(parents=True, exist_ok=True)
    left = get_method_scores(shot_list, base_dir, method1)
    right = get_method_scores(shot_list, base_dir, method2)
    for index, name, title in [(1, 'features', 'Geographic entity F1'),
                                (2, 'relations', 'Spatial relation F1'),
                                (3, 'combined', 'Geo/Spatial micro-F1')]:
        fig, ax = plt.subplots(figsize=(9, 6))
        try:
            for values, method, marker in [(left, method1, 'o'), (right, method2, 'D')]:
                ax.plot(values[0], values[index], marker=marker, label=method, linewidth=2)
                for shot, score in zip(values[0], values[index]):
                    ax.annotate(f'{score:.3f}', (shot, score), xytext=(0, 8), textcoords='offset points', ha='center')
            ax.set(xlabel='Few-shot samples', ylabel='F1', title=title, xticks=shot_list)
            ax.grid(linestyle='--', alpha=.5)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f'{name}_comparison.png', dpi=300)
        finally:
            plt.close(fig)
    return output
