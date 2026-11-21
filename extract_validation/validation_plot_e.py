import ast
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def normalize_text(text):
    """Normalize names using the original extraction evaluation rules."""
    if not isinstance(text, str):
        return str(text) if text is not None else ''
    return re.sub('[^\\u4e00-\\u9fa5a-zA-Z0-9]', '', text).lower()


def is_fuzzy_match(pred, true, threshold=0.5):
    """Match names by containment or the original character Jaccard threshold."""
    p_name = pred.get('name', '') if isinstance(pred, dict) else str(pred)
    t_name = true.get('name', '') if isinstance(true, dict) else str(true)
    s1, s2 = (normalize_text(p_name), normalize_text(t_name))
    if not s1 or not s2:
        return False
    if s1 in s2 or s2 in s1:
        return True
    set1, set2 = (set(s1), set(s2))
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union >= threshold if union > 0 else False


def extract_info_from_dict(data_str):
    """Extract and filter the event entities and arguments used in the paper."""
    if pd.isna(data_str) or str(data_str).strip() in ['', 'nan', 'None']:
        return ([], [])
    try:
        data = ast.literal_eval(str(data_str))
    except:
        return ([], [])
    sub_obj_list = []
    arg_list = []
    for item in data.get('disaster', []):
        if isinstance(item.get('feature'), dict):
            sub_obj_list.append({'name': item['feature'].get('name'), 'type': item['feature'].get('geo_type')})
        if isinstance(item.get('impact'), dict):
            sub_obj_list.append({'name': item['impact'].get('name'), 'type': item['impact'].get('impact_type')})
        if item.get('time'):
            arg_list.append({'name': str(item.get('time')), 'type': 'occurrence_time'})
        factors = item.get('factors', {})
        if isinstance(factors, dict):
            for f_type in ['feature_factors', 'environment_factors']:
                val = factors.get(f_type)
                if isinstance(val, list):
                    for v in val:
                        arg_list.append({'name': str(v), 'type': f_type})
    for item in data.get('observation', []):
        if isinstance(item.get('sensor'), dict):
            sensor_info = item['sensor']
            sub_obj_list.append({'name': sensor_info.get('name'), 'type': f"{sensor_info.get('type')}_{sensor_info.get('platform')}"})
        if isinstance(item.get('observed_object'), dict):
            sub_obj_list.append({'name': item['observed_object'].get('name'), 'type': item['observed_object'].get('geo_type')})
        factors = item.get('factors', {})
        if isinstance(factors, dict):
            for f_type in ['feature_factors', 'environment_factors']:
                val = factors.get(f_type)
                if isinstance(val, list):
                    for v in val:
                        sub_obj_list.append({'name': str(v), 'type': f_type})
                        arg_list.append({'name': str(v), 'type': f_type})
    sub_obj_list = [x for x in sub_obj_list if x.get('name') and str(x.get('name')).strip() != 'None']
    arg_list = [x for x in arg_list if x.get('name') and str(x.get('name')).strip() != 'None']
    return (sub_obj_list, arg_list)


def calculate_event_metrics(file_path):
    """Compute micro metrics over the combined event entities and arguments."""
    df = pd.read_excel(file_path)
    total_tp, total_fp, total_fn = (0, 0, 0)
    tp_type, total_matched_names = (0, 0)
    for _, row in df.iterrows():
        p_so, p_arg = extract_info_from_dict(row['raw_json'])
        t_so, t_arg = extract_info_from_dict(row['true_json'])
        preds = p_so + p_arg
        trues = t_so + t_arg
        temp_trues = trues[:]
        for p in preds:
            match_idx = -1
            for i, t in enumerate(temp_trues):
                if is_fuzzy_match(p, t):
                    match_idx = i
                    break
            if match_idx != -1:
                total_tp += 1
                total_matched_names += 1
                if str(p.get('type')) == str(temp_trues[match_idx].get('type')):
                    tp_type += 1
                del temp_trues[match_idx]
            else:
                total_fp += 1
        total_fn += len(temp_trues)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    type_accuracy = tp_type / total_matched_names if total_matched_names > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'type_accuracy': type_accuracy, 'true_positive': total_tp, 'false_positive': total_fp, 'false_negative': total_fn}


def calculate_file_metrics(file_path):
    """Return precision, combined Event-F1, and type accuracy as in KGC_2."""
    result = calculate_event_metrics(file_path)
    return result['precision'], result['f1'], result['type_accuracy']


def get_method_scores(shot_list, base_dir, method_prefix):
    """Collect the original combined event metrics for available few-shot workbooks."""
    shots, precisions, f1_scores, accuracies = [], [], [], []
    for shot in shot_list:
        path = Path(base_dir) / f'{method_prefix}-{shot}shot.xlsx'
        if not path.exists():
            raise FileNotFoundError(path)
        precision, f1, accuracy = calculate_file_metrics(path)
        shots.append(shot)
        precisions.append(precision)
        f1_scores.append(f1)
        accuracies.append(accuracy)
    return shots, precisions, f1_scores, accuracies


def plot_comparison_results(shot_list, base_dir, method1='default', method2='code', output_dir=None):
    """Plot combined Event-F1, precision, and type accuracy using the evaluation metric."""
    output = Path(output_dir or base_dir)
    output.mkdir(parents=True, exist_ok=True)
    left = get_method_scores(shot_list, base_dir, method1)
    methods = [(left, method1, 'o')]
    if method2 is not None:
        methods.append((get_method_scores(shot_list, base_dir, method2), method2, 'D'))
    for index, name, label in [(1, 'precision', 'Precision'), (2, 'event_f1', 'Event micro-F1'),
                               (3, 'type_accuracy', 'Type accuracy')]:
        fig, ax = plt.subplots(figsize=(9, 6))
        try:
            for values, method, marker in methods:
                ax.plot(values[0], values[index], marker=marker, label=method, linewidth=2)
                for shot, score in zip(values[0], values[index]):
                    ax.annotate(f'{score:.3f}', (shot, score), xytext=(0, 8), textcoords='offset points', ha='center')
            ax.set(xlabel='Few-shot samples', ylabel=label, title=f'Event extraction: {label}', xticks=shot_list)
            ax.grid(linestyle='--', alpha=.5)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f'comparison_{name}.png', dpi=300)
        finally:
            plt.close(fig)
    return output


def plot_fewshot_results(shot_list, base_dir, method, output_dir=None):
    """Plot one method with the same combined metrics used by the comparison workflow."""
    return plot_comparison_results(shot_list, base_dir, method, None, output_dir)
