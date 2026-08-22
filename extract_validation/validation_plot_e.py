import pandas as pd
import re
import ast
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use('TkAgg')

def normalize_text(text):
    """
    Normalizes the input text by removing all non-alphanumeric or non-Chinese
    characters and converting the text to lowercase.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text).lower()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def is_fuzzy_match(pred, true, threshold=0.5):
    """
    Determines whether two strings match based on substring inclusion or a Jaccard
    similarity score that meets or exceeds the specified threshold.
    """
    s1, s2 = normalize_text(pred), normalize_text(true)
    if not s1 or not s2: return False
    if s1 in s2 or s2 in s1: return True
    set1, set2 = set(s1), set(s2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return (intersection / union) >= threshold if union > 0 else False


def extract_info_from_dict(data_str):
    """
    Parses a JSON-like string into a dictionary and extracts subject/object entities
    along with their arguments for both 'disaster' and 'observation' events.
    Returns a tuple containing a list of subject/object dictionaries and a list of argument dictionaries.
    """
    if pd.isna(data_str) or str(data_str).strip() in ['', 'nan', 'None']:
        return [], []
    try:
        data = ast.literal_eval(str(data_str))
    except:
        return [], []

    sub_obj_list = []
    arg_list = []

    for item in data.get('disaster', []):
        if isinstance(item.get('feature'), dict):
            sub_obj_list.append({
                "name": item['feature'].get('name'),
                "type": item['feature'].get('geo_type')
            })
        if isinstance(item.get('impact'), dict):
            sub_obj_list.append({
                "name": item['impact'].get('name'),
                "type": item['impact'].get('impact_type')
            })

        if item.get('time'):
            arg_list.append({"name": str(item.get('time')), "type": "occurrence_time"})

        factors = item.get('factors', {})
        if isinstance(factors, dict):
            for f_type in ['feature_factors', 'environment_factors']:
                val = factors.get(f_type)
                if isinstance(val, list):
                    for v in val:
                        arg_list.append({"name": str(v), "type": f_type})

    for item in data.get('observation', []):
        if isinstance(item.get('sensor'), dict):
            sensor_info = item['sensor']
            sub_obj_list.append({
                "name": sensor_info.get('name'),
                "type": f"{sensor_info.get('type')}_{sensor_info.get('platform')}"
            })

        if isinstance(item.get('observed_object'), dict):
            sub_obj_list.append({
                "name": item['observed_object'].get('name'),
                "type": item['observed_object'].get('geo_type')
            })

        factors = item.get('factors', {})
        if isinstance(factors, dict):
            for f_type in ['feature_factors', 'environment_factors']:
                val = factors.get(f_type)
                if isinstance(val, list):
                    for v in val:
                        sub_obj_list.append({"name": str(v), "type": f_type})
                        arg_list.append({"name": str(v), "type": f_type})

    sub_obj_list = [x for x in sub_obj_list if x.get('name') and str(x.get('name')).strip() != 'None']
    arg_list = [x for x in arg_list if x.get('name') and str(x.get('name')).strip() != 'None']

    return sub_obj_list, arg_list


def calculate_file_metrics(file_path):
    """
    Reads extraction results from an Excel file and calculates evaluation metrics.
    Returns the F1 scores for Subjects/Objects, Arguments, and the Type Accuracy score.
    """
    df = pd.read_excel(file_path)

    tp_so, fp_so, fn_so = 0, 0, 0
    tp_type, total_matched_names = 0, 0
    tp_arg, fp_arg, fn_arg = 0, 0, 0

    for _, row in df.iterrows():
        pred_so, pred_arg = extract_info_from_dict(row['raw_json'])
        true_so, true_arg = extract_info_from_dict(row['true_json'])

        temp_true_so = true_so[:]
        for p in pred_so:
            match_idx = -1
            for i, t in enumerate(temp_true_so):
                if is_fuzzy_match(p['name'], t['name']):
                    match_idx = i
                    break

            if match_idx != -1:
                tp_so += 1
                total_matched_names += 1
                if str(p.get('type')) == str(temp_true_so[match_idx].get('type')):
                    tp_type += 1
                del temp_true_so[match_idx]
            else:
                fp_so += 1
        fn_so += len(temp_true_so)

        temp_true_arg = true_arg[:]
        for p in pred_arg:
            found = False
            for i, t in enumerate(temp_true_arg):
                if is_fuzzy_match(p, t):
                    found = True
                    del temp_true_arg[i]
                    break
            if found:
                tp_arg += 1
            else:
                fp_arg += 1
        fn_arg += len(temp_true_arg)

    def get_metrics(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f1

    type_accuracy = tp_type / total_matched_names if total_matched_names > 0 else 0

    so_p, so_r, so_f1 = get_metrics(tp_so, fp_so, fn_so)
    arg_p, arg_r, arg_f1 = get_metrics(tp_arg, fp_arg, fn_arg)

    return so_f1, arg_f1, type_accuracy


def plot_fewshot_results(shot_list, base_dir, method):
    """
    Evaluates extraction performance for a specific method across various few-shot settings
    and plots the resulting Subject/Object and Argument F1 score curves.
    """
    so_f1_scores = []
    arg_f1_scores = []

    valid_shots = []

    for shot in shot_list:
        file_path = Path(base_dir) / f"{method}-{shot}shot.xlsx"
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        print(f"Evaluating {shot}-shot results...")
        so_f1, arg_f1, type_accuracy = calculate_file_metrics(file_path)
        so_f1_scores.append(so_f1)
        arg_f1_scores.append(arg_f1)
        valid_shots.append(shot)

    plt.figure(figsize=(10, 6))
    plt.plot(valid_shots, so_f1_scores, marker='o', linestyle='-', linewidth=2, label='Subject/Object F1')
    plt.plot(valid_shots, arg_f1_scores, marker='s', linestyle='--', linewidth=2, label='Arguments F1')

    plt.title('Event Extraction Performance vs. Few-shot Samples', fontsize=14)
    plt.xlabel('Number of Few-shot Samples', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(valid_shots)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    for i, txt in enumerate(so_f1_scores):
        plt.annotate(f"{txt:.3f}", (valid_shots[i], so_f1_scores[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    plt.tight_layout()
    plt.savefig("./data/extraction_dataset/event/default_performance_curve.png")
    plt.show()


def get_method_scores(shot_list, base_dir, method_prefix):
    """
    Iterates through available few-shot files for a specific method to collect and return
    the lists of valid shots, Subject/Object F1, Argument F1, and Type Accuracy scores.
    """
    so_scores = []
    arg_scores = []
    valid_shots = []
    type_acc_scores = []

    for shot in shot_list:
        file_path = Path(base_dir) / f"{method_prefix}-{shot}shot.xlsx"
        if file_path.exists():
            print(f"Reading {shot}-shot data for method {method_prefix}...")
            so_f1, arg_f1, type_acc = calculate_file_metrics(file_path)
            so_scores.append(so_f1)
            arg_scores.append(arg_f1)
            type_acc_scores.append(type_acc)
            valid_shots.append(shot)
        else:
            print(f"Warning: File not found {file_path}")

    return valid_shots, so_scores, arg_scores, type_acc_scores


def plot_comparison_results(shot_list, base_dir, method1='default', method2='code'):
    """
    Generates comparative visualizations plotting Subject/Object extraction F1, Arguments F1,
    and Type Accuracy between two different extraction methods across varying few-shot settings.
    """
    shots1, so1, arg1, type_acc1 = get_method_scores(shot_list, base_dir, method1)
    shots2, so2, arg2, type_acc2 = get_method_scores(shot_list, base_dir, method2)

    plt.style.use('seaborn-v0_8-muted')

    plt.figure(figsize=(10, 6))
    plt.plot(shots1, so1, marker='o', label=f'{method1} - Sub/Obj', linewidth=2)
    plt.plot(shots2, so2, marker='D', label=f'{method2} - Sub/Obj', linewidth=2)
    plt.title('Comparison of Subject/Object Extraction (F1 Score)', fontsize=14)
    plt.xlabel('Number of Few-shot Samples', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(shot_list)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    for i in range(len(so1)):
        plt.annotate(f"{so1[i]:.3f}", (shots1[i], so1[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    for i in range(len(so2)):
        plt.annotate(f"{so2[i]:.3f}", (shots2[i], so2[i]), textcoords="offset points", xytext=(0, -15), ha='center')
    plt.tight_layout()
    plt.savefig(Path(base_dir) / "comparison_subject_object.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(shots1, arg1, marker='s', label=f'{method1} - Arguments', linewidth=2, color='orange')
    plt.plot(shots2, arg2, marker='^', label=f'{method2} - Arguments', linewidth=2, color='green')
    plt.title('Comparison of Arguments Extraction (F1 Score)', fontsize=14)
    plt.xlabel('Number of Few-shot Samples', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(shot_list)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    for i in range(len(arg1)):
        plt.annotate(f"{arg1[i]:.3f}", (shots1[i], arg1[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    for i in range(len(arg2)):
        plt.annotate(f"{arg2[i]:.3f}", (shots2[i], arg2[i]), textcoords="offset points", xytext=(0, -15), ha='center')
    plt.tight_layout()
    plt.savefig(Path(base_dir) / "comparison_arguments.png")
    plt.show()

    plt.figure(figsize=(10, 6))

    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plt.plot(
        shots1, type_acc1,
        marker='o',
        color='#4878D0',
        label=f'{method1}',
        linewidth=2
    )

    plt.plot(
        shots2, type_acc2,
        marker='D',
        color='#6ACC64',
        label=f'{method2}',
        linewidth=2
    )

    plt.title('灾害观测事件论元类型识别对比', fontsize=14, pad=10)
    plt.xlabel('参考案例数量（Number of Few-shot Samples）', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.xticks(shot_list)

    all_data = type_acc1 + type_acc2
    if all_data:
        d_min, d_max = min(all_data), max(all_data)
        margin = (d_max - d_min) * 0.2
        plt.ylim(max(0, d_min - margin), min(1.0, d_max + margin))

    plt.grid(True, linestyle='--', color='gray', alpha=0.5)

    for i in range(len(type_acc1)):
        plt.annotate(
            f"{type_acc1[i]:.3f}",
            (shots1[i], type_acc1[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=10
        )

    for i in range(len(type_acc2)):
        plt.annotate(
            f"{type_acc2[i]:.3f}",
            (shots2[i], type_acc2[i]),
            textcoords="offset points",
            xytext=(0, -18),
            ha='center',
            fontsize=10
        )

    plt.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    plt.savefig(
        Path(base_dir) / "comparison_type_accuracy.png",
        dpi=300,
        facecolor='white'
    )
    plt.show()


if __name__ == "__main__":
    shots = [1, 5, 10, 30]
    data_dir = "./data/extraction_dataset/event/"
    plot_comparison_results(shots, data_dir, method1='default', method2='code')