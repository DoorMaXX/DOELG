import pandas as pd
import ast
import matplotlib.pyplot as plt
from pathlib import Path
from validation import is_fuzzy_match
import json
import matplotlib

matplotlib.use('TkAgg')
plt.style.use('ggplot')


def extract_geo_info(json_str):
    """
    Parses a geographic spatial data JSON or dictionary string into structured
    lists of features (with names and types) and spatial relations.
    """
    if not isinstance(json_str, str) or not json_str.strip():
        return [], []
    try:
        data = json.loads(json_str)
    except:
        try:
            data = ast.literal_eval(json_str)
        except:
            return [], []

    features = [{"name": f.get('name'), "type": f.get('type')} for f in data.get('features', [])]
    relations = [{
        "sub": r.get('subject'),
        "obj": r.get('object'),
        "rel": r.get('relation_type') or r.get('relation')
    } for r in data.get('spatial_relations', [])]

    return features, relations


def calculate_geo_metrics(file_path):
    """
    Calculates and returns the F1 score for geographic features and spatial relations
    by evaluating model predictions against the ground truth in the given file.
    """
    df = pd.read_csv(file_path) if str(file_path).endswith('.csv') else pd.read_excel(file_path)

    tp_f, fp_f, fn_f = 0, 0, 0
    tp_r, fp_r, fn_r = 0, 0, 0

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
                forward_match = is_fuzzy_match(pr['sub'], tr['sub']) and is_fuzzy_match(pr['obj'], tr['obj'])
                backward_match = is_fuzzy_match(pr['sub'], tr['obj']) and is_fuzzy_match(pr['obj'], tr['sub'])

                if forward_match or backward_match:
                    match_idx = i
                    break

            if match_idx != -1:
                tp_r += 1
                del temp_true_r[match_idx]
            else:
                fp_r += 1
        fn_r += len(temp_true_r)

    def get_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    return get_f1(tp_f, fp_f, fn_f), get_f1(tp_r, fp_r, fn_r)


def get_method_scores(shot_list, base_dir, method_prefix):
    """
    Iterates through available few-shot files for a specific method prefix to collect
    and return lists of valid shots, feature F1 scores, and relation F1 scores.
    """
    f_scores = []
    r_scores = []
    valid_shots = []

    for shot in shot_list:
        file_path = Path(base_dir) / f"{method_prefix}-{shot}shot.xlsx"
        if not file_path.exists():
            file_path = Path(base_dir) / f"{method_prefix}-{shot}shot.csv"

        if file_path.exists():
            print(f"Reading {shot}-shot data for method {method_prefix}...")
            f_f1, r_f1 = calculate_geo_metrics(file_path)
            f_scores.append(f_f1)
            r_scores.append(r_f1)
            valid_shots.append(shot)
        else:
            print(f"Warning: File not found {file_path}")

    return valid_shots, f_scores, r_scores


def plot_geo_comparison_results(shot_list, base_dir, method1='default', method2='code'):
    """
    Generates comparative line chart visualizations for both geographic features and
    spatial relations extraction F1 scores between two given methods across various few-shot settings.
    """
    s_def, f_def, r_def = get_method_scores(shot_list, base_dir, method1)
    s_code, f_code, r_code = get_method_scores(shot_list, base_dir, method2)

    plt.figure(figsize=(9, 6))
    plt.plot(s_def, f_def, marker='o', label='Default (Features)', linewidth=2, color='#3498db')
    plt.plot(s_code, f_code, marker='D', label='Code (Features)', linewidth=2, color='#2ecc71')
    plt.title('Geographic Features Extraction Comparison', fontsize=14, pad=15)
    plt.xlabel('Few-shot Samples', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(shot_list)
    plt.grid(True, linestyle='--', alpha=0.6)

    all_f = f_def + f_code
    if all_f:
        plt.ylim(min(all_f) - 0.05, max(all_f) + 0.05)

    for i in range(len(f_def)):
        plt.annotate(
            f"{f_def[i]:.3f}",
            (s_def[i], f_def[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10
        )
    for i in range(len(f_code)):
        plt.annotate(
            f"{f_code[i]:.3f}",
            (s_code[i], f_code[i]),
            textcoords="offset points",
            xytext=(0, -18),
            ha='center',
            fontsize=10
        )

    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig("features_comparison.png", dpi=300, facecolor='white')
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(s_def, r_def, marker='s', label='Default (Relations)', linewidth=2, color='#e67e22', linestyle='--')
    plt.plot(s_code, r_code, marker='^', label='Code (Relations)', linewidth=2, color='#e74c3c', linestyle='--')
    plt.title('Spatial Relations Extraction Comparison', fontsize=14, pad=15)
    plt.xlabel('Few-shot Samples', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(shot_list)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    for i in range(len(r_code)):
        plt.annotate(f"{r_code[i]:.3f}", (s_code[i], r_code[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')
    for i in range(len(r_def)):
        plt.annotate(f"{r_def[i]:.3f}", (s_def[i], r_def[i]), textcoords="offset points", xytext=(0, -15),
                     ha='center')
    plt.tight_layout()
    plt.savefig("relations_comparison.png", dpi=300, facecolor='white')
    plt.show()


if __name__ == "__main__":
    shots = [1, 5, 10]
    data_dir = "./data/extraction_dataset/geo/"
    plot_geo_comparison_results(shots, data_dir, method1='default', method2='code')