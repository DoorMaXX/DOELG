import pandas as pd
import ast
import json
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib

matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def normalize_str(s):
    """
    Normalizes the input string or list by converting it to lowercase and removing
    specific punctuation marks, separators, and whitespaces.
    """
    if isinstance(s, list):
        s = "".join([str(i) for i in s])
    s = str(s).lower()
    return re.sub(r'[ /，,。；;、\-\_\s]', '', s)


def is_event_match(ev1, ev2):
    """
    Determines whether two event dictionaries match by comparing their 'subject'
    and 'object' values using string normalization and substring inclusion checking.
    """
    if not isinstance(ev1, dict) or not isinstance(ev2, dict):
        return False
    s1, o1 = normalize_str(ev1.get('subject', '')), normalize_str(ev1.get('object', ''))
    s2, o2 = normalize_str(ev2.get('subject', '')), normalize_str(ev2.get('object', ''))
    subject_ok = (s1 in s2 or s2 in s1) if (s1 and s2) else (s1 == s2)
    object_ok = (o1 in o2 or o2 in o1) if (o1 and o2) else (o1 == o2)
    return subject_ok and object_ok


def get_event_dict(rel_dict, part="head"):
    """
    Retrieves a specific nested dictionary representing an event (e.g., 'head' or 'tail')
    from a relationship dictionary based on a given key string part.
    """
    if not isinstance(rel_dict, dict): return None
    for k, v in rel_dict.items():
        if part in k.lower():
            return v
    return None


def extract_combined_relations(json_str):
    """
    Parses a JSON-formatted string and extracts a combined list of 'cause_relations'
    and 'trigger_relations'. Handles basic formatting issues before parsing.
    """
    if not isinstance(json_str, str) or not json_str.strip():
        return []
    try:
        text = json_str.replace("None", "null").replace("'", '"')
        data = json.loads(text)
    except:
        try:
            data = ast.literal_eval(json_str)
        except:
            return []
    return data.get('cause_relations', []) + data.get('trigger_relations', [])


def calculate_combined_metrics(file_path):
    """
    Calculates and returns the precision, recall, and F1 score for combined event
    relation extraction by evaluating predictions against the ground truth in the given file.
    """
    df = pd.read_csv(file_path) if str(file_path).endswith('.csv') else pd.read_excel(file_path)
    tp, fp, fn = 0, 0, 0

    for _, row in df.iterrows():
        preds = extract_combined_relations(row['raw_json'])
        trues = extract_combined_relations(row['true_json'])

        temp_trues = trues[:]
        for p in preds:
            match_idx = -1
            p_head = get_event_dict(p, "head")
            p_tail = get_event_dict(p, "tail")

            for i, t in enumerate(temp_trues):
                t_head = get_event_dict(t, "head")
                t_tail = get_event_dict(t, "tail")

                if is_event_match(p_head, t_head) and is_event_match(p_tail, t_tail):
                    match_idx = i
                    break

            if match_idx != -1:
                tp += 1
                del temp_trues[match_idx]
            else:
                fp += 1
        fn += len(temp_trues)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def print_performance_table(shot_list, base_dir, methods=["default", "code"]):
    """
    Prints a formatted performance table in the console, comparing the precision,
    recall, and F1 scores of different extraction methods across various few-shot settings.
    """
    print("\n" + "=" * 90)
    print(f"{'Few-shot count':^15} | {'Method':^12} | {'Precision':^15} | {'Recall':^15} | {'F1 Score':^15}")
    print("-" * 90)

    for shot in shot_list:
        for i, method in enumerate(methods):
            f_path = Path(base_dir) / f"{method}-{shot}shot.csv"
            if not f_path.exists():
                f_path = Path(base_dir) / f"{method}-{shot}shot.xlsx"

            if f_path.exists():
                prec, rec, f1 = calculate_combined_metrics(f_path)
                shot_str = str(shot) if i == 0 else ""
                print(f"{shot_str:^15} | {method:^12} | {prec:>12.2%} | {rec:>12.2%} | {f1:^15.3f}")
            else:
                shot_str = str(shot) if i == 0 else ""
                print(f"{shot_str:^15} | {method:^12} | {'N/A':^15} | {'N/A':^15} | {'N/A':^15}")
        print("-" * 90)
    print("=" * 90 + "\n")


def plot_combined_relation_comparison(shot_list, base_dir, output_dir=None):
    """
    Generates, formats, and saves a line chart visualizing the comparison of F1 scores
    between different extraction methods across various few-shot data configurations.
    """
    results = {"default": [], "code": []}

    for method in ["default", "code"]:
        for shot in shot_list:
            f_path = Path(base_dir) / f"{method}-{shot}shot.csv"
            if not f_path.exists():
                f_path = Path(base_dir) / f"{method}-{shot}shot.xlsx"

            if f_path.exists():
                _, _, f1 = calculate_combined_metrics(f_path)
                results[method].append(f1)
            else:
                results[method].append(float('nan'))

    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    plt.plot(
        shot_list, results["default"],
        marker='o', color='#4878D0', linewidth=2, label='default'
    )

    plt.plot(
        shot_list, results["code"],
        marker='D', color='#6ACC64', linewidth=2, label='code'
    )

    import matplotlib.font_manager as fm
    font_path = 'C:/Windows/Fonts/msyh.ttc'

    if Path(font_path).exists():
        prop = fm.FontProperties(fname=font_path)
        font_path = 'C:/Windows/Fonts/msyh.ttc'

        if Path(font_path).exists():
            prop = fm.FontProperties(fname=font_path)
            plt.title('灾害观测事件关系识别任务 - 综合评估 (F1 Score)', fontproperties=prop, fontsize=14, pad=10)
            plt.xlabel('参考案例数量 (Number of Few-shot Samples)', fontproperties=prop, fontsize=12)
            plt.ylabel('F1 Score', fontproperties=prop, fontsize=12)
        else:
            print("Font not found; check C:/Windows/Fonts/")
    else:
        plt.title('Event Relation Extraction Performance (F1 Score)', fontsize=14, pad=10)
        plt.xlabel('Number of Few-shot Samples', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)

    plt.xticks(shot_list)
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)

    all_data = [value for value in results["default"] + results["code"] if pd.notna(value)]
    if all_data:
        d_min, d_max = min(all_data), max(all_data)
        margin = (d_max - d_min) * 0.2 if d_max != d_min else 0.1
        plt.ylim(max(0, d_min - margin), min(1.0, d_max + margin))

    for i in range(len(results["default"])):
        if pd.isna(results['default'][i]):
            continue
        plt.annotate(f"{results['default'][i]:.3f}",
                     (shot_list[i], results["default"][i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    for i in range(len(results["code"])):
        if pd.isna(results['code'][i]):
            continue
        plt.annotate(f"{results['code'][i]:.3f}",
                     (shot_list[i], results["code"][i]),
                     textcoords="offset points", xytext=(0, -20), ha='center')

    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()

    output_dir = Path(output_dir or base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined_relations_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
