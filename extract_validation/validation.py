import pandas as pd
import json
import ast
import re
import os


def normalize_text(text):
    """
    Standardizes the input text by removing non-alphanumeric and non-Chinese characters,
    and converting alphabetical characters to lowercase.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text).lower()


def is_fuzzy_match(pred, true, threshold=0.5):
    """
    Determines whether two text strings are semantically close or have a containment relationship
    based on character inclusion or Jaccard similarity threshold.
    """
    s1 = normalize_text(pred)
    s2 = normalize_text(true)

    if not s1 or not s2: return False

    if s1 in s2 or s2 in s1:
        return True

    set1, set2 = set(s1), set(s2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return (intersection / union) >= threshold if union > 0 else False


def event_validata_main(file_path):
    """
    Main evaluation function for validated structured event data (Disaster and Observation).
    Computes Precision, Recall, and F1 Scores for both Subject/Object entities and Arguments.
    """
    df = pd.read_excel(file_path)

    def extract_info_from_dict(data_str):
        """
        Extracts subjects, objects, and arguments from a structured JSON/dictionary string
        representing disaster and observation events.
        """
        if pd.isna(data_str) or str(data_str).strip() == '':
            return [], []

        try:
            data = ast.literal_eval(str(data_str))
        except Exception as e:
            return [], []

        sub_obj_list = []
        arg_list = []

        for item in data.get('disaster', []):
            if isinstance(item.get('feature'), dict):
                sub_obj_list.append(item['feature'].get('name'))
            if isinstance(item.get('impact'), dict):
                sub_obj_list.append(item['impact'].get('name'))

            arg_list.append(item.get('predicate'))
            arg_list.append(str(item.get('time')))

            factors = item.get('factors', {})
            if isinstance(factors, dict):
                for key in ['feature_factors', 'environment_factors']:
                    val = factors.get(key)
                    if isinstance(val, list):
                        arg_list.extend([str(v) for v in val])
                    elif isinstance(val, str):
                        arg_list.append(val)

        for item in data.get('observation', []):
            if isinstance(item.get('sensor'), dict):
                sub_obj_list.append(item['sensor'].get('name'))
            if isinstance(item.get('observed_object'), dict):
                sub_obj_list.append(item['observed_object'].get('name'))

            factors = item.get('factors', {})
            if isinstance(factors, dict):
                for key in ['feature_factors', 'environment_factors']:
                    val = factors.get(key)
                    if isinstance(val, list):
                        arg_list.extend([str(v) for v in val])

        sub_obj_list = [str(x) for x in sub_obj_list if x and str(x).strip() != 'None']
        arg_list = [str(x) for x in arg_list if x and str(x).strip() != 'None']

        return sub_obj_list, arg_list

    tp_so, fp_so, fn_so = 0, 0, 0
    tp_arg, fp_arg, fn_arg = 0, 0, 0

    for idx, row in df.iterrows():
        pred_so, pred_arg = extract_info_from_dict(row['raw_json'])
        true_so, true_arg = extract_info_from_dict(row['true_json'])

        current_true_so = true_so[:]

        for p in pred_so:
            match_found = False
            for i, t in enumerate(current_true_so):
                if is_fuzzy_match(p, t):
                    match_found = True
                    del current_true_so[i]
                    break
            if match_found:
                tp_so += 1
            else:
                fp_so += 1
        fn_so += len(current_true_so)

        current_true_arg = true_arg[:]

        for p in pred_arg:
            match_found = False
            for i, t in enumerate(current_true_arg):
                if is_fuzzy_match(p, t):
                    match_found = True
                    del current_true_arg[i]
                    break
            if match_found:
                tp_arg += 1
            else:
                fp_arg += 1
        fn_arg += len(current_true_arg)

    def calc_score(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f1

    p_so, r_so, f1_so = calc_score(tp_so, fp_so, fn_so)
    p_arg, r_arg, f1_arg = calc_score(tp_arg, fp_arg, fn_arg)

    results = pd.DataFrame({
        "评估维度": ["事件主体信息 (Sub/Obj)", "事件论元信息 (Args)"],
        "Precision (准确率)": [f"{p_so:.2%}", f"{p_arg:.2%}"],
        "Recall (召回率)": [f"{r_so:.2%}", f"{r_arg:.2%}"],
        "F1 Score": [f"{f1_so:.4f}", f"{f1_arg:.4f}"],
        "详情 (TP/FP/FN)": [f"{tp_so}/{fp_so}/{fn_so}", f"{tp_arg}/{fp_arg}/{fn_arg}"]
    })

    print("### 结构化数据评估结果 (基于 JSON 字段) ###")
    try:
        print(results.to_markdown(index=False))
    except:
        print(results)


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


def geo_validata_main(file_path):
    """
    Main evaluation function for geospatial data extraction. Evaluates geographic features,
    spatial relationships, and geographic type classification accuracy.
    """
    df = pd.read_excel(file_path)

    tp_f, fp_f, fn_f = 0, 0, 0
    tp_r, fp_r, fn_r = 0, 0, 0
    tp_f_type, matched_f_names = 0, 0

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
                matched_f_names += 1
                if str(pf.get('type')) == str(temp_true_f[match_idx].get('type')):
                    tp_f_type += 1
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

    def format_geo_row(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return f"{p:.2%}", f"{r:.2%}", f"{f1:.4f}", f"{tp}/{fp}/{fn}"

    p_f, r_f, f1_f, det_f = format_geo_row(tp_f, fp_f, fn_f)
    p_r, r_r, f1_r, det_r = format_geo_row(tp_r, fp_r, fn_r)
    type_acc = tp_f_type / matched_f_names if matched_f_names > 0 else 0
    det_type = f"{tp_f_type}/{matched_f_names - tp_f_type}/-"

    results = pd.DataFrame({
        "评估维度": ["地物实体 (Features)", "空间关系 (Relations)", "地物类型识别 (Type Acc)"],
        "Precision (准确率)": [p_f, p_r, f"{type_acc:.2%}"],
        "Recall (召回率)": [r_f, r_r, "-"],
        "F1 Score": [f1_f, f1_r, "-"],
        "详情 (TP/FP/FN)": [det_f, det_r, det_type]
    })

    print(f"\n### 地理空间抽取评估结果 (文件: {os.path.basename(file_path)}) ###")
    try:
        print(results.to_markdown(index=False))
    except ImportError:
        print(results.to_string(index=False))


def dataset_process_geo_main(file_path):
    """
    Processes a geographic dataset file by converting raw text extractions into
    standardized JSON strings and applying backoff compensation logic for missing ground truth values.
    """
    from geo_extraction import parse_geo_string_to_dict
    dataset = pd.read_excel(file_path)

    def get_json_string(text):
        if pd.isna(text) or str(text).strip() == '' or str(text).strip() == 'nan':
            return ""
        data_dict = parse_geo_string_to_dict(str(text))
        return json.dumps(data_dict, ensure_ascii=False)

    def process_row(row):
        raw_val = row['extraction_raw']
        true_val = row['extraction_true']

        raw_json = get_json_string(raw_val)

        if (pd.isna(true_val) or str(true_val).strip() == '') and not (pd.isna(raw_val) or str(raw_val).strip() == ''):
            effective_true_val = raw_val
        else:
            effective_true_val = true_val

        true_json = get_json_string(effective_true_val)

        return pd.Series([raw_json, true_json])

    dataset[['raw_json', 'true_json']] = dataset.apply(process_row, axis=1)

    print(dataset[['raw_json', 'true_json']].head())

    dataset.to_excel(file_path, index=False)


def dataset_process_event_main(file_path, only_raw):
    """
    Processes an event extraction dataset file by transforming raw text representations of
    disaster and observation records into structured dictionaries and saving them as JSON.
    """
    from event_extraction import parse_disaster_results, parse_observation_results
    dataset = pd.read_excel(file_path)

    def get_json_string(text):
        if pd.isna(text) or str(text).strip() == '' or str(text).strip() == 'nan':
            return [], []
        disaster_dict = parse_disaster_results(str(text))
        obs_dict = parse_observation_results(str(text))
        return disaster_dict, obs_dict

    def process_row(row):
        raw_val = row.get('extraction_raw', "")
        true_val = row.get('extraction_true', "")

        raw_json_dis, raw_json_obs = get_json_string(raw_val)
        raw_json = {'disaster': raw_json_dis, 'observation': raw_json_obs}

        if only_raw:
            true_json = ""
        else:
            if (pd.isna(true_val) or str(true_val).strip() == '') and not (
                    pd.isna(raw_val) or str(raw_val).strip() == ''):
                effective_true_val = raw_val
            else:
                effective_true_val = true_val

            true_json_dis, true_json_obs = get_json_string(effective_true_val)
            true_json = {'disaster': true_json_dis, 'observation': true_json_obs}

        return pd.Series({
            'raw_json': raw_json,
            'true_json': true_json
        })

    dataset[['raw_json', 'true_json']] = dataset.apply(process_row, axis=1)

    print(dataset[['raw_json', 'true_json']].head())

    dataset.to_excel(file_path, index=False)


def dataset_process_event_relation_main(file_path, only_raw=False):
    """
    Processes a relationship extraction dataset by compiling structural ground truth maps
    and model predictions for disaster cause and observation trigger relationships into JSON format.
    """
    from event_extraction import (parse_disaster_results, parse_observation_results,
                                  parse_disaster_cause_safe, parse_disaster_trigger_safe)
    dataset = pd.read_excel(file_path)

    def get_json_string_event(text: str):
        if pd.isna(text) or str(text).strip() == '' or str(text).strip() == 'nan':
            return [], []
        disaster_dict = parse_disaster_results(str(text))
        obs_dict = parse_observation_results(str(text))
        return disaster_dict, obs_dict

    def get_json_string_event_relation(text):
        if pd.isna(text) or str(text).strip() == '' or str(text).strip() == 'nan':
            return [], []
        cause_string = text.split('trigger_relations')[0].replace('cause_relations', 'results')
        parts = text.split('trigger_relations', 1)
        trigger_string = 'results' + parts[1] if len(parts) > 1 else ''
        relations = parse_disaster_cause_safe(cause_string)
        cause_dict = [{'head_disaster': {'subject': r.head_disaster.subject, 'object': r.head_disaster.object},
                       'tail_disaster': {'subject': r.tail_disaster.subject, 'object': r.tail_disaster.object},
                       'judgment_evidence': r.judgment_evidence} for r in relations]
        trigger_dict = parse_disaster_trigger_safe(trigger_string)
        return cause_dict, trigger_dict

    def process_row(row):
        true_val_event = row.get('event_extraction_true', "")

        raw_val = row.get('extraction_raw', "")
        true_val = row.get('extraction_true', "")

        json_dis, json_obs = get_json_string_event(true_val_event)
        json_event = {'disaster': json_dis, 'observation': json_obs}

        raw_json_cause, raw_json_trigger = get_json_string_event_relation(raw_val)
        raw_json = {'cause_relations': raw_json_cause, 'trigger_relations': raw_json_trigger}

        if only_raw:
            true_json = ""
        else:
            if (pd.isna(true_val) or str(true_val).strip() == '') and not (
                    pd.isna(raw_val) or str(raw_val).strip() == ''):
                effective_true_val = raw_val
            else:
                effective_true_val = true_val

            true_json_cause, true_json_trigger = get_json_string_event_relation(effective_true_val)
            true_json = {'cause_relations': true_json_cause, 'trigger_relations': true_json_trigger}

        return pd.Series({
            'event_true_json': json_event,
            'raw_json': raw_json,
            'true_json': true_json
        })

    dataset[['event_true_json', 'raw_json', 'true_json']] = dataset.apply(process_row, axis=1)

    print(dataset[['raw_json', 'true_json']].head())

    dataset.to_excel(file_path, index=False)


