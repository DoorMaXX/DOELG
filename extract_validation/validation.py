import pandas as pd
import json
import ast
import re


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
    """Evaluate the paper's combined event micro-F1 and return its component metrics."""
    from extract_validation.validation_plot_e import calculate_event_metrics
    result = calculate_event_metrics(file_path)
    print(f"Event micro-F1: {result['f1']:.6f}; Precision: {result['precision']:.6f}; Recall: {result['recall']:.6f}; Type accuracy: {result['type_accuracy']:.6f}")
    return result


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
    """Evaluate geographic entities and relations with the paper's combined micro-F1."""
    from extract_validation.validation_plot_geo import calculate_geo_separated_metrics
    result = {}
    for name, metrics in zip(('entities', 'relations', 'combined'), calculate_geo_separated_metrics(file_path)):
        result[name] = dict(zip(('precision', 'recall', 'f1'), metrics))
        print(f"{name}: Precision: {metrics[0]:.6f}; Recall: {metrics[1]:.6f}; F1: {metrics[2]:.6f}")
    return result


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
