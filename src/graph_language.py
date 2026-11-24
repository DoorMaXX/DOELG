"""Translate graph vocabulary before lookup and insertion, preserving source text."""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re

from src.construct_config import get_setting, resolve_project_path

_SUPPLEMENTARY_TERMS = {
    '高层建筑': 'High-rise Building', '烟囱': 'Chimney',
    '流域重叠': 'Basin Overlap', '流域不重叠': 'No Basin Overlap',
    '上下游': 'Upstream-Downstream', '位于下游': 'Downstream',
    '相接': 'Touches', '重叠': 'Overlap',
    '高频': 'High Frequency', '高分辨率': 'High Resolution',
}
_RELATION_TERMS = {
    '发生': 'Happen', '论元': 'Argument', '来源于': 'From', '存在': 'Exist',
    '引起': 'Cause', '观测因子': 'Observed Factor', '发生时间': 'Occurred At',
    '对齐': 'Alignment', '空间几何': 'Geometry', '包括': 'Include', '关联': 'Associate',
    '映射': 'Mapping', '包含': 'Contains', '被包含': 'Contained By',
    '相交': 'Intersect', '相等': 'Equal', '相离': 'Disjoint', '相邻': 'Adjacent',
    '相接': 'Touches', '重叠': 'Overlap', '毗邻': 'Adjacent', '邻接': 'Adjacent',
    '连接': 'Connected To', '相连': 'Connected To', '连通': 'Connected To',
    '临近': 'Near', '邻近': 'Near', '靠近': 'Near', '附近': 'Nearby', '远离': 'Far From',
    '位于': 'Located In', '属于': 'Part Of', '流经': 'Flows Through', '穿过': 'Crosses',
    '上游': 'Upstream', '下游': 'Downstream', '属于下游': 'Downstream Of',
    '属于上游': 'Upstream Of', '位于下游': 'Downstream Of', '位于上游': 'Upstream Of',
    '以东': 'East Of', '以西': 'West Of', '以南': 'South Of', '以北': 'North Of',
    '东侧': 'East Of', '西侧': 'West Of', '南侧': 'South Of', '北侧': 'North Of',
    '高海拔': 'High Elevation', '低海拔': 'Low Elevation',
    '中心': 'Center', '四周': 'Surrounding',
}


def contains_chinese(value):
    """Return whether a value contains Chinese ideographs."""
    return bool(re.search(r'[\u3400-\u9fff\U00020000-\U0003134f]', str(value)))


def _term_lookup_key(value):
    """Normalize punctuation variants for dictionary lookup without changing stored text."""
    punctuation = str.maketrans({'、': ',', '，': ',', '（': '(', '）': ')'})
    return re.sub(r'\s*([,()])\s*', r'\1', value.translate(punctuation)).strip()


@lru_cache(maxsize=8)
def _read_dictionary(path):
    """Load canonical translations and unambiguous punctuation aliases for each path."""
    try:
        with Path(path).open(encoding='utf-8-sig') as stream:
            dictionary = json.load(stream)
        if not isinstance(dictionary, dict):
            raise ValueError('Expected a JSON object')
    except (OSError, ValueError) as error:
        print(f'[Translation] Dictionary unavailable ({type(error).__name__}): {path}. Using model fallback.')
        dictionary = {}
    dictionary = {key: value.strip() for key, value in dictionary.items()
                  if isinstance(key, str) and isinstance(value, str) and value.strip()}
    translations = {**_SUPPLEMENTARY_TERMS, **dictionary}
    aliases = {}
    for term, translation in translations.items():
        aliases.setdefault(_term_lookup_key(term), set()).add(translation)
    aliases = {key: next(iter(values)) for key, values in aliases.items() if len(values) == 1}
    return {**aliases, **translations}


def translation_dictionary():
    """Return the project-local vocabulary selected by the construction config."""
    try:
        path = get_setting('translation.dictionary_path')
    except KeyError:
        path = 'data/dictionary/dictionary.json'
    return _read_dictionary(str(resolve_project_path(path)))


def english_term(value, strict=False):
    """Translate a dictionary term or punctuation variant, retaining unknown proper names."""
    if value is None:
        return value
    text = str(value).strip()
    dictionary = translation_dictionary()
    translated = dictionary.get(text) or dictionary.get(_term_lookup_key(text)) or text
    if strict and contains_chinese(translated):
        return _translate_missing_term(text)
    return translated


def _translate_term_with_model(text):
    """Translate an unseen graph term with bounded requests to the construction model."""
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    from src.construct_config import get_chat_model_options

    class TranslationResult(BaseModel):
        english_term: str = Field(description='Professional English graph term without explanation')

    options = get_chat_model_options('geo')
    options.setdefault('timeout', 30)
    options.setdefault('max_retries', 1)
    agent = create_agent(
        ChatOpenAI(**options),
        system_prompt='Translate the Chinese geospatial, disaster-monitoring, sensor, or spatial-relation '
                      'term into concise professional English. Preserve its meaning and any relation '
                      'direction. Return only the requested English term.',
        response_format=TranslationResult,
    )
    result = agent.invoke({'messages': [{'role': 'user', 'content': text}]},
                          config={'recursion_limit': 5})
    translated = result['structured_response'].english_term.strip()
    if not translated or contains_chinese(translated):
        raise ValueError('The model did not return a nonempty English term')
    return translated


@lru_cache(maxsize=2048)
def _translate_missing_term(text):
    """Cache successful or failed translations and return an empty value on failure."""
    try:
        translated = _translate_term_with_model(text)
        if not isinstance(translated, str) or not translated.strip() or contains_chinese(translated):
            raise ValueError('The model did not return a nonempty English term')
        print(f'[Translation] {text!r} -> {translated.strip()!r}')
        return translated.strip()
    except Exception as error:
        print(f'[Translation skipped] {text!r}: {type(error).__name__}. '
              'No English value is available; affected records will be skipped.')
        return ''


def _translate_relation_with_model(text):
    """Use the same nonfatal translation cache for unseen spatial predicates."""
    return _translate_missing_term(text)


def english_relation(value):
    """Translate spatial predicates and semicolon-separated relationship labels."""
    if not value:
        return value
    parts = []
    for part in str(value).split(';'):
        text = part.strip()
        translated = _RELATION_TERMS.get(text) or english_term(text)
        if contains_chinese(translated):
            translated = _translate_relation_with_model(text)
        if translated and translated not in parts:
            parts.append(translated)
    return ';'.join(parts)


def english_geo(feature):
    """Copy a geographic argument and translate its category and generic name."""
    result = deepcopy(feature)
    if not isinstance(result, dict):
        return result
    if 'name' in result:
        result['name'] = english_term(result['name'])
    for key in ('geo_type', 'type'):
        if key in result:
            result[key] = english_term(result[key], strict=True)
    return result


def english_disaster(disaster):
    """Normalize disaster categories while retaining extracted surface predicates."""
    result = deepcopy(disaster)
    result['feature'] = english_geo(result['feature'])
    result['impact']['impact_type'] = english_term(result['impact']['impact_type'], strict=True)
    return result


def english_observation(observation):
    """Normalize sensor categories, platforms, and geographic observation arguments."""
    result = deepcopy(observation)
    sensor = result.get('sensor')
    if isinstance(sensor, dict):
        for key in ('type', 'platform'):
            if key in sensor:
                sensor[key] = english_term(sensor[key], strict=True)
    if isinstance(result.get('observed_object'), dict):
        result['observed_object'] = english_geo(result['observed_object'])
    return result


def english_geo_results(results):
    """Normalize geographic extraction output before any graph lookup or write."""
    result = deepcopy(results)
    features = []
    for feature in result['features']:
        translated = english_geo(feature)
        if not translated.get('type', translated.get('geo_type')):
            print(f"[Geo skipped] {feature.get('name')!r}: no English geographic type.")
            continue
        features.append(translated)
    result['features'] = features
    names = {feature.get('name') for feature in features}
    relations = []
    for relation in result['spatial_relations']:
        for key in ('subject', 'object'):
            relation[key] = english_term(relation[key])
        relation['relation_type'] = english_relation(relation['relation_type'])
        if relation['subject'] in names and relation['object'] in names and relation['relation_type']:
            relations.append(relation)
        else:
            print(f"[Spatial relation skipped] {relation['subject']!r} -> {relation['object']!r}: "
                  'an endpoint or English predicate is unavailable.')
    result['spatial_relations'] = relations
    return result


def english_event_description(description):
    """Translate both category components of an abstract disaster description."""
    geo, impact = description.split(',', 1)
    return f'{english_term(geo, strict=True)},{english_term(impact, strict=True)}'


def split_event_pair(label, geo_types, impact_types):
    """Split a typed event pair without splitting hyphens inside English categories."""
    matches = []
    for index, character in enumerate(label):
        if character != '-':
            continue
        left, right = label[:index], label[index + 1:]
        try:
            left = english_event_description(left)
            right = english_event_description(right)
        except ValueError:
            continue
        if all(desc.split(',', 1)[0] in geo_types and desc.split(',', 1)[1] in impact_types
               for desc in (left, right)):
            matches.append((left, right))
    if len(matches) != 1:
        raise ValueError(f'Expected one typed disaster pair: {label!r}')
    return matches[0]
