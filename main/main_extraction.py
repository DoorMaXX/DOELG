import argparse
import os
import sys
import time
import json
import concurrent.futures
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.construct_config import (
    configure_from_cli, get_config_path, get_configured_path, get_setting, resolve_project_path, resolve_database_name
)
from src.neo4j_database import ensure_database
from src.graph_language import english_disaster, english_observation, english_term
from src.other.causal_relation_rules import classify_event_cause, enforce_existing_causal_rules

configure_from_cli()

from event_extraction import (
    event_extraction_combined,
    parse_disaster_results,
    parse_observation_results,
    update_D_argument_relations,
    _worker_cause_relation,
    _worker_trigger_relation
)

from geo_extraction import (
        geo_ner_re,
        parse_geo_string_to_dict,
        insert_database_geo,
        is_geo_location_parallel,
        event_pair_geo_encoding_parallel,
        main_calculate_spatial_relations
)

from ontology.neo4jmodel import (
    Neo4jDatabaseConnector,
    DisasterEvent,
    OriginText,
    EnvironmentDescription,
    GeoObject,
    Time,
    Factor,
    Sensor,
    ObservationEvent,
    Impact,
    ObservationPattern
)

from fusion import (
    geo_fusion,
    impact_fusion,
    factor_fusion,
    sensor_fusion,
    disaster_event_fusion,
    obs_event_fusion,
    relation_deduplication
)
from other.environment_descrption import get_event_envDes_llm
from other.observation_pattern import get_event_obsPat_llm, get_vector_store

# ==========================================
# 1. DATABASE INGESTION LOGIC (Events)
# ==========================================

def insert_database(disaster: dict, origin_text: str, title: str, event_num: int):
    """
    Ingests a parsed disaster event dictionary into the Neo4j graph database.
    Creates and links OriginText, GeoObject, TargetImpactNode, Factor, Time, and DisasterEvent nodes.
    """
    disaster = english_disaster(disaster)
    if not disaster['feature'].get('geo_type') or not disaster['impact'].get('impact_type'):
        print(f'[Disaster skipped] Record {event_num}: an English argument category is unavailable.')
        return 0
    origin_text_node = OriginText.nodes.first_or_none(text_content=origin_text)
    if not origin_text_node:
        origin_text_node = OriginText(text_content=origin_text, text_title=title).save()

    geo_node = GeoObject.nodes.first_or_none(geo_object_name=disaster["feature"]["name"])
    if not geo_node:
        geo_node = GeoObject(
            geo_object_name=disaster["feature"]["name"],
            geo_type=disaster["feature"]["geo_type"],
            geometry=disaster["feature"]["geometry"]
        ).save()

    impact_node = Impact.nodes.first_or_none(impact_name=disaster["impact"]["name"],
                                                       impact_type=disaster["impact"]["impact_type"])
    if not impact_node:
        impact_node = Impact(impact_name=disaster["impact"]["name"],
                                       impact_type=disaster["impact"]["impact_type"]).save()

    if not geo_node.happen.is_connected(impact_node):
        geo_node.happen.connect(impact_node)

    event_desc = f"{disaster['feature']['name']},{disaster['impact']['name']}"
    event_node = DisasterEvent(event_num=event_num, event_description=event_desc).save()

    for factor_type, factor_list in disaster.get("factors", {}).items():
        for factor in factor_list:
            factor_node = Factor.nodes.first_or_none(factor_type=factor_type, factor_name=factor)
            if not factor_node:
                factor_node = Factor(factor_type=factor_type, factor_name=factor).save()

            if not event_node.observed.is_connected(factor_node):
                event_node.observed.connect(factor_node)

    time_val = disaster.get("time")
    if time_val and str(time_val).strip() != 'None':
        time_node = Time.nodes.first_or_none(time_value=str(time_val))
        if not time_node:
            time_node = Time(time_value=str(time_val)).save()

        if not event_node.occurred_at.is_connected(time_node):
            event_node.occurred_at.connect(time_node)

    event_node.geo_argument.connect(geo_node)
    event_node.impact_argument.connect(impact_node)
    origin_text_node.exist.connect(event_node)
    event_node.origin.connect(origin_text_node)

    update_D_argument_relations(event_node)
    return 1


def insert_database_obs(observation: dict, origin_text: str, event_num: int):
    """
    Ingests a parsed observation event dictionary into the Neo4j graph database.
    Creates and links Sensor, GeoObject, Factor, Time, and ObservationEvent nodes[cite: 17].
    """
    observation = english_observation(observation)
    sensor = observation.get("sensor", {})
    sensor_name, sensor_type, sensor_platform = "", "", ""
    if isinstance(sensor, dict):
        sensor_name = sensor.get("name", "")
        sensor_type = sensor.get("type", "")
        sensor_platform = sensor.get("platform", "")

    sensor_node = Sensor.nodes.first_or_none(sensor_name=sensor_name,
                                             sensor_type=sensor_type,
                                             platform=sensor_platform)
    if not sensor_node and sensor_name and sensor_type and sensor_platform:
        sensor_node = Sensor(sensor_name=sensor_name,
                             sensor_type=sensor_type,
                             platform=sensor_platform).save()

    if not sensor_node:
        print(f'[Observation skipped] Record {event_num}: sensor name, English type, or platform is missing.')
        return 0

    observed_object = observation.get("observed_object", {})
    observed_object_name, observed_object_type, observed_object_geometry = "", "", ""
    if isinstance(observed_object, dict):
        observed_object_name = observed_object.get("name", "")
        observed_object_type = observed_object.get("geo_type", "")
        observed_object_geometry = observed_object.get("geometry", "")

    geo_node = GeoObject.nodes.first_or_none(geo_object_name=observed_object_name)
    if not geo_node and observed_object_name and observed_object_type and observed_object_geometry:
        geo_node = GeoObject(
            geo_object_name=observed_object_name,
            geo_type=observed_object_type,
            geometry=observed_object_geometry
        ).save()

    factor_nodes = []
    for factor_type, factor_list in observation.get("factors", {}).items():
        for factor in factor_list:
            if not factor:
                continue
            factor_node = Factor.nodes.first_or_none(factor_type=factor_type, factor_name=factor)
            if not factor_node:
                factor_node = Factor(factor_type=factor_type, factor_name=factor).save()
            if factor_node:
                factor_nodes.append(factor_node)

    if not geo_node and len(factor_nodes) == 0:
        print(f'[Observation skipped] Record {event_num}: no valid observed object or factor.')
        return 0

    origin_text_node = OriginText.nodes.first_or_none(text_content=origin_text)
    if not origin_text_node:
        origin_text_node = OriginText(text_content=origin_text).save()

    time_val = observation.get('time')
    time_node = None
    if time_val is not None and str(time_val).strip() != 'None':
        time_node = Time.nodes.first_or_none(time_value=str(time_val))
        if not time_node:
            time_node = Time(time_value=str(time_val)).save()

    obs_events = []
    observe_span = observation.get('observed_span', '')

    for factor_node in factor_nodes:
        if geo_node:
            event_desc = f"{sensor_name},[{observed_object_name}, {factor_node.factor_name}]"
            event_node = ObservationEvent(event_num=event_num, event_description=event_desc,
                                          observe_span=observe_span).save()

            event_node.geo_argument.connect(geo_node)
            sensor_node.observe_geo.connect(geo_node)

            if factor_node.factor_type == 'feature_factors':
                geo_node.has_factor.connect(factor_node)
        else:
            event_desc = f"{sensor_name},[None, {factor_node.factor_name}]"
            event_node = ObservationEvent(event_num=event_num, event_description=event_desc,
                                          observe_span=observe_span).save()

        event_node.sensor_argument.connect(sensor_node)
        event_node.factor_argument.connect(factor_node)
        sensor_node.observe_factor.connect(factor_node)

        origin_text_node.exist_obs.connect(event_node)
        event_node.origin.connect(origin_text_node)

        if time_node and not event_node.occurred_at.is_connected(time_node):
            event_node.occurred_at.connect(time_node)

        obs_events.append(event_node)

    if len(obs_events) == 0 and geo_node:
        event_desc = f"{sensor_name},[{observed_object_name}, None]"
        event_node = ObservationEvent(event_num=event_num, event_description=event_desc,
                                      observe_span=observe_span).save()

        event_node.geo_argument.connect(geo_node)
        event_node.sensor_argument.connect(sensor_node)
        sensor_node.observe_geo.connect(geo_node)

        origin_text_node.exist_obs.connect(event_node)
        event_node.origin.connect(origin_text_node)

        if time_node and not event_node.occurred_at.is_connected(time_node):
            event_node.occurred_at.connect(time_node)
        obs_events.append(event_node)

    return len(obs_events)


# ==========================================
# 2. FILE PARSING LOGIC
# ==========================================

def parse_tasks_from_json(file_path):
    """
    Dynamically parses different JSON formats to extract text chunks for processing.
    """
    tasks = []
    index = 0
    default_title = os.path.basename(file_path).replace('.json', '')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            return tasks

        try:
            data = json.loads(content)
            if isinstance(data, list):
                for obj in data:
                    if not isinstance(obj, dict): continue
                    title = obj.get('wiki_title', obj.get('title', default_title))
                    chunks = obj.get('content_chunks', obj.get('content', []))
                    if isinstance(chunks, str):
                        chunks = [chunks]
                    for text in chunks:
                        if isinstance(text, str) and text.strip():
                            tasks.append({'text': text.strip(), 'title': title, 'index': index})
                            index += 1
            elif isinstance(data, dict):
                for title_key, chunks in data.items():
                    title = f"{default_title}_{title_key}"
                    if isinstance(chunks, str):
                        chunks = [chunks]
                    elif not isinstance(chunks, list):
                        continue
                    for text in chunks:
                        if isinstance(text, str) and text.strip():
                            tasks.append({'text': text.strip(), 'title': title, 'index': index})
                            index += 1
        except json.JSONDecodeError:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict): continue
                    text = obj.get('text', '')
                    title = obj.get('title', default_title)
                    if text.strip():
                        tasks.append({'text': text.strip(), 'title': title, 'index': index})
                        index += 1
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return tasks


# ==========================================
# 3. PHASE 1: GEO EXTRACTION
# ==========================================

def process_geo_item(task):
    """
    Worker function to process a single text task using the LLM for Geo extraction[cite: 16, 18].
    """
    text = task.get('text', '')
    title_key = task.get('title', '')

    if not text:
        return None
    try:
        raw_result = geo_ner_re(text)

        raw_result_clean = (raw_result
                            .replace('：', ':')
                            .replace('，', ',')
                            .replace('“', '"')
                            .replace('”', '"')
                            .replace('（', '(')
                            .replace('）', ')')
                            .replace('、', ',')
                            .replace('。', '')
                            .replace("‘", "'")
                            .replace("’", "'"))

        parsed = parse_geo_string_to_dict(raw_result_clean)

        return {
            'results': parsed,
            'original_text': text,
            'title_key': title_key
        }
    except Exception as e:
        print(f"Geo Extraction Error under title {title_key}: {e}")
        return None


def run_geo_extraction(input_dir, max_workers):
    """
    Phase 1: Extracts geographic entities and spatial relations, then saves to Neo4j[cite: 16].
    """
    print(f"\n{'=' * 50}")
    print(f"🌍 [PHASE 1] Starting Geographic Entity Extraction (Input: {input_dir})")
    print(f"{'=' * 50}")

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    for file in files:
        print(f"📂 Geo Processing file: {file}")
        tasks_data = parse_tasks_from_json(file)
        if not tasks_data:
            continue

        collected_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process_geo_item, task): task for task in tasks_data}
            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(tasks_data), desc="Geo Ext"):
                res = future.result()
                if res:
                    collected_data.append(res)

        if collected_data:
            written = 0
            for item in collected_data:
                written += bool(insert_database_geo(item['results'], item['original_text'], item['title_key']))
            print(f"Geo processing complete for {os.path.basename(file)}: "
                  f"{written} text records written, {len(collected_data) - written} skipped.")


# ==========================================
# 4. PHASE 2: EVENT EXTRACTION
# ==========================================

def process_event_item(task):
    """
    Worker function to process a single text task using the LLM for both disaster
    and observation event extraction[cite: 17].
    """
    text = task.get('text', '')
    title_key = task.get('title', '')
    index = task.get('index', 0)

    if not text:
        return None
    try:
        raw_result = event_extraction_combined(text)
        raw_result_clean = (raw_result
                            .replace('：', ':')
                            .replace('，', ',')
                            .replace('“', '"')
                            .replace('”', '"')
                            .replace('（', '(')
                            .replace('）', ')')
                            .replace('、', ',')
                            .replace('。', '')
                            .replace("‘", "'")
                            .replace("’", "'"))

        disasters = parse_disaster_results(raw_result_clean)
        observations = parse_observation_results(raw_result_clean)

        return {
            'disasters': disasters,
            'observations': observations,
            'original_text': text,
            'title_key': title_key,
            'index': index
        }
    except Exception as e:
        print(f"Event Extraction Error under title {title_key}: {e}")
        return None


def run_event_extraction(input_dir, max_workers):
    """
    Phase 2: Extracts disaster and observation events, then saves to Neo4j.
    """
    print(f"\n{'=' * 50}")
    print(f"🔥 [PHASE 2] Starting Disaster & Observation Event Extraction (Input: {input_dir})")
    print(f"{'=' * 50}")

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    for file in files:
        print(f"📂 Event Processing file: {file}")
        tasks_data = parse_tasks_from_json(file)
        if not tasks_data:
            continue

        collected_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process_event_item, task): task for task in tasks_data}
            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(tasks_data),
                               desc="Event Ext"):
                res = future.result()
                if res:
                    collected_data.append(res)

        if collected_data:
            disaster_count = observation_count = skipped = 0
            for item in collected_data:
                for dis in item['disasters']:
                    if dis.get('feature') and dis.get('impact'):
                        count = insert_database(dis, item['original_text'], item['title_key'], item['index'])
                        disaster_count += count
                        skipped += count == 0
                for obs in item['observations']:
                    count = insert_database_obs(obs, item['original_text'], item['index'])
                    observation_count += count
                    skipped += count == 0
            print(f"Event processing complete for {os.path.basename(file)}: "
                  f"{disaster_count} disaster events, {observation_count} observation events written; "
                  f"{skipped} candidate events skipped.")


# ==========================================
# 5. PHASE 3: RELATION RECOGNITION
# ==========================================


def run_relation_recognition(max_workers):
    """
    Phase 3: Extracts cause and trigger relations between events and links them in Neo4j[cite: 16, 17].
    """
    print(f"\n{'=' * 50}")
    print(f"🔗 [PHASE 3] Starting Event Relation Recognition (Cause & Trigger)")
    print(f"{'=' * 50}")

    existing = enforce_existing_causal_rules(DisasterEvent.nodes.all())
    print(f"Appendix E: {existing['removed']} existing CAUSE edges removed; "
          f"{existing['updated']} classified as Immediate/Altering.")

    text_nodes = list(OriginText.nodes.all())
    cause_tasks = []
    trigger_tasks = []

    # Prepare Tasks
    for text_node in text_nodes:
        text = text_node.text_content
        events_dict = {}
        disaster_events = []
        obs_events = []

        # Check Disaster Events
        for event_node in text_node.exist.all():
            if len(event_node.geo_argument) == 1 and len(event_node.impact_argument) == 1:
                geo_name = event_node.geo_argument.all()[0].geo_object_name
                impact_name = event_node.impact_argument.all()[0].impact_name
                desc = f"Disaster(subject='{geo_name}', predicate='发生', object='{impact_name}')"
                disaster_events.append(desc)
                events_dict[desc] = event_node.node_id

        # Check Observation Events
        for obs_event_node in text_node.exist_obs.all():
            if len(obs_event_node.sensor_argument) == 1:
                sensor_name = obs_event_node.sensor_argument.all()[0].sensor_name
                geo_name = obs_event_node.geo_argument.all()[0].geo_object_name if len(
                    obs_event_node.geo_argument) == 1 else ''
                factor_name = obs_event_node.factor_argument.all()[0].factor_name if len(
                    obs_event_node.factor_argument) == 1 else ''
                desc = f"Observation(subject='{sensor_name}', predicate='观测', object=[{geo_name}, {factor_name}])"
                obs_events.append(desc)
                events_dict[desc] = obs_event_node.node_id

        # Task generation condition for Cause
        if len(disaster_events) > 1 and not any(event.cause.all() for event in text_node.exist.all()):
            cause_tasks.append({
                'text': text,
                'events': disaster_events,
                'events_dict': events_dict
            })

        # Task generation condition for Trigger
        if len(disaster_events) >= 1 and len(obs_events) >= 1:
            trigger_tasks.append({
                'text': text,
                'disasters': disaster_events,
                'observations': obs_events,
                'events_dict': events_dict
            })

    # Process Cause Relations
    final_cause_jobs = []
    if cause_tasks:
        print(f"-> Processing {len(cause_tasks)} tasks for Cause relations...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(_worker_cause_relation, t['text'], t['events'], t['events_dict']): t for t
                              in cause_tasks}
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(cause_tasks), desc="Cause"):
                res = future.result()
                if res: final_cause_jobs.extend(res)

    # Process Trigger Relations
    final_trigger_jobs = []
    if trigger_tasks:
        print(f"-> Processing {len(trigger_tasks)} tasks for Trigger relations...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(_worker_trigger_relation, t['text'], t['disasters'], t['observations'],
                                              t['events_dict']): t for t in trigger_tasks}
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(trigger_tasks),
                               desc="Trigger"):
                res = future.result()
                if res: final_trigger_jobs.extend(res)

    # DB Ingestion (Main Thread)
    print(f"-> Syncing {len(final_cause_jobs)} Cause and {len(final_trigger_jobs)} Trigger edges to Neo4j...")
    filtered_cause_jobs = 0
    for job in final_cause_jobs:
        h_id, t_id = job.get('head_id'), job.get('tail_id')
        if h_id and t_id:
            e1 = DisasterEvent.nodes.first_or_none(node_id=h_id)
            e2 = DisasterEvent.nodes.first_or_none(node_id=t_id)
            if e1 and e2:
                metadata = classify_event_cause(e1, e2)
                if metadata is None:
                    filtered_cause_jobs += 1
                    print(f'[Causal rule skipped] {e1.event_description} -> {e2.event_description}')
                    continue
                if not e1.cause.is_connected(e2):
                    e1.cause.connect(e2, {'evidence': job['evidence'], 'relation_count': 1, **metadata})
                else:
                    rel = e1.cause.relationship(e2)
                    rel.relation_count = (rel.relation_count or 1) + 1
                    rel.evidence = (rel.evidence or "") + " || " + job['evidence']
                    for key, value in metadata.items():
                        setattr(rel, key, value)
                    rel.save()

    for job in final_trigger_jobs:
        h_id, t_id = job.get('head_id'), job.get('tail_id')
        if h_id and t_id:
            e1 = DisasterEvent.nodes.first_or_none(node_id=h_id)
            e2 = ObservationEvent.nodes.first_or_none(node_id=t_id)
            if e1 and e2:
                if not e1.trigger.is_connected(e2):
                    e1.trigger.connect(e2, {'evidence': job['evidence'], 'relation_count': 1})
                else:
                    rel = e1.trigger.relationship(e2)
                    rel.relation_count = (rel.relation_count or 1) + 1
                    rel.evidence = (rel.evidence or "") + " || " + job['evidence']
                    rel.save()

    print(f'Relation recognition completed: {filtered_cause_jobs} candidate CAUSE edges rejected by Appendix E.')


def run_extraction_pipeline(input_dir, max_workers):
    """
    Executes the three-phase extraction pipeline strictly in order.
    1. Geo Extraction
    2. Event Extraction
    3. Relation Recognition
    """
    start_time = time.time()

    run_geo_extraction(input_dir, max_workers)
    run_event_extraction(input_dir, max_workers)
    run_relation_recognition(max_workers)

    print(f"\n✅ End-to-end extraction pipeline completed in: {time.time() - start_time:.2f} seconds")
    return True


# ==========================================
# 6. GRAPH NODE & EDGE FUSION WORKFLOW
# ==========================================

def run_fusion():
    """
    Executes the complete fusion pipeline:
    1. Argument Node Fusion
    2. Event Node Fusion
    3. Relation Deduplication
    """
    start_time = time.time()
    print(f"\n{'=' * 50}")
    print(f"🧩 [FUSION] Starting Complete Graph Fusion Pipeline")
    print(f"{'=' * 50}")

    print("-> [FUSION STEP 1/3] Aligning Argument Entities (Geo, Impact, Factor, Sensor)...")
    geo_fusion()
    impact_fusion()
    factor_fusion()
    sensor_fusion()

    print("\n-> [FUSION STEP 2/3] Resolving Duplicate Event Instances...")
    disaster_event_fusion()
    obs_event_fusion()

    print("\n-> [FUSION STEP 3/3] Deduplicating and Aggregating Relations...")
    relation_deduplication()
    summary = enforce_existing_causal_rules(DisasterEvent.nodes.all())
    print(f"Appendix E after fusion: {summary['removed']} CAUSE edges removed; "
          f"{summary['updated']} classified as Immediate/Altering.")

    print(f"\n✅ Complete fusion pipeline finished in: {time.time() - start_time:.2f} seconds")
    return True


# ==========================================
# 7. HAZARD-FORMATIVE ENVIRONMENT LOGIC
# ==========================================

def fetch_env_des_from_llm(event_id, origin_t, event_desc):
        try:
                res_info = get_event_envDes_llm(origin_t, event_desc)
                return {
                        "event_id": event_id,
                        "event_des": event_desc,
                        "env_des_list": res_info.env_des if hasattr(res_info, 'env_des') else getattr(res_info, 'env_des', [])
                }
        except Exception as e:
                print(f"LLM extraction error for event {event_id}: {e}")
                return None

def run_env_description(max_workers):
        print(f"\n{'=' * 50}")
        print(f"🌍 [ENV] Starting Hazard-Formative Environment Extraction")
        print(f"{'=' * 50}")

        start_time = time.time()
        disaster_event_nodes = DisasterEvent.nodes.all()

        tasks = []
        for en in disaster_event_nodes:
                if not en.envDes_argument.all():
                        event_id = en.node_id
                        event_desc = en.event_description
                        text_nodes = en.origin.all()
                        text_contents = [tn.text_content for tn in text_nodes]
                        origin_t = "".join(text_contents)

                        if origin_t.strip():
                                tasks.append((event_id, origin_t, event_desc))

        if not tasks:
                print("✅ All disaster events already have environment descriptions assigned.")
                return True

        print(f"Found {len(tasks)} target disaster event nodes requiring updates. Processing...")
        llm_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_env = {
                        executor.submit(fetch_env_des_from_llm, eid, txt, desc): eid
                        for eid, txt, desc in tasks
                }
                for future in tqdm(concurrent.futures.as_completed(future_to_env), total=len(tasks), desc="Env Ext"):
                        res = future.result()
                        if res:
                                llm_results.append(res)

        for res in llm_results:
                try:
                        node = DisasterEvent.nodes.get(node_id=res['event_id'])
                        event_envDes_str = "-".join(res['env_des_list'])

                        if not event_envDes_str:
                                continue

                        envDes_node = EnvironmentDescription.nodes.first_or_none(description=event_envDes_str)
                        if not envDes_node:
                                envDes_node = EnvironmentDescription(description=event_envDes_str).save()

                        if not node.envDes_argument.is_connected(envDes_node):
                                node.envDes_argument.connect(envDes_node)
                except Exception as e:
                        print(f"Database sync failed for event ID {res['event_id']}: {e}")

        print(f"✅ Hazard-formative environment analysis finished. Cost: {time.time() - start_time:.2f} seconds")
        return True


# ==========================================
# 8. OBSERVATION PATTERN GENERATION
# ==========================================

def _worker_obs_pattern(obs_id, dis_id, origin_t, obs_desc, dis_desc):
        """Worker to call LLM for observation pattern analysis."""
        try:
                res_info = get_event_obsPat_llm(origin_t, obs_desc, dis_desc)
                return {
                        "obs_id": obs_id,
                        "dis_id": dis_id,
                        "pattern": res_info
                }
        except Exception as e:
                print(f"Observation Pattern LLM extraction error for event {obs_id}: {e}")
                return None

def run_obs_pattern(max_workers):
        """
        Extracts observation patterns for ObservationEvents triggered by DisasterEvents.
        """
        print(f"\n{'=' * 50}")
        print(f"📡 [OBS PATTERN] Starting Observation Pattern Extraction")
        print(f"{'=' * 50}")

        start_time = time.time()
        disaster_nodes = DisasterEvent.nodes.all()
        tasks = []

        for de in disaster_nodes:
                obs_nodes = de.trigger.all()
                for obs_node in obs_nodes:
                        if not obs_node.obsPat_argument.all():
                                text_nodes = obs_node.origin.all()
                                text_contents = [t.text_content for t in text_nodes if t.exist.is_connected(de)]
                                if text_contents:
                                        origin_text = "\n".join(text_contents)
                                        tasks.append({
                                                'obs_id': obs_node.node_id,
                                                'dis_id': de.node_id,
                                                'obs_desc': obs_node.event_description,
                                                'dis_desc': de.event_description,
                                                'origin_t': origin_text
                                        })

        if not tasks:
                print("✅ All targeted observation events already have observation patterns assigned.")
                return True

        print(f"Found {len(tasks)} observation events requiring patterns. Processing...")
        try:
                print('Initializing the sensor manual index before starting observation workers...')
                get_vector_store()
        except Exception as error:
                print(f'Observation pattern extraction stopped: sensor manual initialization failed: {error}')
                return False
        llm_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_obs = {
                        executor.submit(
                                _worker_obs_pattern,
                                t['obs_id'], t['dis_id'], t['origin_t'], t['obs_desc'], t['dis_desc']
                        ): t['obs_id']
                        for t in tasks
                }
                for future in tqdm(concurrent.futures.as_completed(future_to_obs), total=len(tasks), desc="ObsPat Ext"):
                        res = future.result()
                        if res:
                                llm_results.append(res)

        print(f"Inference complete. Synchronizing {len(llm_results)} records back to Neo4j...")
        saved = 0
        failed = len(tasks) - len(llm_results)
        for res in llm_results:
                try:
                        obs_node = ObservationEvent.nodes.get(node_id=res['obs_id'])
                        dis_node = DisasterEvent.nodes.get(node_id=res['dis_id'])
                        pat = res['pattern']
                        weather_effect = english_term(pat.weather_effect, strict=True)
                        obs_position = english_term(pat.obs_position, strict=True)

                        obsPat_node = ObservationPattern.nodes.first_or_none(
                                space_res=pat.space_res,
                                time_res=pat.time_res,
                                weather_effect=weather_effect,
                                obs_position=obs_position
                        )
                        if not obsPat_node:
                                obsPat_node = ObservationPattern(
                                        space_res=pat.space_res,
                                        time_res=pat.time_res,
                                        weather_effect=weather_effect,
                                        obs_position=obs_position
                                ).save()

                        obsPat_did = obsPat_node.disaster_ids if obsPat_node.disaster_ids else []
                        if dis_node.node_id not in obsPat_did:
                                obsPat_did.append(dis_node.node_id)
                                obsPat_node.disaster_ids = obsPat_did
                                obsPat_node.save()

                        if not obs_node.obsPat_argument.is_connected(obsPat_node):
                                obs_node.obsPat_argument.connect(obsPat_node)
                        saved += 1
                except Exception as e:
                        failed += 1
                        print(f"Database sync failed for observation pattern ID {res['obs_id']}: {e}")

        print(f'Observation patterns: {saved}/{len(tasks)} tasks saved; {failed} failed.')
        if failed:
                print('Observation pattern extraction is incomplete. Rerun obs_pat to retry missing patterns.')
                return False
        print(f"✅ Observation Pattern Extraction finished. Cost: {time.time() - start_time:.2f} seconds")
        return True


# ==========================================
# 9. GEO ENCODING & SPATIAL RELATION COMPUTATION
# ==========================================

def run_spatial_computation(max_workers, neo_connector):
    """
    Computes geographical encoding coordinates via APIs and evaluates topological,
    distance, and basin routing relationships for causally linked events[cite: 20].
    """
    print(f"\n{'=' * 50}")
    print(f"📍 [SPATIAL] Starting Geographic Encoding and Spatial Relation Recognition")
    print(f"{'=' * 50}")

    start_time = time.time()

    print("-> Step 1/3: Identifying Valid Geographic Locations...")
    geo_is_location = is_geo_location_parallel(max_workers)

    print("\n-> Step 2/3: Generating Geo-Encodings for Event Pairs...")
    # Performs encoding and updates the coordinates in DB
    event_pair_geo_encoding_parallel(geo_is_location, do_encoding=get_setting('geocoding.do_encoding'))

    print("\n-> Step 3/3: Calculating Semantic Spatial Relations (Distance, Topology, Basin)...")
    summary = main_calculate_spatial_relations(neo_connector)
    if isinstance(summary, dict) and summary.get('skipped_pairs', 0):
        print('Spatial computation is incomplete. Rerun spatial to retry skipped geometry lookups.')
        return False

    print(f"\n✅ Geographic Encoding and Spatial Relations completed in: {time.time() - start_time:.2f} seconds")
    return True


# ==========================================
# 10. COMMAND LINE INTERFACE ROUTER
# ==========================================

def main():
    """Run the selected construction stage using configuration defaults and CLI overrides."""
    parser = argparse.ArgumentParser(
        description="Standardized Command Line Toolkit for Knowledge Graph Extraction, Ingestion, and Fusion.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    tasks_help = (
        "Select the execution stage profile:\n"
        "  extract       : Parse unstructured texts into schema objects via LLM (Geo -> Event -> Relation)\n"
        "  fuse          : Fuse arguments and events, then deduplicate relations\n"
        "  env           : Generate structured hazard-formative environment rules\n"
        "  obs_pat       : Generate observation patterns for triggered observation tasks\n"
        "  spatial       : Perform geographic encoding and calculate spatial relationships\n"
        "  all           : Execute the entire pipeline sequentially (extract->fuse->env->obs_pat->spatial)"
    )

    parser.add_argument(
        '-t', '--task',
        type=str,
        required=True,
        choices=['extract', 'fuse', 'env', 'obs_pat', 'spatial', 'all'],
        help=tasks_help
    )

    parser.add_argument(
        '--config',
        default=str(get_config_path()),
        help="Construction configuration file (default: construct_config.json in the project root)"
    )

    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        default=get_configured_path('extraction.input_dir'),
        help="Input folder hosting chunked target JSON files (Applies only to 'extract' or 'all')"
    )

    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=get_setting('extraction.workers'),
        help="Thread pool size; overrides extraction.workers in the configuration"
    )

    parser.add_argument(
        '-db', '--database',
        type=str,
        default=get_setting('extraction.database'),
        help="The Neo4j database name to connect to."
    )

    args = parser.parse_args()
    args.database = resolve_database_name(args.database)
    args.input_dir = str(resolve_project_path(args.input_dir))
    overall_start = time.time()

    print(f"Connecting to Neo4j database: {args.database}...")
    try:
        ensure_database(args.database)
        neo_connector = Neo4jDatabaseConnector(args.database)
    except Exception as e:
        print(f"Failed to connect to database '{args.database}'. Error: {e}")
        sys.exit(1)

    if args.task == 'extract':
        run_extraction_pipeline(args.input_dir, args.workers)
    elif args.task == 'fuse':
        run_fusion()
    elif args.task == 'env':
        run_env_description(args.workers)
    elif args.task == 'obs_pat':
        if not run_obs_pattern(args.workers):
            sys.exit(1)
    elif args.task == 'spatial':
        if not run_spatial_computation(args.workers, neo_connector):
            sys.exit(1)
    elif args.task == 'all':
        print("\n🚀 [Pipeline Mode] Initiating End-to-End Graph Generation Pipeline...")

        if not run_extraction_pipeline(args.input_dir, args.workers):
            sys.exit(1)

        run_fusion()
        run_env_description(args.workers)
        if not run_obs_pattern(args.workers):
            sys.exit(1)
        if not run_spatial_computation(args.workers, neo_connector):
            sys.exit(1)

        print(
            f"\n🎉 [Pipeline Mode] Production run completed successfully! Total elapsed time: {time.time() - overall_start:.2f} seconds")


if __name__ == "__main__":
    main()
