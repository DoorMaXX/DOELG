from ontology.neo4jmodel import Neo4jDatabaseConnector, Factor, StandardFactor
from geo_extraction import llm
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import List, Literal
import concurrent.futures
from src.construct_config import get_setting
from src.graph_language import english_term

indicator_mapping = {
    "Precipitation": [],
    "Air Temperature": [],
    "Air Humidity": [],
    "Atmospheric Pressure": [],
    "Wind Speed": [],
    "Wind Direction": [],
    "Cloud Top Height": [],
    "Cloud Top Temperature": [],
    "Cloud Fraction": [],
    "Surface Temperature": [],
    "Land Cover": [],
    "Vegetation Fraction": [],
    "Soil Moisture": [],
    "Water Body Area": [],
    "Water Level": [],
    "Water Flow Rate": [],
    "Water Flow Velocity": [],
    "Water Quality": [
        "River",
        "Lake",
        "Reservoir",
        "Pond",
        "Water Well",
        "Ditch",
        "Wetland",
        "Sea"
    ],
    "Surface Deformation": [],
    "Infrastructure Deformation": [
        "Dam",
        "Dyke",
        "Lock Gate",
        "Bridge",
        "Tunnel",
        "Aerialway",
        "Underpass",
        "Underpass",
        "Train Station",
        "Metro Station",
        "Dock",
        "Port",
        "Lighthouse",
        "Aerodrome",
        "Apron",
        "Building",
        "Underground Building",
        "Industrial Structure",
        "Public Service Structure",
        "Observation Structure",
        "Enclosure Structure",
        "Cable",
        "Tower",
        "Utility Pole",
        "Substation",
        "Communications Tower",
        "Pipeline",
        "Storage Tank",
        "Water Tower",
        "Sewer",
        "Sewer Vent"
    ],
    "Subsidence Depth": [],
    "Crack Width": [
        "Dam",
        "Dyke",
        "Lock Gate",
        "Bridge",
        "Tunnel",
        "Aerialway",
        "Highway",
        "Underpass",
        "Underpass",
        "Train Station",
        "Metro Station",
        "Dock",
        "Port",
        "Lighthouse",
        "Aerodrome",
        "Apron",
        "Building",
        "Underground Building",
        "Industrial Structure",
        "Public Service Structure",
        "Observation Structure",
        "Enclosure Structure",
        "Cable",
        "Tower",
        "Utility Pole",
        "Substation",
        "Communications Tower",
        "Pipeline",
        "Storage Tank",
        "Water Tower",
        "Sewer",
        "Sewer Vent",
        "Mountain"
    ],
    "Tilt Angle": [
        "Tower",
        "Utility Pole",
        "Communications Tower",
        "Lighthouse",
        "Water Tower",
        "Building",
        "High-rise Building",
        "Dam",
        "Bridge",
        "Chimney",
        "Mountain"
    ],
    "Seepage Pressure": [
        "Dam",
        "Dyke",
        "Mountain",
        "Tunnel",
        "Underground Building",
        "Mining Area",
        "Highway"
    ],
    "Structural Characteristics": [
        "Dam",
        "Dyke",
        "Lock Gate",
        "Bridge",
        "Tunnel",
        "Aerialway",
        "Highway",
        "Underpass",
        "Underpass",
        "Train Station",
        "Metro Station",
        "Dock",
        "Port",
        "Lighthouse",
        "Aerodrome",
        "Apron",
        "Building",
        "Underground Building",
        "Industrial Structure",
        "Public Service Structure",
        "Observation Structure",
        "Enclosure Structure",
        "Cable",
        "Tower",
        "Utility Pole",
        "Substation",
        "Communications Tower",
        "Pipeline",
        "Storage Tank",
        "Water Tower",
        "Sewer",
        "Sewer Vent",
        "Mountain"
    ]
}


class MappingItem(BaseModel):
    standard_name: str = Field(..., description="匹配到的标准化观测指标名称（必须严格来自提供的列表）")
    score: float = Field(..., description="关联程度置信度 (0.0 - 1.0)")
    relation_type: Literal["exact_match", "broad_to_narrow", "narrow_to_broad", "related"] = Field(
        ...,
        description="关系类型：exact_match(同义/近义), broad_to_narrow(输入是上位词，标准指标是下位词), narrow_to_broad(输入是下位词，标准指标是上位词), related(相关联)"
    )
    reasoning: str = Field(..., description="简述匹配理由，解释为何存在上下位或语义关联")


class ResultInfo(BaseModel):
    mappings: List[MappingItem] = Field(description="所有匹配到的标准指标列表，按关联程度降序排列")


class MappingItem_en(BaseModel):
    standard_name: str = Field(..., description="Matched standard indicator name, exactly as listed")
    score: float = Field(..., description="Association confidence (0.0 - 1.0)")
    relation_type: Literal["exact_match", "broad_to_narrow", "narrow_to_broad", "related"] = Field(
        ...,
        description="Relation: exact_match, broad_to_narrow (input is broader), narrow_to_broad (input is narrower), or related"
    )
    reasoning: str = Field(..., description="Briefly explain the match and broader, narrower, or semantic association")

class ResultInfo_en(BaseModel):
    mappings: List[MappingItem_en] = Field(description="Matched standard indicators in descending order of relevance")


def map_standard_factor(factor, standard_factors):
    """
    Uses an LLM agent to map a natural language observation factor to one or more
    items in the standardized observation indicator library.


    This is the primary execution function. Since the corpus is predominantly
    Chinese, the prompt is in Chinese. See map_standard_factor_en for the English version.
    """
    sys_prompt = f"""
    你是一位资深的地球观测与空间环境监测专家。你的任务是分析用户的“观测因子”，并将其映射到“标准化观测指标库”中具体的一个或多个指标。

    ### 核心任务
    1. **语义理解**：分析用户输入的自然语言（可能是不规范的、口语化的，或者是上位的科学概念）。
    2. **本体推理**：在标准库中寻找对应的物理量、化学成分或环境参数。
    3. **多维映射**：注意，用户的一个需求可能对应多个标准指标（一对多）。

    ### 标准化观测指标库
    {standard_factors}

    ### 评分标准 (Score)
    - **0.9 - 1.0**: 精确对应 (Exact Match) 或标准术语的直接别名。
    - **0.7 - 0.9**: 上位词到下位词的强关联 (User: "气象参数" -> Std: "空气温度","空气湿度"...)。
    - **0.5 - 0.7**: 现象与物理量的关联 (User: "房屋倒塌" -> Std: "结构破坏特征")。
    - **< 0.5**: 弱相关，仅在没有更好选项时提供。

    ### 输出要求
    - 仅输出 JSON 格式。
    - `standard_name` 必须**严格**等于列表中的字符串，禁止修改一个字符。
    - 如果需要映射多个指标，请在列表中完整列出。
    """

    agent = create_agent(llm, system_prompt=sys_prompt, response_format=ResultInfo)

    output = agent.invoke(
        {"messages": [{"role": "user", "content": f"用户观测需求: '{factor}'"}]}
    )

    result: ResultInfo = output["structured_response"]
    return result


def map_standard_factor_en(factor, standard_factors):
    """
    Uses an LLM agent to map a natural language observation factor to one or more
    items in the standardized observation indicator library.
    """
    sys_prompt = f"""
    You are an Earth observation and environmental monitoring expert. Map the user observation factor to one or more indicators in the standardized library.

    ### Core task
    1. **Semantic understanding**: Analyze natural-language input, including informal wording or broader scientific concepts.
    2. **Ontology reasoning**: Find matching physical quantities, chemical components, or environmental parameters.
    3. **Multiple mappings**: One user requirement may correspond to several standard indicators.

    ### Standardized observation indicator library
    {standard_factors}

    ### Scoring criteria
    - **0.9 - 1.0**: Exact match or direct synonym of the standard term.
    - **0.7 - 0.9**: Strong broader-to-narrower association (User: "meteorological parameters" -> Std: "air temperature","air humidity"...)。
    - **0.5 - 0.7**: Association between a phenomenon and a physical quantity (User: "building collapse" -> Std: "structural damage characteristics")。
    - **< 0.5**: Weak association; provide only if no better option exists.

    ### Output requirements
    - Output JSON only.
    - `standard_name` must exactly match a string in the list without changing any character.
    - List every mapped indicator when multiple mappings are required.
    """

    agent = create_agent(llm, system_prompt=sys_prompt, response_format=ResultInfo_en)

    output = agent.invoke(
        {"messages": [{"role": "user", "content": f"User observation requirement: '{factor}'"}]}
    )

    result: ResultInfo_en = output["structured_response"]
    return result



def main_factor_map(neo, logic_graph_database):
    """
    Main function to map all factors in the database to standard factors and
    create relationships between them in the Neo4j graph.
    """
    standard_factor_names = indicator_mapping.keys()

    neo.connect_database(logic_graph_database)
    for standard_factor_name in standard_factor_names:
        sf_node = StandardFactor.nodes.first_or_none(factor_name=standard_factor_name)
        if not sf_node:
            sf_node = StandardFactor(factor_name=standard_factor_name,
                                     relate_geos=indicator_mapping.get(standard_factor_name, [])).save()

    factor_nodes = Factor.nodes.all()
    for factor_node in factor_nodes:
        factor_name = factor_node.factor_name
        map_result = map_standard_factor(factor_name, standard_factor_names)
        print(f"Factor:{factor_name}\n, result:{map_result}\n")
        map_list = map_result.mappings
        for f_map in map_list:
            standard_name = english_term(f_map.standard_name, strict=True)
            score = f_map.score
            relation_type = f_map.relation_type
            evidence = f_map.reasoning

            sf_node = StandardFactor.nodes.first_or_none(factor_name=standard_name)
            if sf_node:
                rel_m = factor_node.relate.relationship(sf_node)
                if rel_m:
                    rel_m.score = score
                    rel_m.type = relation_type
                    rel_m.evidence = evidence
                    rel_m.save()
                else:
                    factor_node.relate.connect(sf_node, {'score': score, 'type': relation_type, 'evidence': evidence})


def main_factor_map_(neo, logic_graph_database, factor_map):
    """
    Maps factors to standard factors using a predefined dictionary mapping
    and updates the database.
    """
    standard_factor_names = indicator_mapping.keys()

    neo.connect_database(logic_graph_database)
    for standard_factor_name in standard_factor_names:
        sf_node = StandardFactor.nodes.first_or_none(factor_name=standard_factor_name)
        if not sf_node:
            sf_node = StandardFactor(factor_name=standard_factor_name,
                                     relate_geos=indicator_mapping.get(standard_factor_name, [])).save()

    factor_nodes = Factor.nodes.all()
    for factor_node in factor_nodes:
        factor_name = factor_node.factor_name

        standard_name = english_term(factor_map.get(factor_name, ''), strict=True)
        if standard_name == '':
            continue

        sf_node = StandardFactor.nodes.first_or_none(factor_name=standard_name)
        if sf_node:
            rel_m = factor_node.relate.relationship(sf_node)
            if rel_m:
                continue
            else:
                factor_node.relate.connect(sf_node)


def map_single_factor_task(factor_node, standard_factor_names):
    """
    Execution task for evaluating a single factor against the standard factors.
    Returns the node, result, and potential errors without database interaction.
    """
    factor_name = factor_node.factor_name
    try:
        result = map_standard_factor(factor_name, standard_factor_names)
        return factor_node, result, None
    except Exception as e:
        return factor_node, None, str(e)


def main_factor_map_parallel(neo, logic_graph_database, max_workers=None):
    """
    Executes the factor mapping process concurrently using a thread pool
    for faster LLM processing, then saves results back to the database.
    """
    if max_workers is None:
        max_workers = get_setting('factor_mapping.workers')
    standard_factor_names = list(indicator_mapping.keys())
    neo.connect_database(logic_graph_database)

    for name in standard_factor_names:
        if not StandardFactor.nodes.first_or_none(factor_name=name):
            StandardFactor(factor_name=name, relate_geos=indicator_mapping.get(name, [])).save()

    factor_nodes = Factor.nodes.all()
    new_factor_nodes = []
    for factor_node in factor_nodes:
        standard_factor = factor_node.relate.all()
        if len(standard_factor) == 0:
            new_factor_nodes.append(factor_node)
    factor_nodes = new_factor_nodes

    print(f"Start concurrent inference (threads: {max_workers})...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_node = {
            executor.submit(map_single_factor_task, node, standard_factor_names): node
            for node in factor_nodes
        }

        for future in concurrent.futures.as_completed(future_to_node):
            factor_node, map_result, error = future.result()

            if error:
                print(f"Factor '{factor_node.factor_name}' failed: {error}")
                continue

            print(f"Mapping successful: {factor_node.factor_name}, saving to db...")

            for f_map in map_result.mappings:
                sf_node = StandardFactor.nodes.first_or_none(factor_name=english_term(f_map.standard_name, strict=True))
                if sf_node:
                    rel_m = factor_node.relate.relationship(sf_node)
                    data = {
                        'score': f_map.score,
                        'type': f_map.relation_type,
                        'evidence': f_map.reasoning
                    }
                    if rel_m:
                        for key, value in data.items():
                            setattr(rel_m, key, value)
                        rel_m.save()
                    else:
                        factor_node.relate.connect(sf_node, data)

    print("Mapping and database synchronization complete.")


def map_single_factor_task_1(factor_node, standard_factor_names):
    """
    Execution task for evaluating a single factor using strict one-to-one mapping.
    """
    factor_name = factor_node.factor_name
    try:
        result = map_standard_factor(factor_name, standard_factor_names)
        return factor_node, result, None
    except Exception as e:
        return factor_node, None, str(e)


def main_factor_map_parallel_1(neo, logic_graph_database, max_workers=None):
    """
    Executes the strict one-to-one factor mapping process concurrently,
    filters results by score, and saves the best match to the database.
    """
    if max_workers is None:
        max_workers = get_setting('factor_mapping.strict_workers')
    standard_factor_names = list(indicator_mapping.keys())
    neo.connect_database(logic_graph_database)

    for name in standard_factor_names:
        if not StandardFactor.nodes.first_or_none(factor_name=name):
            StandardFactor(factor_name=name, relate_geos=indicator_mapping.get(name, [])).save()

    factor_nodes = Factor.nodes.all()
    new_factor_nodes = []
    for factor_node in factor_nodes:
        standard_factor = factor_node.relate.all()
        if len(standard_factor) == 0:
            new_factor_nodes.append(factor_node)
    factor_nodes = new_factor_nodes

    print(f"Start concurrent inference (threads: {max_workers})...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_node = {
            executor.submit(map_single_factor_task, node, standard_factor_names): node
            for node in factor_nodes
        }

        for future in concurrent.futures.as_completed(future_to_node):
            factor_node, map_result, error = future.result()

            if error:
                print(f"Factor '{factor_node.factor_name}' failed: {error}")
                continue

            if not map_result.mappings:
                print(f"Factor '{factor_node.factor_name}' found no matching indicator.")
                continue

            print(f"Mapping successful: {factor_node.factor_name}, saving to db...")

            best_map = sorted(map_result.mappings, key=lambda x: x.score, reverse=True)[0]

            if best_map.score < get_setting('factor_mapping.minimum_score'):
                print(
                    f"Factor '{factor_node.factor_name}' best match '{best_map.standard_name}' scored {best_map.score}, discarded.")
                continue

            sf_node = StandardFactor.nodes.first_or_none(factor_name=english_term(best_map.standard_name, strict=True))
            if sf_node:
                rel_m = factor_node.relate.relationship(sf_node)
                data = {
                    'score': best_map.score,
                    'type': getattr(best_map, 'relation_type', 'Mapping'),
                    'evidence': getattr(best_map, 'reasoning', '')
                }

                if rel_m:
                    for key, value in data.items():
                        setattr(rel_m, key, value)
                    rel_m.save()
                else:
                    factor_node.relate.connect(sf_node, data)

    print("Mapping and database synchronization complete.")


def update_standard_factors():
    """
    Updates all standard factors in the database with their associated
    geographic object types from the local mapping dictionary.
    """
    sf_nodes = StandardFactor.nodes.all()
    for sf_node in sf_nodes:
        sf_name = sf_node.factor_name
        sf_node.relate_geos = indicator_mapping.get(sf_name, [])
        sf_node.save()


if __name__ == '__main__':
    neo = Neo4jDatabaseConnector(get_setting('standalone.factor_database'))
    update_standard_factors()
