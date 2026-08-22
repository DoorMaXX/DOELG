from ontology.neo4jmodel import Neo4jDatabaseConnector, Factor, StandardFactor
from geo_extraction import llm
from langchain.agents import create_agent
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
import pandas as pd
from langchain_core.documents import Document
import pickle
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import concurrent.futures
from threading import Lock

indicator_mapping = {
    '降水量': [],
    '空气温度': [],
    '空气湿度': [],
    '气压': [],
    '风速': [],
    '风向': [],
    '云顶高度': [],
    '云顶温度': [],
    '云覆盖率': [],
    '地表温度': [],
    '地表覆盖情况': [],
    '植被覆盖率': [],
    '土壤湿度': [],
    '水体面积': [],
    '水位': [],
    '水流速度': [],
    '水流量': [],
    '水质': ['河流', '湖泊', '水库', '池', '井', '沟渠', '湿地', '海域'],
    '地表形变速率、形变量': [],
    '设施形变速率、形变量': [
        '坝', '堤坊', '闸', '桥梁', '隧道', '索道',
        '地下人行通道', '下跨道', '火车站', '地铁站', '码头', '港口', '灯塔',
        '机场', '停机坪', '房屋', '地下建筑', '生产类构筑物', '公共服务类构筑物',
        '科学观测类构筑物', '垣栅类构筑物', '输电线', '电塔', '电杆', '变电站',
        '通信塔', '油气水输送管道', '储燃气罐', '水塔', '下水道', '窨井'
    ],
    '沉降深度': [],
    '裂缝宽度': [
        '坝', '堤坊', '闸', '桥梁', '隧道', '索道', '道路',
        '地下人行通道', '下跨道', '火车站', '地铁站', '码头', '港口', '灯塔',
        '机场', '停机坪', '房屋', '地下建筑', '生产类构筑物', '公共服务类构筑物',
        '科学观测类构筑物', '垣栅类构筑物', '输电线', '电塔', '电杆', '变电站',
        '通信塔', '油气水输送管道', '储燃气罐', '水塔', '下水道', '窨井', '山体'
    ],
    '倾斜倾角': [
        '电塔', '电杆', '通信塔', '灯塔', '水塔', '房屋', '高层建筑',
        '坝', '桥梁', '烟囱', '山体'
    ],
    '渗透（孔隙）水压力': [
        '坝', '堤坊', '山体', '隧道', '地下建筑', '矿区', '道路'
    ],
    '结构破坏特征': [
        '坝', '堤坊', '闸', '桥梁', '隧道', '索道', '道路',
        '地下人行通道', '下跨道', '火车站', '地铁站', '码头', '港口', '灯塔',
        '机场', '停机坪', '房屋', '地下建筑', '生产类构筑物', '公共服务类构筑物',
        '科学观测类构筑物', '垣栅类构筑物', '输电线', '电塔', '电杆', '变电站',
        '通信塔', '油气水输送管道', '储燃气罐', '水塔', '下水道', '窨井', '山体'
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


def map_standard_factor(factor, standard_factors):
    """
    Uses an LLM agent to map a natural language observation factor to one or more
    items in the standardized observation indicator library.
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
            standard_name = f_map.standard_name
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

        standard_name = factor_map.get(factor_name, '')
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


def main_factor_map_parallel(neo, logic_graph_database, max_workers=10):
    """
    Executes the factor mapping process concurrently using a thread pool
    for faster LLM processing, then saves results back to the database.
    """
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
                sf_node = StandardFactor.nodes.first_or_none(factor_name=f_map.standard_name)
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


def map_standard_factor_1(factor, standard_factors):
    """
    Uses an LLM agent to map a natural language observation factor to strictly
    one standardized observation indicator (best match).
    """
    sys_prompt = f"""
    你是一位资深的地球观测与空间环境监测专家。你的任务是分析用户的“观测因子”，并将其映射到“标准化观测指标库”中**唯一一个最精确匹配**的指标。

    ### 核心任务
    1. **语义理解**：分析用户输入的自然语言（可能是不规范的、口语化的，或者是上位的科学概念）。
    2. **本体推理**：在标准库中寻找对应的物理量、化学成分或环境参数。
    3. **一对一精确映射**：严格限制！无论用户的输入多么宽泛，你**必须且只能**选择一个最核心、相关度最高的标准指标。

    ### 标准化观测指标库
    {standard_factors}

    ### 评分标准 (Score)
    - **0.9 - 1.0**: 精确对应 (Exact Match) 或标准术语的直接别名。
    - **0.7 - 0.9**: 上位词到下位词的强关联 (User: "气象参数" -> 选取最具代表性的如"空气温度")。
    - **0.5 - 0.7**: 现象与物理量的关联 (User: "房屋倒塌" -> Std: "结构破坏特征")。
    - **< 0.5**: 弱相关，仅在没有更好选项时提供。

    ### 输出要求
    - 仅输出 JSON 格式。
    - `standard_name` 必须**严格**等于列表中的字符串，禁止修改一个字符。
    - **禁止输出多个指标的列表，只允许返回得分最高的那一个指标。**
    """

    agent = create_agent(llm, system_prompt=sys_prompt, response_format=ResultInfo)

    output = agent.invoke(
        {"messages": [{"role": "user", "content": f"用户观测需求: '{factor}'"}]}
    )

    result = output["structured_response"]

    return result


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


def main_factor_map_parallel_1(neo, logic_graph_database, max_workers=20):
    """
    Executes the strict one-to-one factor mapping process concurrently,
    filters results by score, and saves the best match to the database.
    """
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

            if best_map.score < 0.5:
                print(
                    f"Factor '{factor_node.factor_name}' best match '{best_map.standard_name}' scored {best_map.score}, discarded.")
                continue

            sf_node = StandardFactor.nodes.first_or_none(factor_name=best_map.standard_name)
            if sf_node:
                rel_m = factor_node.relate.relationship(sf_node)
                data = {
                    'score': best_map.score,
                    'type': getattr(best_map, 'relation_type', '映射'),
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
    neo = Neo4jDatabaseConnector('events-instantiation-beijing-2')
    update_standard_factors()