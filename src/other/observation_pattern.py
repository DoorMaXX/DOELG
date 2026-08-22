from ontology.neo4jmodel import Neo4jDatabaseConnector, ObservationEvent, ObservationPattern, DisasterEvent, Factor
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



embeddings = OllamaEmbeddings(model="bge-m3")
vector_db = None
retriever = None

def build_vector_store(csv_path: str):
    """
    Reads sensor data from a CSV file, constructs descriptive text documents
    for each sensor, and builds a local FAISS vector store for retrieval.
    """
    df = pd.read_excel(csv_path)

    documents = []
    for _, row in df.iterrows():
        content = f"传感器型号: {row['instrument_name']} ({row['full_name']})\n" \
                  f"搭载卫星: {row['satellite_name']}\n" \
                  f"传感器类型: {row['type_of_sensor']}\n" \
                  f"分辨率: {row['resolution']}\n" \
                  f"覆盖周期: {row['coverage_cycle']}\n" \
                  f"观测范围: {row['swatch_width']}\n" \
                  f"光谱范围: {row.get('spectral_range', '未知')}\n" \
                  f"主要任务: {row.get('primary_mission_objectives', '未知')}"

        doc = Document(page_content=content, metadata={"source": csv_path, "id": row['instrument_name']})
        documents.append(doc)

    vector_store = FAISS.from_documents(documents, embeddings)

    vector_store.save_local("sensor_faiss_index")
    return vector_store


def get_vector_store():
    global vector_db, retriever
    if vector_db is None:
        vector_db = build_vector_store("./data/sensor_dataset/sensor_id.xlsx")
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    return vector_db

@tool
def search_sensor_manual(query: str) -> str:
    """
    Queries the vector store for technical parameters, resolution limits,
    weather adaptabilities, and observation modes of sensors based on a manual search query.
    """
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


def get_event_obsPat_llm(text, obs_event_des, disaster_event_des):
    """
    Uses an LLM agent along with the sensor manual search tool to extract and deduce
    the spatial and temporal resolution, weather effects, and positioning of observation events.
    """
    sys_prompt = """
    你是一位地理空间传感网管理专家。给定一段【来源文本】和其中描述的一个由【灾害事件】所触发的【观测事件】，请从文本中分析并总结出导致该观测事件中传感器的“观测模式”。
    工作流：
    1. 分析原始文本中的该观测事件中传感器相关观测模式的描述。
    2. 使用 search_sensor_manual 工具观测事件中提及传感器的技术参数。
    3. 输出结构化的观测模式参数。
    4. 如果查询后找不到完全一致的型号，请根据文本进行解释，禁止主观臆断，禁止反复调用工具，允许输出为空。
    5. **一旦获得信息，必须立即生成结构化响应，停止继续思考。**
    """
    user_prompt = f"""
    【原始文本】: {text}
    【灾害事件】: {disaster_event_des}
    【观测事件】: {obs_event_des}

    请分析文本并查询手册，给出该观测事件下传感器的观测模式：
    """

    from pydantic import BaseModel, Field
    from typing import List, Literal

    class EnvDesInfo(BaseModel):
        space_res: float = Field(
            description="执行该传感器观测事件的空间分辨率数值，单位：米。若无法获取请填 -1 或依据逻辑推断。")
        time_res: float = Field(description="执行传感器观测事件的时间分辨率数值，单位：小时")
        weather_effect: Literal[
            '全天候观测', '受云雾遮挡', '受光照条件限制', '受极端气象限制', '受大气质量影响'] = Field(
            description="传感器观测效果受天气影响的情况")
        obs_position: Literal['临近点位观测', '均匀分布点位观测', '遥感观测'] = Field(
            description="传感器的观测覆盖方式")
        evidence: str = Field(description="得出此观测模式的判断依据，包含文本摘录和逻辑推理过程")

    agent = create_agent(llm, system_prompt=sys_prompt, tools=[search_sensor_manual], response_format=EnvDesInfo)
    output = agent.invoke(
        {"messages": [
            {"role": "user", "content": user_prompt}
        ]},
        config={"recursion_limit": 30}
    )
    return output["structured_response"]


def get_save_event_obsPat(obs_event_node, disaster_event_node):
    """
    Extracts the observation patterns from the textual origins of an observation event,
    and saves or updates this pattern within the Neo4j database.
    """
    obs_event_des = obs_event_node.event_description
    disaster_event_des = disaster_event_node.event_description
    text_nodes = obs_event_node.origin.all()
    text_contents = []
    for text_node in text_nodes:
        if text_node.exist.is_connected(disaster_event_node):
            text_contents.append(text_node.text_content)
    if len(text_contents) == 0:
        return

    origin_text = "\n".join(text_contents)

    obs_pat = get_event_obsPat_llm(origin_text, obs_event_des, disaster_event_des)
    space_res = obs_pat.space_res
    time_res = obs_pat.time_res
    weather_effect = obs_pat.weather_effect
    obs_position = obs_pat.obs_position
    obsPat_node = ObservationPattern.nodes.first_or_none(space_res=space_res, time_res=time_res,
                                                         weather_effect=weather_effect, obs_position=obs_position)
    if not obsPat_node:
        obsPat_node = ObservationPattern(space_res=space_res, time_res=time_res,
                                         weather_effect=weather_effect, obs_position=obs_position)
        obsPat_node.save()

    obsPat_did = obsPat_node.disaster_ids
    if disaster_event_node.node_id not in obsPat_did:
        obsPat_did.append(disaster_event_node.node_id)
        obsPat_node.disaster_ids = obsPat_did
        obsPat_node.save()

    if not obs_event_node.obsPat_argument.is_connected(obsPat_node):
        obs_event_node.obsPat_argument.connect(obsPat_node)
    print(f"{obs_event_des}\n{origin_text}\n------------------------------{obs_pat}-------------------------------")


def factor_identify(factor: str):
    """
    Evaluates a given factor name using an LLM to determine if it meets the criteria
    of a valid observable attribute within a Sensor Web framework.
    """
    sys_prompt = f"""
        你是一名顶尖的**传感器网络本体（SSN）**专家，你的任务判断输入的名称是否是传感器的可观测因子（observable factor）。
        ### 角色与目标
        你的核心目标是严格判断用户输入中的`factor`是否：
        **满足定义**：本身是一个有效的可观测因子（属性或特性）。
        ### 决策依据（重要）
        ### 关键规则 ：可观测因子的定义
        可观测因子定义为：**被传感器间接或直接观测、测量或感知到的世界的属性或特性。** 它必须具备以下特点：
        * **是属性，而非实体或事件。** (e.g., '淹没范围'、'温度'、'降水量' 是属性; '洪水'、'强降雨' 是事件; '河流' 是实体)
        ### 格式说明
        用户输入格式为： factor='名称'。
    """
    from pydantic import BaseModel, Field

    class InsertInfo(BaseModel):
        is_factor: bool = Field(
            description="严格根据规则判断，当且仅当输入的factor符合可观测因子的抽象属性定义时，才设置为`True`")
        evidence: str = Field(
            description="判断用户输入其是否作为可观测因子的判断依据"
        )

    agent = create_agent(llm, system_prompt=sys_prompt, response_format=InsertInfo)
    output = agent.invoke(
        {"messages": [{"role": "user", "content": f"factor=‘{factor}’"}]}
    )
    return output["structured_response"]


def do_factor_delete():
    """
    Iterates through all factors in the database, uses the identification process to
    determine their validity, saves evidence of failure, and deletes invalid factors.
    """
    factor_nodes = Factor.nodes.all()
    delete_factors = []
    factor_delete_evidence = []
    for factor_node in factor_nodes:
        factor_name = factor_node.factor_name
        idf_factor = factor_identify(factor_name)
        print(factor_name)
        print(idf_factor.is_factor, '\n', idf_factor.evidence)

        factor_delete_evidence.append([factor_name, idf_factor.is_factor, idf_factor.evidence])
        if not idf_factor.is_factor:
            delete_factors.append(factor_node)

    filename = f'factor_delete_evidence.pickle'
    try:
        with open(filename, 'wb') as file:
            pickle.dump(factor_delete_evidence, file)
        print(f"成功将变量保存到 {filename}")
    except Exception as e:
        print(f"保存过程中发生错误: {e}")

    for df in delete_factors:
        df.delete()


if __name__ == '__main__':
    neo = Neo4jDatabaseConnector('events-do-fusion-2')
    disaster_nodes = DisasterEvent.nodes.all()
    for de in disaster_nodes:
        observation_nodes = de.trigger.all()
        for observation_node in observation_nodes:
            get_save_event_obsPat(observation_node, de)