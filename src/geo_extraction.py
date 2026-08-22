from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from ontology.neo4jmodel import Neo4jDatabaseConnector, GeoObject, DisPhen, DisasterEvent, OriginText, Geometry
import json
import ast
import re
from other.geo_encoding import get_location
from tqdm import tqdm
from other.geo_encoding import get_address_lookup_local, get_entity_combined_geometry, analyze_spatial_relation, \
    get_topological_relation
import pandas as pd


"""
1. Extract 'geographic object' entities and spatial relations between geographic objects from text.
2. Identify spatial relations between disaster events (geocoding & spatial computation + path extraction).
"""

llm = ChatOpenAI(temperature=1.0,
                 model="deepseek-chat",
                 openai_api_base="https://api.deepseek.com",
                 openai_api_key="")


def geo_ner_re(text: str):
    """
    Extracts static geographic entities (Features) and their static spatial relationships
    from the provided text using a Large Language Model and a predefined ontology.
    """
    ontology_prompt = """
        class Entity:
            # 所有实体的基类
            def __init__(self, name:str):
                self.name = name

        class Spatial_object(Entity):
            # 地理空间对象的通用抽象类
            pass

        class Geometry(Spatial_object):
            # 地理空间对象的空间范围形状类型，限定为'point'、'linestring'、'polygon'
            def __init__(self, name:str):
                self.name = name
                if name not in ['point', 'linestring', 'polygon']:
                    self.name = ''

        class Feature(Spatial_object):
            # 定义：文本中提及的现实【静止】地理空间要素实体。
            # 范围：指向现实某一具体空间范围或为一类地物的统称，包括自然地貌（山、河）和人工设施（路、楼）。
            # 【重要负面约束】：
            # 1. 严禁抽取任何可移动物体，如：列车、车辆、汽车、船舶、飞机、行人、动物等。
            # 2. 严禁抽取自然现象，如：洪水、积水、由于降雨形成的临时水坑等。
            def __init__(self, 
                name:str,   #地理空间要素实体名称，如武汉市、洞庭湖、南京路
                geo_type:str,   #地理空间要素实体类型，如城市、湖泊、道路
                geometry:Geometry   #地理空间要素空间范围形状类型
            ):
                self.name = name
                self.geo_type = geo_type
                self.geometry = geometry
                # 允许的类型列表（仅限静止地物）：
                valid_types = [
                '河流', '湖泊', '流域', '河、湖岛',
                '河滩', '泉', '瀑布', '山体', '山洞', '漏斗', '沟壑', '山谷', '海域', '海岸', '海岛', '田地', '林地', '草地',
                '沙地', '湿地', '沟渠', '水库', '池', '井', '闸', '坝', '堤坊',
                '泵站', '行、蓄、滞洪区', '道路', '铁路', '地铁', '水运航道', '桥梁', '隧道', '索道', '地下人行通道', '下跨道',
                '交通连通交叉口', '火车站', '地铁站', '电车站', '码头', '港口', '灯塔', '机场', '停机坪',
                '地下停车场', '地面停车场', '多楼层停车场', '汽车站', '加油站', '房屋', '地下建筑', '生产类构筑物',
                '公共服务类构筑物', '科学观测类构筑物', '垣栅类构筑物', '输电线', '电塔', '电杆', '变电站',
                '通信线', '基站', '通信塔', '油气水输送管道', '储燃气罐', '水塔', '下水道', '窨井', '居民区',
                '公共管理与服务区', '商业区', '矿区', '仓库区', '绿地', '农业区', '公用设施区', '行政区'
                ]
                if geo_type not in valid_types:
                    self.geo_type = ''
        
        class Relation:
            # 所有关系的基类.
            def __init__(self, head_entity: Entity, tail_entity: Entity):
                self.head_entity = head_entity
                self.tail_entity = tail_entity
        
        class SpatialRelation(Relation):
            # 定义：两个Feature之间的【静态】空间位置关系。
            # 包含类别：
            # 1. 拓扑关系：包含、被包含、相交、毗邻、连接等。
            # 2. 距离关系：远离、靠近、附近等。
            # 3. 方位关系：上游、下游、以东、以北、高海拔、低海拔、中心、四周等。
            def __init__(self, head_feature: Feature, tail_feature: Feature, spatial_relation_name: str):
                super.__init__(head_feature, tail_feature)
                self.spatial_relation_name = spatial_relation_name
                
    """

    system_prompt = ("这是一个面向对象的编程任务，上面定义了一些实体（Entity）类及关系（Relation）类，请将下列文本转换为若干个Feature和SpatialRelation"
                     "对象实例，要求："
                     "1、仅提取文本中明确提及的信息，不添加推测内容, 并只能输出results变量,输出格式严格按照案例。"
                     "2. Feature提取：仅提取静态地理要素，**自动忽略**文本中的列车、车辆等移动物体。"
                     "3. SpatialRelation提取：仅提取表示两地物间存在的静态空间关系，**自动过滤**“行驶至”、“涌入”等动态动作。")

    chat_example_1 = [
        "郑东新区白沙镇一带临近贾鲁河，属于常庄水库泄洪的下游地区，受灾严重。",
        """
        result=[Feature('郑东新区白沙镇', '行政区', Geometry('polygon')),
                Feature('贾鲁河', '河流', Geometry('linestring')),
                Feature('常庄水库', '水库', Geometry('polygon')),
                SpatialRelation(Feature('郑东新区白沙镇', '行政区', Geometry('polygon')), Feature('贾鲁河', '河流', Geometry('linestring')), '临近')
                SpatialRelation(Feature('郑东新区白沙镇', '行政区', Geometry('polygon')), Feature('常庄水库', '水库', Geometry('polygon')), '属于下游')
                ]
        """]

    prompt_template = ChatPromptTemplate(
        [
            ("system", ontology_prompt),
            ("system", system_prompt),
            ("human", chat_example_1[0]),
            ("ai", chat_example_1[1]),
            ("human", "{user_input}")
        ]
    )

    chain = prompt_template | llm

    output = chain.invoke(
        {
            'user_input': text
        }
    )
    return output.content


def parse_geo_string_to_dict(data_str):
    """
    Parses the string output generated by the LLM (which resembles Python object instantiation)
    using regular expressions, returning a structured dictionary of features and spatial relations.
    """
    FEATURE_PATTERN = re.compile(
        r"Feature\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*Geometry\s*\(\s*'([^']+)'\s*\)\s*\)",
        flags=re.DOTALL
    )

    # SpatialRelation 模式
    # 难点在于嵌套括号。这里利用 Feature 结尾必然是 Geometry(...) ) 的结构进行锚定。
    # Group 1: Subject Feature 完整字符串
    # Group 2: Object Feature 完整字符串
    # Group 3: 关系名称
    SPATIAL_RELATION_PATTERN = re.compile(
        r"SpatialRelation\s*\(\s*"  # 匹配头: SpatialRelation(
        r"(Feature\s*\(.+?Geometry\s*\(.+?\)\s*\))"  # 匹配主体 Feature (非贪婪匹配直到 Geometry 结束)
        r"\s*,\s*"  # 匹配中间逗号
        r"(Feature\s*\(.+?Geometry\s*\(.+?\)\s*\))"  # 匹配客体 Feature
        r"\s*,\s*"  # 匹配中间逗号
        r"'([^']+)'"  # 匹配关系字符串 '位于'
        r"\s*\)",  # 匹配尾部 )
        flags=re.DOTALL
    )

    features = []
    spatial_relations = []

    feature_matches = FEATURE_PATTERN.findall(data_str)

    for name, f_type, geometry in feature_matches:
        features.append({
            "name": name.strip(),
            "type": f_type.strip(),
            "geometry": geometry.strip()
        })

    relation_matches = SPATIAL_RELATION_PATTERN.findall(data_str)

    for sub_str, obj_str, relation_type in relation_matches:
        sub_match = FEATURE_PATTERN.search(sub_str)
        subject = sub_match.group(1).strip() if sub_match else "Unknown Subject"

        obj_match = FEATURE_PATTERN.search(obj_str)
        object_name = obj_match.group(1).strip() if obj_match else "Unknown Object"

        spatial_relations.append({
            "subject": subject,
            "object": object_name,
            "relation_type": relation_type.strip()
        })

    parsed_data = {
        "features": features,
        "spatial_relations": spatial_relations
    }

    return parsed_data


def insert_database_geo(llm_result, origin_text, title):
    """
    Inserts the parsed geographic features and spatial relationships into the Neo4j database,
    linking the newly created or updated GeoObject nodes to their source OriginText node.
    """
    origin_text_node = OriginText.nodes.first_or_none(text_content=origin_text)
    if not origin_text_node:
        origin_text_node = OriginText(text_content=origin_text, text_title=title).save()

    for geo in llm_result['features']:
        geo_node = GeoObject.nodes.first_or_none(geo_object_name=geo['name'])
        if not geo_node:
            geo_node = GeoObject(
                geo_object_name=geo["name"],
                geo_type=geo["type"],
                geometry=geo["geometry"]
            ).save()
        if not geo_node.origin.is_connected(origin_text_node):
            geo_node.origin.connect(origin_text_node)
            origin_text_node.hasGeo.connect(geo_node)

    for geo_re in llm_result['spatial_relations']:
        geo_head_node = GeoObject.nodes.first_or_none(geo_object_name=geo_re['subject'])
        geo_tail_node = GeoObject.nodes.first_or_none(geo_object_name=geo_re['object'])
        if geo_head_node and geo_tail_node:
            if geo_head_node.spatial.is_connected(geo_tail_node):
                rel = geo_head_node.spatial.relationship(geo_tail_node)
                if geo_re['relation_type'] not in rel.relation_name.split(";"):
                    rel.relation_name = f"{rel.relation_name};{geo_re['relation_type']}"
                    rel.save()
            else:
                rel = geo_head_node.spatial.connect(geo_tail_node)
                rel.relation_name = geo_re['relation_type']
                rel.save()


def identify_is_location(geo_name: str):
    """
    Uses the LLM to determine whether a given geographic name represents a specific,
    real-world physical location (returning True or False).
    """
    prompt_template = ChatPromptTemplate(
        [
            ("system",
             "你是一名地理学专家，你需要判断用户输入的地理对象名称是否具体指向一个现实地理位置或范围，若是返回True，否则返回False"),
            ("human", "洞庭湖"),
            ("ai", "True"),
            ("human", "地铁"),
            ("ai", "False"),
            ("human", "村庄"),
            ("ai", "False"),
            ("human", "句容水库"),
            ("ai", "True"),
            ("human", "{user_input}"),
        ]
    )
    chain = prompt_template | llm

    output = chain.invoke(
        {
            'user_input': geo_name
        }
    )
    oc = output.content
    print(f'{geo_name}-result:', oc)
    if oc in ['True', 'true']:
        return True
    else:
        return False


def get_geo_location(text, geo_name, geo_type):
    """
    Invokes an LLM agent equipped with a geocoding tool to query OpenStreetMap.
    It resolves ambiguities using the provided text context and returns the location validity,
    coordinates, and OSM IDs.
    """
    sys_prompt = f"""
        你是一名资深的地理信息系统（GIS）与地理编码（Geocoding）专家，熟悉 OpenStreetMap（OSM）与现实世界地理对象的语义对应关系。
        ## 核心任务
        你的任务是：**对给定文本中的地理对象名称进行严格的地理定位判定与查询结果筛选**，判断该名称在现实世界中是否指向一个**具体、可唯一或明确限定范围的地理实体**。
        若名称在查询结果中存在**多个同名地理对象**，必须结合“参考文本”中的上下文进行消歧
        ### 🛑 工作流程与关键约束
        1. **使用工具**：你必须且只能调用**一次** `get_location` 工具，结合上下文信息使用目标地理对象 '{geo_name}' 的无歧义标准化名称（带有明确的地址参考信息如县、市、省）作为查询参数。
        2. **结果分析与消歧**：根据 `get_location` 的输出和“参考文本”的语境进行判定。
        3. **最终输出**：一旦你完成判定和结果筛选，**你必须立即停止任何工具调用**，并直接以严格的 Pydantic 格式返回最终的 `GeoInfo` 结构。
        4. **不允许重复查询或工具调用**：无论查询结果如何，**在一次完整的调用中，你不能再次调用 `get_location` 工具。**
    """

    from pydantic import BaseModel, Field
    from typing import List

    class GeoInfo(BaseModel):
        is_location: bool = Field(description=f"查询到地理编码结果是否与输入的地理对象'{geo_name}'匹配")
        coordinates: List[List[float]] = (
            Field(
                description="若'is_location'为'True', 在地理编码结果中选择该地理对象名称指向的地理坐标，尽可能选择最准确的一个，"
                            "若有多个坐标代表该地理对象的不同组成部分也可返回多个，返回 [[lon1, lat1],... ]。"))
        osm_type: List[str] = Field(description="若'is_location'为'True', 地理编码结果中所选地理对象的osm_type")
        osm_id: List[int] = Field(description="若'is_location'为'True', 地理编码结果中所选地理对象的osm_id")

    agent = create_agent(llm, tools=[get_location], system_prompt=sys_prompt, response_format=GeoInfo)
    output = agent.invoke(
        {"messages": [
            {"role": "user", "content": f"参考文本：‘{text}’"},
            {"role": "user", "content": f"解析参考文本中'{geo_type}'类型地理对象‘{geo_name}’"}
        ]}
    )
    return output["structured_response"]


def delete_all_geometry_node():
    """
    Deletes all Geometry nodes currently stored in the Neo4j database.
    """
    geometry_nodes = Geometry.nodes.all()
    for geometry_node in geometry_nodes:
        geometry_node.delete()


def is_geo_location():
    """
    Iterates through all GeoObject nodes in the database, sequentially checking
    and updating whether each object represents a valid physical location.
    Returns a list of valid location names.
    """
    geo_nodes = GeoObject.nodes.all()
    geos_location = []
    for geo_node in tqdm(geo_nodes, desc="地理实体定位性判定"):
        geo_name = geo_node.geo_object_name
        if geo_node.is_location is None:
            geo_node.is_location = identify_is_location(geo_name)
            geo_node.save()
        if geo_node.is_location:
            geos_location.append(geo_name)
    return geos_location


from tqdm import tqdm


def is_geo_location_parallel(max_workers=15):
    """
    A multi-threaded version of is_geo_location. It batches API requests to the LLM
    for location validation and subsequently updates the Neo4j database in the main thread.
    """
    geo_nodes = GeoObject.nodes.all()
    geos_location = []

    tasks = []
    name_to_node_map = {}

    for geo_node in geo_nodes:
        geo_name = geo_node.geo_object_name
        if geo_node.is_location is None:
            tasks.append(geo_name)
            name_to_node_map[geo_name] = geo_node
        elif geo_node.is_location:
            geos_location.append(geo_name)

    if not tasks:
        return geos_location

    def fetch_location_data(name):
        try:
            is_loc = identify_is_location(name)

            results = []
            if is_loc is not None:
                results.append({
                    'geo_name': name,
                    'is_location': is_loc
                })
                print(results)
            return results
        except Exception as e:
            print(f"处理实体 {name} 时出错: {e}")
            return []

    collected_data = []
    print(f"🚀 开始并行调用 API (并发数: {max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(fetch_location_data, name): name for name in tasks}

        for future in tqdm(as_completed(future_to_item), total=len(tasks), desc="大模型判定"):
            res = future.result()
            if res:
                collected_data.extend(res)

    if collected_data:
        print(f"存储中... 共有 {len(collected_data)} 条记录待入库")
        for item in collected_data:
            name = item['geo_name']
            is_loc = item['is_location']

            node = name_to_node_map.get(name)
            if node:
                node.is_location = is_loc
                node.save()

            if is_loc:
                geos_location.append(name)

        print("✅ 地理实体处理并入库完成。")
    else:
        print("⚠️ 未提取到有效数据。")

    return geos_location


def get_save_geo_geometry(geo_node, do_encoding=False):
    """
    Retrieves the OpenStreetMap ID (Geometry) associated with a GeoObject node.
    If none exists and do_encoding is True, it triggers the geocoding process,
    saves the new Geometry node, and connects it.
    """
    if geo_node.has_geometry.all():
        print(f"{geo_node.geo_object_name}已定位")
        geometry = geo_node.has_geometry.all()[0]
        return geometry.osm_id
    if not do_encoding:
        return ''

    text_nodes = geo_node.origin.all()
    text_contents = [text_node.text_content for text_node in text_nodes]
    origin_text = "".join(text_contents)
    geo_name = geo_node.geo_object_name
    geo_type = geo_node.geo_type

    if geo_node.is_location is None:
        geo_node.is_location = identify_is_location(geo_name)
        geo_node.save()
    if not geo_node.is_location:
        print(f"{geo_node.geo_object_name}不可定位")
        return ''

    location_info = get_geo_location(origin_text, geo_name, geo_type)
    print(f"{geo_name}:\n{location_info}")
    if location_info.is_location and location_info.coordinates and location_info.osm_type and location_info.osm_id:
        coordinates = []
        osm_ids = ""
        for co in location_info.coordinates:
            if len(co) == 2:
                coordinates.extend(co)
        if len(location_info.osm_type) == len(location_info.osm_id):
            for i in range(len(location_info.osm_type)):
                osm_id = ""
                if location_info.osm_type[i] == 'relation':
                    osm_id = f"R{location_info.osm_id[i]}"
                elif location_info.osm_type[i] == 'way':
                    osm_id = f"W{location_info.osm_id[i]}"
                elif location_info.osm_type[i] == 'node':
                    osm_id = f"N{location_info.osm_id[i]}"
                if osm_ids != "":
                    osm_ids += f",{osm_id}"
                else:
                    osm_ids += osm_id
        geometry_node = Geometry.nodes.first_or_none(osm_id=osm_ids)
        if not geometry_node:
            geometry_node = Geometry(location=coordinates, osm_id=osm_ids)
            geometry_node.save()
            print(f"creat geometry node : {osm_ids}")
        if not geo_node.has_geometry.is_connected(geometry_node):
            geo_node.has_geometry.connect(geometry_node)
            print(f"connect: {geo_name} to {osm_ids}")
        geo_node.is_location = True
        geo_node.save()
        return osm_ids
    return ''


def geo_location():
    """
    Iterates over all valid geographic location nodes in the database and performs
    geocoding (fetching coordinates and OSM IDs) for nodes that lack spatial geometry.
    """
    geo_nodes = GeoObject.nodes.all()
    for geo_node in tqdm(geo_nodes, desc="地理编码"):
        if geo_node.has_geometry.all():
            print(f"{geo_node.geo_object_name}已定位")
            continue
        text_nodes = geo_node.origin.all()
        text_contents = [text_node.text_content for text_node in text_nodes]
        origin_text = "".join(text_contents)
        geo_name = geo_node.geo_object_name
        geo_type = geo_node.geo_type

        if geo_node.is_location is None:
            geo_node.is_location = identify_is_location(geo_name)
            geo_node.save()
        if not geo_node.is_location:
            continue

        location_info = get_geo_location(origin_text, geo_name, geo_type)
        print(f"{geo_name}:\n{location_info}")
        if location_info.is_location and location_info.coordinates and location_info.osm_type and location_info.osm_id:
            coordinates = []
            osm_ids = ""
            for co in location_info.coordinates:
                if len(co) == 2:
                    coordinates.extend(co)
            if len(location_info.osm_type) == len(location_info.osm_id):
                for i in range(len(location_info.osm_type)):
                    osm_id = ""
                    if location_info.osm_type[i] == 'relation':
                        osm_id = f"R{location_info.osm_id[i]}"
                    elif location_info.osm_type[i] == 'way':
                        osm_id = f"W{location_info.osm_id[i]}"
                    elif location_info.osm_type[i] == 'node':
                        osm_id = f"N{location_info.osm_id[i]}"
                    if osm_ids != "":
                        osm_ids += f"-{osm_id}"
                    else:
                        osm_ids += osm_id
            geometry_node = Geometry.nodes.first_or_none(osm_id=osm_ids)
            if not geometry_node:
                geometry_node = Geometry(location=coordinates, osm_id=osm_ids)
                geometry_node.save()
                print(f"creat geometry node : {osm_ids}")
            if not geo_node.has_geometry.is_connected(geometry_node):
                geo_node.has_geometry.connect(geometry_node)
                print(f"connect: {geo_name} to {osm_ids}")
            geo_node.is_location = True
            geo_node.save()


def event_pair_geo_encoding(geo_has_location: list, do_encoding: True):
    """
    Iterates over causally linked DisasterEvent pairs. If their associated geographical
    objects are valid locations, it fetches their geometries and returns a list mapping
    the event pairs to their corresponding OpenStreetMap ID pairs.
    """
    disaster_event_nodes = DisasterEvent.nodes.all()
    event_pairs_can_sn = []
    has_no_geometry = []
    for en in disaster_event_nodes:
        ca_geo_object = en.geo_argument.all()[0]
        ca_geo = ca_geo_object.geo_object_name
        if ca_geo in geo_has_location and ca_geo not in has_no_geometry:
            ca_geometry = get_save_geo_geometry(ca_geo_object, do_encoding)
            if ca_geometry == '':
                has_no_geometry.append(ca_geo)
                continue
            for re_en in en.cause.all():
                re_geo_object = re_en.geo_argument.all()[0]
                re_geo = re_geo_object.geo_object_name
                if re_geo in geo_has_location and re_geo not in has_no_geometry:
                    re_geometry = get_save_geo_geometry(re_geo_object, do_encoding)
                    if re_geometry != '':
                        event_pairs_can_sn.append({'event_pair': [en.node_id, re_en.node_id],
                                                   'osm_ids_pair': [ca_geometry, re_geometry]})
                    else:
                        has_no_geometry.append(re_geo)
    return event_pairs_can_sn


from concurrent.futures import ThreadPoolExecutor, as_completed

def _fetch_geo_info_worker(geo_name, geo_type, origin_text, is_location_known):
    """
    A worker function for multi-threading that handles LLM API requests for location
    validation and geocoding without interacting directly with the database.
    """
    print(f"[Thread-Start] 开始请求大模型/API 解析: {geo_name}")
    result = {
        "geo_name": geo_name,
        "is_location_updated": False,
        "new_is_location": is_location_known,
        "location_info": None
    }

    if is_location_known is None:
        result["new_is_location"] = identify_is_location(geo_name)
        result["is_location_updated"] = True
        print(f"[Thread-Info] {geo_name} 属性判别完成: is_location={result['new_is_location']}")

    if not result["new_is_location"]:
        print(f"[Thread-End] {geo_name} 不是有效地理位置，跳过查询。")
        return result

    result["location_info"] = get_geo_location(origin_text, geo_name, geo_type)
    print(f"[Thread-End] {geo_name} 坐标信息获取完成。")
    return result


def _save_geometry_to_db(geo_node, worker_result):
    """
    Takes the geocoding results processed by the worker thread and safely saves
    the Geometry nodes and relationships into the Neo4j database on the main thread.
    """
    geo_name = worker_result["geo_name"]
    print(f"[DB-Save] 正在处理 {geo_name} 的入库逻辑...")

    if worker_result["is_location_updated"]:
        geo_node.is_location = worker_result["new_is_location"]
        geo_node.save()

    if not geo_node.is_location:
        print(f"  -> [Skip] {geo_name} 判定为不可定位。")
        return ''

    location_info = worker_result["location_info"]

    if location_info and location_info.is_location and location_info.coordinates and location_info.osm_type and location_info.osm_id:
        coordinates = []
        osm_ids = ""

        for co in location_info.coordinates:
            if len(co) == 2:
                coordinates.extend(co)

        if len(location_info.osm_type) == len(location_info.osm_id):
            for i in range(len(location_info.osm_type)):
                osm_type = location_info.osm_type[i]
                o_id = location_info.osm_id[i]

                if osm_type == 'relation':
                    osm_id_str = f"R{o_id}"
                elif osm_type == 'way':
                    osm_id_str = f"W{o_id}"
                elif osm_type == 'node':
                    osm_id_str = f"N{o_id}"
                else:
                    osm_id_str = ""

                if osm_id_str:
                    osm_ids += f",{osm_id_str}" if osm_ids else osm_id_str

        geometry_node = Geometry.nodes.first_or_none(osm_id=osm_ids)
        if not geometry_node:
            geometry_node = Geometry(location=coordinates, osm_id=osm_ids)
            geometry_node.save()
            print(f"  -> [Create] 创建新 Geometry 节点: {osm_ids}")

        if not geo_node.has_geometry.is_connected(geometry_node):
            geo_node.has_geometry.connect(geometry_node)
            print(f"  -> [Connect] 建立关联: {geo_name} <-> {osm_ids}")

        geo_node.is_location = True
        geo_node.save()
        print(f"  -> [Success] {geo_name} 成功落库，OSM_ID: {osm_ids}")
        return osm_ids

    print(f"  -> [Fail] {geo_name} 提取的 location_info 不完整或无效。")
    return ''


def event_pair_geo_encoding_parallel(geo_has_location: list, do_encoding: bool = True):
    """
    A multi-threaded version of event_pair_geo_encoding. It parallelizes the geocoding
    API requests to significantly speed up spatial coordinate extraction for causal event pairs.
    """
    print("\n========== 开始执行事件对地理编码 (event_pair_geo_encoding) ==========")
    disaster_event_nodes = DisasterEvent.nodes.all()
    print(f"[Phase 1] 扫描到 {len(disaster_event_nodes)} 个灾害事件节点，开始收集需要处理的地理实体...")

    nodes_to_process = {}
    osm_id_cache = {}
    has_no_geometry = set()

    for en in disaster_event_nodes:
        ca_geo_object = en.geo_argument.all()[0]
        ca_geo = ca_geo_object.geo_object_name

        if ca_geo in geo_has_location and ca_geo not in has_no_geometry:
            if ca_geo_object.has_geometry.all():
                osm_id_cache[ca_geo] = ca_geo_object.has_geometry.all()[0].osm_id
            elif do_encoding:
                nodes_to_process[ca_geo] = ca_geo_object
            else:
                has_no_geometry.add(ca_geo)

        for re_en in en.cause.all():
            re_geo_object = re_en.geo_argument.all()[0]
            re_geo = re_geo_object.geo_object_name
            if re_geo in geo_has_location and re_geo not in has_no_geometry:
                if re_geo_object.has_geometry.all():
                    osm_id_cache[re_geo] = re_geo_object.has_geometry.all()[0].osm_id
                elif do_encoding:
                    nodes_to_process[re_geo] = re_geo_object
                else:
                    has_no_geometry.add(re_geo)

    print(f"[Phase 1 统计] 已有坐标的实体(缓存): {len(osm_id_cache)} 个")
    print(f"[Phase 1 统计] 缺失坐标需请求的实体: {len(nodes_to_process)} 个")
    print(f"[Phase 1 统计] 明确无坐标的实体: {len(has_no_geometry)} 个\n")

    if do_encoding and nodes_to_process:
        print(f"========== [Phase 2] 启动多线程进行 API 请求 (共 {len(nodes_to_process)} 个任务) ==========")
        tasks_data = []
        for name, node in nodes_to_process.items():
            text_nodes = node.origin.all()
            origin_text = "".join([t.text_content for t in text_nodes])
            tasks_data.append({
                "geo_name": name,
                "geo_type": node.geo_type,
                "origin_text": origin_text,
                "is_location": node.is_location
            })

        worker_results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_name = {
                executor.submit(
                    _fetch_geo_info_worker,
                    t["geo_name"], t["geo_type"], t["origin_text"], t["is_location"]
                ): t["geo_name"]
                for t in tasks_data
            }

            completed_count = 0
            total_tasks = len(tasks_data)
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                completed_count += 1
                try:
                    worker_results[name] = future.result()
                    print(f"[Progress] {completed_count}/{total_tasks} - {name} 任务正常返回。")
                except Exception as exc:
                    print(f"[Error] 进度 {completed_count}/{total_tasks} - [{name}] API 请求异常: {exc}")

        print(f"\n========== [Phase 3] 主线程开始执行数据库写入操作 ==========")
        for name, result in worker_results.items():
            node = nodes_to_process[name]
            osm_id = _save_geometry_to_db(node, result)
            if osm_id:
                osm_id_cache[name] = osm_id
            else:
                has_no_geometry.add(name)

    print(f"\n========== [Phase 4] 组装最终因果对结果 ==========")
    event_pairs_can_sn = []
    for en in disaster_event_nodes:
        ca_geo_object = en.geo_argument.all()[0]
        ca_geo = ca_geo_object.geo_object_name

        if ca_geo not in osm_id_cache:
            continue
        ca_geometry = osm_id_cache[ca_geo]

        for re_en in en.cause.all():
            re_geo_object = re_en.geo_argument.all()[0]
            re_geo = re_geo_object.geo_object_name

            if re_geo in osm_id_cache:
                re_geometry = osm_id_cache[re_geo]
                event_pairs_can_sn.append({
                    'event_pair': [en.node_id, re_en.node_id],
                    'osm_ids_pair': [ca_geometry, re_geometry]
                })

    print(f"[结束] 成功提取并生成 {len(event_pairs_can_sn)} 个具备空间坐标的灾害因果对。==========\n")
    return event_pairs_can_sn


def extract_path_triples(path, rel_tip='relation_name', node_tip='geo_object_name'):
    """
    Converts a Neo4j Path object into a structured list of triples (start_node, relationship, end_node)
    based on the specified property keys.
    """
    triples = []

    for rel in path.relationships:
        start_node = rel.start_node
        end_node = rel.end_node

        start_name = start_node.get(node_tip, '')
        end_name = end_node.get(node_tip, '')

        rel_type = rel.get(rel_tip, '')

        triples.append([start_name, rel_type, end_name])

    return triples


def find_geo_semantics_path(ca_geo_object, re_geo_object):
    """
    Queries the database to find the shortest spatial semantic path (up to 8 hops)
    between two GeoObject nodes connected by 'SPATIAL' relationships.
    """
    if ca_geo_object and re_geo_object:
        query = """
        MATCH path = shortestPath((start:GeoObject)-[:SPATIAL*1..8]-(end:GeoObject))
        WHERE start.node_id = $start_id AND end.node_id = $end_id
        RETURN path
        """

        params = {
            "start_id": ca_geo_object.node_id,
            "end_id": re_geo_object.node_id
        }

        from neomodel import db
        results, meta = db.cypher_query(query, params)

        if results:
            path = results[0][0]

            path_triples = extract_path_triples(path)
            return path_triples

        else:
            return []


def find_geo_type_path(ca_geo_type, re_geo_type):
    """
    Switches the database context to query and return the shortest path between
    two geographic entity types connected by 'SPATIAL' relationships.
    """
    ca_geo_object = GeoObject.nodes.first_or_none(geo_type=ca_geo_type)
    re_geo_object = GeoObject.nodes.first_or_none(geo_type=re_geo_type)
    if ca_geo_object and re_geo_object:
        query = """
        MATCH path = shortestPath((start:GeoObject)-[:SPATIAL*1..6]-(end:GeoObject))
        WHERE start.node_id = $start_id AND end.node_id = $end_id
        RETURN path
        """

        params = {
            "start_id": ca_geo_object.node_id,
            "end_id": re_geo_object.node_id
        }

        from neomodel import db
        results, meta = db.cypher_query(query, params)

        if results:
            path = results[0][0]

            path_triples = extract_path_triples(path, 'relation_name', 'geo_type')
            return path_triples

        else:
            return []


def main_calculate_spatial_relations():
    """
    The main pipeline for calculating spatial relations. It geocodes disaster event pairs,
    computes topographical, distance, and basin relations, extracts semantic graph paths,
    and saves all computed spatial metrics to the relations between DisasterEvents.
    """
    geo_is_location = is_geo_location()

    # 获取两地理对象都进行了地理编码的灾害事件对，do_encoding为True即执行空间编码，为False代表之前已完成地理编码
    event_pairs_encoding = event_pair_geo_encoding_parallel(geo_is_location, False)

    # 执行空间计算，获取拓扑、距离、流域关系
    for pair in event_pairs_encoding:
        ca_osm_ids = pair['osm_ids_pair'][0].replace('-', ',')
        re_osm_ids = pair['osm_ids_pair'][1].replace('-', ',')
        spatial = analyze_spatial_relation(ca_osm_ids, re_osm_ids)
        if spatial == {}:
            continue
        topo_relation = get_topological_relation(spatial["topological_relations"])
        distance = spatial["distance_meters"]
        basin_relation = spatial["basin_relations"]

        ca_event_node = DisasterEvent.nodes.first_or_none(node_id=pair['event_pair'][0])
        re_event_node = DisasterEvent.nodes.first_or_none(node_id=pair['event_pair'][1])

        if ca_event_node and re_event_node:
            if ca_event_node.spatial_to.is_connected(re_event_node):
                spatial_r = ca_event_node.spatial_to.relationship(re_event_node)
                new_relations = spatial_r.relations if spatial_r.relations else {}
                new_relations["topology_computation"] = topo_relation
                new_relations["distance_computation"] = distance
                new_relations["basin_computation"] = basin_relation
                spatial_r.relations = new_relations
                spatial_r.save()
                print(
                    f"1  {ca_event_node.event_description}-{re_event_node.event_description} : {topo_relation}  {distance}m  {basin_relation}")
            else:
                computation_results = {
                    "topology_computation": topo_relation,
                    "distance_computation": distance,
                    "basin_computation": basin_relation
                }
                ca_event_node.spatial_to.connect(
                    re_event_node, {'relations': computation_results})
                print(f"2  {ca_event_node.event_description}-{re_event_node.event_description} : {topo_relation}  {distance}m  {basin_relation}")


    event_pair_all = []
    disaster_event_nodes = DisasterEvent.nodes.all()
    for en in disaster_event_nodes:
        ca_geo_object = en.geo_argument.all()[0]
        ca_geo = ca_geo_object.geo_object_name
        ca_geo_type = ca_geo_object.geo_type
        for re_en in en.cause.all():
            re_geo_object = re_en.geo_argument.all()[0]
            re_geo = re_geo_object.geo_object_name
            re_geo_type = re_geo_object.geo_type

            event_pair_all.append({'events_node': [en.node_id, re_en.node_id],
                                   'events_desc': [en.event_description, re_en.event_description],
                                   'geos_type': [ca_geo_type, re_geo_type]})

            if en.spatial_to.is_connected(re_en):
                spatial_r = en.spatial_to.relationship(re_en)
            else:
                spatial_r = en.spatial_to.connect(re_en)

            new_relations = spatial_r.relations if spatial_r.relations else {}

            geo_semantics_path = []
            if ca_geo != re_geo:
                geo_semantics_path = find_geo_semantics_path(ca_geo_object, re_geo_object)
                print(f"{en.event_description}-{re_en.event_description} : {geo_semantics_path}")

            new_relations["semantics_path"] = geo_semantics_path
            spatial_r.relations = new_relations
            spatial_r.save()

    ori_database = neo.database_name
    neo.connect_database('geos-type-1')
    for i in event_pair_all:
        ca_geo_type = i['geos_type'][0]
        re_geo_type = i['geos_type'][1]

        geo_type_path = []
        if ca_geo_type != re_geo_type:
            geo_type_path = find_geo_type_path(ca_geo_type, re_geo_type)
            print(f"{i['events_desc'][0]}-{i['events_desc'][1]} : {geo_type_path}")

        i['type_path'] = geo_type_path
    neo.connect_database(ori_database)

    for i in event_pair_all:
        en = DisasterEvent.nodes.first_or_none(node_id=i['events_node'][0])
        re_en = DisasterEvent.nodes.first_or_none(node_id=i['events_node'][1])
        if en and re_en:
            if en.spatial_to.is_connected(re_en):
                spatial_r = en.spatial_to.relationship(re_en)
            else:
                spatial_r = en.spatial_to.connect(re_en)
            new_relations = spatial_r.relations if spatial_r.relations else {}
            new_relations["type_path"] = i['type_path']
            spatial_r.relations = new_relations
            spatial_r.save()


def main_calculate_spatial_relations_only():
    """
    A streamlined version of the spatial relations pipeline that exclusively calculates
    and updates the geographic type-level paths ('type_path') between causally linked DisasterEvents.
    """
    event_pair_all = []
    disaster_event_nodes = DisasterEvent.nodes.all()
    for en in disaster_event_nodes:
        ca_geo_object = en.geo_argument.all()[0]
        ca_geo = ca_geo_object.geo_object_name
        ca_geo_type = ca_geo_object.geo_type
        for re_en in en.cause.all():
            re_geo_object = re_en.geo_argument.all()[0]
            re_geo = re_geo_object.geo_object_name
            re_geo_type = re_geo_object.geo_type

            event_pair_all.append({'events_node': [en.node_id, re_en.node_id],
                                   'events_desc': [en.event_description, re_en.event_description],
                                   'geos_type': [ca_geo_type, re_geo_type]})

    ori_database = neo.database_name
    neo.connect_database('geos-type-1')
    for i in event_pair_all:
        ca_geo_type = i['geos_type'][0]
        re_geo_type = i['geos_type'][1]

        geo_type_path = []
        if ca_geo_type != re_geo_type:
            geo_type_path = find_geo_type_path(ca_geo_type, re_geo_type)
            print(f"{i['events_desc'][0]}-{i['events_desc'][1]} : {geo_type_path}")

        i['type_path'] = geo_type_path
    neo.connect_database(ori_database)

    for i in event_pair_all:
        en = DisasterEvent.nodes.first_or_none(node_id=i['events_node'][0])
        re_en = DisasterEvent.nodes.first_or_none(node_id=i['events_node'][1])
        if en and re_en:
            if en.spatial_to.is_connected(re_en):
                spatial_r = en.spatial_to.relationship(re_en)
            else:
                spatial_r = en.spatial_to.connect(re_en)
            new_relations = spatial_r.relations if spatial_r.relations else {}
            new_relations["type_path"] = i['type_path']
            spatial_r.relations = new_relations
            spatial_r.save()


def main_geo_extraction_henan():
    """
    Reads text chunks from a JSON dataset (wiki-henan-1), extracts geographic entities
    and spatial relations using the LLM, and sequentially inserts them into the database.
    """
    with open('data/wiki-henan-1.json', "r", encoding="utf-8") as f:
        wiki_data = json.load(f)
    de_num = 0
    for key, value in wiki_data.items():
        # if de_num >= 1:
        #     break
        for index in range(len(value)):
            # print(v)
            text = value[index]
            print("text:\n", text)
            llm_result = geo_ner_re(f'(title:{key})-' + text)
            print("raw:\n", llm_result)
            results = parse_geo_string_to_dict(llm_result)
            print("dict:\n", results)
            insert_database_geo(results, text, key)


def main_geo_extraction_henan_ref():
    """
    Reads structured reference data from a JSON dataset (wiki-henan-ref-1), extracts
    geographic entities and spatial relations using the LLM, and inserts them into the database.
    """
    with open('data/wiki-henan-ref-1.json', "r", encoding="utf-8") as f:
        wiki_data = json.load(f)

    for value in wiki_data:
        title = value['title']
        for index in range(len(value['content'])):
            # print(v)
            text = value['content'][index]
            print("text:\n", text)
            llm_result = geo_ner_re(text)
            print("raw:\n", llm_result)
            results = parse_geo_string_to_dict(llm_result)
            print("dict:\n", results)
            insert_database_geo(results, text, title)


if __name__ == '__main__':
    neo = Neo4jDatabaseConnector('events-all-1-mini')
    input_text = ("7月20日的暴雨导致郑州地铁多处车站和隧道被淹，郑州地铁18时10分全线网停运[88][89]。当时郑州地铁5号线五龙口停车场内的水深几乎快到2米["
                  "90]。7月20日18时左右，积水冲垮出入场线挡水墙进入5号线海滩寺站和沙口路站之间的行车隧道，导致一列编号0501的列车被困隧道内并进水["
                  "91]，车厢外积水高过车厢内，积水漫过乘客胸口，最高水位甚至到达乘客脖子。郑州地铁5号线严重水浸事故已导致14人死亡，5人受伤[14][92][89]。7月21日上午，郑州地铁官方网站变为黑白色["
                  "93]。")
    result = geo_ner_re(input_text)
    parsed_result = parse_geo_string_to_dict(result)
    insert_database_geo(parsed_result, input_text, "河南暴雨")

    main_geo_extraction_henan()

    delete_all_geometry_node()
    geo_location()
    geos = GeoObject.nodes.all()
    geo_num = len(geos)
    gm_num = 0
    for geo in geos:
        if geo.has_geometry.all():
            gm_num += 1
    print(geo_num)
    print(gm_num, gm_num/geo_num)

    main_geo_extraction_henan_ref()

    geo_is_location = is_geo_location()
    event_pairs_encoding = event_pair_geo_encoding(geo_is_location)
    print(event_pairs_encoding)
    for geom in Geometry.nodes.all():
        osm_ids = str(geom.osm_id).replace('-', ',')
        print(get_entity_combined_geometry(osm_ids))

    # 执行地理编码，识别空间关系
    main_calculate_spatial_relations_only()
    geo_is_location = is_geo_location()
    event_pair_geo_encoding_parallel(geo_is_location, True)
    main_calculate_spatial_relations()
