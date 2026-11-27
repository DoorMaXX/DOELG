from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.construct_config import get_chat_model_options, get_setting
from src.graph_language import english_geo_results, english_term
from langchain.agents import create_agent
from ontology.neo4jmodel import GeoObject, DisasterEvent, OriginText, Geometry
import re
import math
from other.geo_encoding import get_location
from tqdm import tqdm
from other.geo_encoding import analyze_spatial_relation, get_topological_relation, normalize_osm_ids
from src.other.geo_type_graph import canonical_geo_type

"""
1. Extract 'geographic object' entities and spatial relations between geographic objects from text.
2. Identify spatial relations between disaster events (geocoding & spatial computation + path extraction).
"""

llm = ChatOpenAI(**get_chat_model_options('geo'))


def geo_ner_re(text: str):
    """
    Extracts static geographic entities (Features) and their static spatial relationships
    from the provided text using a Large Language Model and a predefined ontology.


    This is the primary execution function. Since the corpus is predominantly
    Chinese, the prompt is in Chinese. See geo_ner_re_en for the English version.
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


def geo_ner_re_en(text: str):
    """
    Extracts static geographic entities (Features) and their static spatial relationships
    from the provided text using a Large Language Model and a predefined ontology.
    """
    ontology_prompt = """
        class Entity:
            # Base class for all entities
            def __init__(self, name:str):
                self.name = name

        class Spatial_object(Entity):
            # General abstract class for geographic objects
            pass

        class Geometry(Spatial_object):
            # Geographic extent shape, restricted to 'point', 'linestring', or 'polygon'
            def __init__(self, name:str):
                self.name = name
                if name not in ['point', 'linestring', 'polygon']:
                    self.name = ''

        class Feature(Spatial_object):
            # Definition: A real, stationary geographic feature mentioned in the text.
            # Scope: A specific geographic extent or a general feature type, including natural landforms and infrastructure.
            # [Important negative constraints]:
            # 1. Do not extract moving objects such as trains, vehicles, cars, vessels, aircraft, pedestrians, or animals.
            # 2. Do not extract natural phenomena such as floods, standing water, or rain puddles.
            def __init__(self, 
                name:str,   #Geographic feature name, such as Wuhan, Dongting Lake, or Nanjing Road
                geo_type:str,   #Geographic feature type, such as city, lake, or road
                geometry:Geometry   #Geographic extent shape type
            ):
                self.name = name
                self.geo_type = geo_type
                self.geometry = geometry
                # Allowed types (stationary features only):
                valid_types = [
                'River', 'Lake', 'Basin', 'Islet',
                'Shoal', 'Spring', 'Waterfall', 'Mountain', 'Cave', 'Sinkhole', 'Earth Bank', 'Valley', 'Sea', 'Coastline', 'Island', 'Farmland', 'Forest', 'Meadow',
                'Sand', 'Wetland', 'Ditch', 'Reservoir', 'Pond', 'Water Well', 'Lock Gate', 'Dam', 'Dyke',
                'Pumping Station', 'Flood Storage Area', 'Highway', 'Railway', 'Subway', 'Waterway', 'Bridge', 'Tunnel', 'Aerialway', 'Underpass', 'Underpass',
                'Traffic Intersection', 'Train Station', 'Metro Station', 'Tram Station', 'Dock', 'Port', 'Lighthouse', 'Aerodrome', 'Apron',
                'Underground Parking', 'Ground Parking', 'In-Building Parking', 'Bus Station', 'Charging Station', 'Building', 'Underground Building', 'Industrial Structure',
                'Public Service Structure', 'Observation Structure', 'Enclosure Structure', 'Cable', 'Tower', 'Utility Pole', 'Substation',
                'Telecom Line', 'Telecom Base Station', 'Communications Tower', 'Pipeline', 'Storage Tank', 'Water Tower', 'Sewer', 'Sewer Vent', 'Residential Area',
                'Public Management and Service Area', 'Commercial Area', 'Mining Area', 'Warehouse', 'Greenery', 'Farmyard', 'Infrastructure Area', 'Administrative Region'
                ]
                if geo_type not in valid_types:
                    self.geo_type = ''
        
        class Relation:
            # Base class for all relations.
            def __init__(self, head_entity: Entity, tail_entity: Entity):
                self.head_entity = head_entity
                self.tail_entity = tail_entity
        
        class SpatialRelation(Relation):
            # Definition: A static spatial relationship between two features.
            # Categories:
            # 1. Topology: contains, within, intersects, adjacent, connected, etc.
            # 2. Distance: far, close, nearby, etc.
            # 3. Orientation: upstream, downstream, east, north, high or low elevation, center, surroundings, etc.
            def __init__(self, head_feature: Feature, tail_feature: Feature, spatial_relation_name: str):
                super.__init__(head_feature, tail_feature)
                self.spatial_relation_name = spatial_relation_name
                
    """

    system_prompt = ("This is an object-oriented programming task. Entity and Relation classes are defined above. Convert the text into Feature and SpatialRelation"
                     " objects, subject to these requirements:"
                     "1. Extract only explicitly stated information without speculation. Output only results, strictly following the example."
                     "2. Feature extraction: extract stationary geographic features only; automatically ignore trains, vehicles, and other moving objects."
                     "3. SpatialRelation extraction: extract only static relations between features; automatically filter dynamic actions such as driving to or flowing into.")

    chat_example_1 = [
        "The Baisha Town area of Zhengdong New District lies near the Jialu River and downstream of flood discharge from Changzhuang Reservoir; it was severely affected.",
        """
        result=[Feature('Baisha Town, Zhengdong New District', 'Administrative Region', Geometry('polygon')),
                Feature('Jialu River', 'River', Geometry('linestring')),
                Feature('Changzhuang Reservoir', 'Reservoir', Geometry('polygon')),
                SpatialRelation(Feature('Baisha Town, Zhengdong New District', 'Administrative Region', Geometry('polygon')), Feature('Jialu River', 'River', Geometry('linestring')), 'near')
                SpatialRelation(Feature('Baisha Town, Zhengdong New District', 'Administrative Region', Geometry('polygon')), Feature('Changzhuang Reservoir', 'Reservoir', Geometry('polygon')), 'downstream of')
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
    llm_result = english_geo_results(llm_result)
    if not llm_result['features']:
        print(f'[Geo record skipped] {title}: no valid geographic features remain.')
        return False
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
    return True


def identify_is_location(geo_name: str):
    """
    Uses the LLM to determine whether a given geographic name represents a specific,
    real-world physical location (returning True or False).


    This is the primary execution function. Since the corpus is predominantly
    Chinese, the prompt is in Chinese. See identify_is_location_en for the English version.
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


def identify_is_location_en(geo_name: str):
    """
    Uses the LLM to determine whether a given geographic name represents a specific,
    real-world physical location (returning True or False).
    """
    prompt_template = ChatPromptTemplate(
        [
            ("system",
             "You are a geography expert. Determine whether the geographic name refers to a specific real-world location or extent. Return True if it does, otherwise False."),
            ("human", "Dongting Lake"),
            ("ai", "True"),
            ("human", "Subway"),
            ("ai", "False"),
            ("human", "Village"),
            ("ai", "False"),
            ("human", "Jurong Reservoir"),
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


    This is the primary execution function. Since the corpus is predominantly
    Chinese, the prompt is in Chinese. See get_geo_location_en for the English version.
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


def get_geo_location_en(text, geo_name, geo_type):
    """
    Invokes an LLM agent equipped with a geocoding tool to query OpenStreetMap.
    It resolves ambiguities using the provided text context and returns the location validity,
    coordinates, and OSM IDs.
    """
    sys_prompt = f"""
        You are a GIS and geocoding expert familiar with semantic correspondence between OpenStreetMap (OSM) and real-world geographic objects.
        ## Core task
        Your task is to assess geographic locatability and filter search results, determining whether the name refers to a specific, uniquely or clearly bounded real-world entity.
        When search results contain multiple objects with the same name, disambiguate using the reference text.
        ### 🛑 Workflow and key constraints
        1. **Use the tool**: Call `get_location` exactly once with an unambiguous normalized name for '{geo_name}', including address context such as county, city, or province.
        2. **Analyze and disambiguate** using the `get_location` output and reference-text context.
        3. **Final output**: Once selection is complete, immediately stop tool calls and return `GeoInfo` in strict Pydantic format.
        4. **No repeated queries**: Regardless of results, do not call `get_location` again during the same invocation.
    """

    from pydantic import BaseModel, Field
    from typing import List

    class GeoInfo(BaseModel):
        is_location: bool = Field(description=f"Whether the geocoding result matches the input geographic object '{geo_name}'")
        coordinates: List[List[float]] = (
            Field(
                description="If 'is_location' is 'True', select the most accurate coordinates for the named object where possible, "
                            "or multiple coordinates for different components; return [[lon1, lat1], ...]."))
        osm_type: List[str] = Field(description="If 'is_location' is 'True', the selected geographic object's osm_type")
        osm_id: List[int] = Field(description="If 'is_location' is 'True', the selected geographic object's osm_id")

    agent = create_agent(llm, tools=[get_location], system_prompt=sys_prompt, response_format=GeoInfo)
    output = agent.invoke(
        {"messages": [
            {"role": "user", "content": f"Reference text: ‘{text}’"},
            {"role": "user", "content": f"Resolve the geographic object in the reference text of type '{geo_type}' named ‘{geo_name}’"}
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
    for geo_node in tqdm(geo_nodes, desc="Geographic entity locatability"):
        geo_name = geo_node.geo_object_name
        if geo_node.is_location is None:
            geo_node.is_location = identify_is_location(geo_name)
            geo_node.save()
        if geo_node.is_location:
            geos_location.append(geo_name)
    return geos_location


def is_geo_location_parallel(max_workers=None):
    """
    A multi-threaded version of is_geo_location. It batches API requests to the LLM
    for location instantiation_validation and subsequently updates the Neo4j database in the main thread.
    """
    if max_workers is None:
        max_workers = get_setting('geocoding.location_workers')
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
            print(f"Processing entity {name} failed: {e}")
            return []

    collected_data = []
    print(f"🚀 Starting parallel API requests (workers: {max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(fetch_location_data, name): name for name in tasks}

        for future in tqdm(as_completed(future_to_item), total=len(tasks), desc="LLM classification"):
            res = future.result()
            if res:
                collected_data.extend(res)

    if collected_data:
        print(f"Saving... {len(collected_data)} records to insert")
        for item in collected_data:
            name = item['geo_name']
            is_loc = item['is_location']

            node = name_to_node_map.get(name)
            if node:
                node.is_location = is_loc
                node.save()

            if is_loc:
                geos_location.append(name)

        print("✅ Geographic entity processing and database insertion completed.")
    else:
        print("⚠️ No valid data extracted.")

    return geos_location


def _get_cached_osm_ids(geo_node):
    """Return valid cached OSM IDs while ignoring previously saved empty identifiers."""
    for geometry in geo_node.has_geometry.all():
        identifiers = normalize_osm_ids(geometry.osm_id)
        if identifiers:
            return identifiers
    return ''


def get_save_geo_geometry(geo_node, do_encoding=False):
    """Return valid cached IDs or geocode and persist the location through shared validation."""
    cached = _get_cached_osm_ids(geo_node)
    if cached:
        print(f"{geo_node.geo_object_name} located")
        return cached
    if not do_encoding:
        return ''
    origin_text = "".join(node.text_content for node in geo_node.origin.all())
    result = _fetch_geo_info_worker(geo_node.geo_object_name, geo_node.geo_type,
                                    origin_text, geo_node.is_location)
    return _save_geometry_to_db(geo_node, result)


def geo_location():
    """Geocode missing or invalid location caches using the shared geometry validation."""
    for geo_node in tqdm(GeoObject.nodes.all(), desc="Geocoding"):
        get_save_geo_geometry(geo_node, do_encoding=True)


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
    instantiation_validation and geocoding without interacting directly with the database.
    """
    print(f"[Thread-Start] Requesting model/API analysis: {geo_name}")
    result = {
        "geo_name": geo_name,
        "is_location_updated": False,
        "new_is_location": is_location_known,
        "location_info": None
    }

    if is_location_known is None:
        result["new_is_location"] = identify_is_location(geo_name)
        result["is_location_updated"] = True
        print(f"[Thread-Info] {geo_name} attribute classification completed: is_location={result['new_is_location']}")

    if not result["new_is_location"]:
        print(f"[Thread-End] {geo_name} is not a valid geographic location; skipping lookup.")
        return result

    result["location_info"] = get_geo_location(origin_text, geo_name, geo_type)
    print(f"[Thread-End] {geo_name} coordinates retrieved.")
    return result


def _save_geometry_to_db(geo_node, worker_result):
    """
    Takes the geocoding results processed by the worker thread and safely saves
    the Geometry nodes and relationships into the Neo4j database on the main thread.
    """
    geo_name = worker_result["geo_name"]
    print(f"[DB-Save] Processing {geo_name} for database insertion...")

    if worker_result["is_location_updated"]:
        geo_node.is_location = worker_result["new_is_location"]
        geo_node.save()

    if not geo_node.is_location:
        print(f"  -> [Skip] {geo_name} classified as not locatable.")
        return ''

    location_info = worker_result["location_info"]
    validated = _validated_location_geometry(location_info)
    if validated is not None:
        coordinates, osm_ids = validated
        geometry_node = Geometry.nodes.first_or_none(osm_id=osm_ids)
        if not geometry_node:
            geometry_node = Geometry(location=coordinates, osm_id=osm_ids)
            geometry_node.save()
            print(f"  -> [Create] Created Geometry node: {osm_ids}")

        if not geo_node.has_geometry.is_connected(geometry_node):
            geo_node.has_geometry.connect(geometry_node)
            print(f"  -> [Connect] Connected: {geo_name} <-> {osm_ids}")

        geo_node.is_location = True
        geo_node.save()
        print(f"  -> [Success] {geo_name} saved to database, OSM_ID: {osm_ids}")
        return osm_ids

    print(f"  -> [Fail] {geo_name} extracted location_info is incomplete or invalid.")
    return ''


def _validated_location_geometry(location_info):
    """Validate coordinates and typed OSM IDs before creating any Geometry node."""
    if location_info is None or not location_info.is_location:
        return None
    coordinates = location_info.coordinates
    osm_types = location_info.osm_type
    osm_values = location_info.osm_id
    if (not coordinates or not osm_types or not osm_values
            or len(osm_types) != len(osm_values)):
        return None
    prefixes = {'node': 'N', 'way': 'W', 'relation': 'R'}
    identifiers = []
    for osm_type, osm_value in zip(osm_types, osm_values):
        prefix = prefixes.get(str(osm_type).strip().lower())
        value = str(osm_value).strip()
        if not prefix or not re.fullmatch(r'[1-9][0-9]*', value):
            return None
        identifiers.append(prefix + value)
    flattened = []
    try:
        for coordinate in coordinates:
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                return None
            lon, lat = map(float, coordinate)
            if not math.isfinite(lon) or not math.isfinite(lat) or not (-180 <= lon <= 180 and -90 <= lat <= 90):
                return None
            flattened.extend((lon, lat))
    except (TypeError, ValueError, OverflowError):
        return None
    return flattened, ','.join(dict.fromkeys(identifiers))


def event_pair_geo_encoding_parallel(geo_has_location: list, do_encoding: bool = None):
    """
    A multi-threaded version of event_pair_geo_encoding. It parallelizes the geocoding
    API requests to significantly speed up spatial coordinate extraction for causal event pairs.
    """
    if do_encoding is None:
        do_encoding = get_setting('geocoding.do_encoding')
    print("\n========== Starting event-pair geocoding (event_pair_geo_encoding) ==========")
    disaster_event_nodes = DisasterEvent.nodes.all()
    print(f"[Phase 1] Found {len(disaster_event_nodes)} disaster events; collecting geographic entities...")

    nodes_to_process = {}
    osm_id_cache = {}
    has_no_geometry = set()

    for en in disaster_event_nodes:
        ca_geo_object = en.geo_argument.all()[0]
        ca_geo = ca_geo_object.geo_object_name

        if ca_geo in geo_has_location and ca_geo not in has_no_geometry:
            cached_ids = _get_cached_osm_ids(ca_geo_object)
            if cached_ids:
                osm_id_cache[ca_geo] = cached_ids
            elif do_encoding:
                nodes_to_process[ca_geo] = ca_geo_object
            else:
                has_no_geometry.add(ca_geo)

        for re_en in en.cause.all():
            re_geo_object = re_en.geo_argument.all()[0]
            re_geo = re_geo_object.geo_object_name
            if re_geo in geo_has_location and re_geo not in has_no_geometry:
                cached_ids = _get_cached_osm_ids(re_geo_object)
                if cached_ids:
                    osm_id_cache[re_geo] = cached_ids
                elif do_encoding:
                    nodes_to_process[re_geo] = re_geo_object
                else:
                    has_no_geometry.add(re_geo)

    print(f"[Phase 1] Entities with cached coordinates: {len(osm_id_cache)}")
    print(f"[Phase 1] Entities requiring coordinate lookup: {len(nodes_to_process)}")
    print(f"[Phase 1] Entities without coordinates: {len(has_no_geometry)}\n")

    if do_encoding and nodes_to_process:
        print(f"========== [Phase 2] Starting parallel API requests ({len(nodes_to_process)} tasks) ==========")
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
        with ThreadPoolExecutor(max_workers=get_setting('geocoding.encoding_workers')) as executor:
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
                    print(f"[Progress] {completed_count}/{total_tasks} - {name} completed.")
                except Exception as exc:
                    print(f"[Error] Progress {completed_count}/{total_tasks} - [{name}] API request failed: {exc}")

        print(f"\n========== [Phase 3] Main thread is writing to the database ==========")
        for name, result in worker_results.items():
            node = nodes_to_process[name]
            osm_id = _save_geometry_to_db(node, result)
            if osm_id:
                osm_id_cache[name] = osm_id
            else:
                has_no_geometry.add(name)

    print(f"\n========== [Phase 4] Assembling final causal pairs ==========")
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

    print(f"[Complete] Successfully extracted and generated {len(event_pairs_can_sn)} disaster causal pairs with spatial coordinates.==========\n")
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
    Queries the database to find the shortest spatial semantic path within the configured hop limit
    between two GeoObject nodes connected by 'SPATIAL' relationships.
    """
    if ca_geo_object and re_geo_object:
        max_hops = int(get_setting('geocoding.semantic_path_max_hops'))
        query = f"""
        MATCH path = shortestPath((start:GeoObject)-[:SPATIAL*1..{max_hops}]-(end:GeoObject))
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
    cause_type = canonical_geo_type(english_term(ca_geo_type, strict=True))
    result_type = canonical_geo_type(english_term(re_geo_type, strict=True))
    if not cause_type or not result_type or cause_type == result_type:
        return []
    ca_geo_object = GeoObject.nodes.first_or_none(geo_type=cause_type)
    re_geo_object = GeoObject.nodes.first_or_none(geo_type=result_type)
    if ca_geo_object and re_geo_object:
        max_hops = int(get_setting('geocoding.type_path_max_hops'))
        query = f"""
        MATCH path = shortestPath((start:GeoObject)-[:SPATIAL*1..{max_hops}]-(end:GeoObject))
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


def _populate_geo_type_paths(event_pairs, neo_connector):
    """Read geographic type paths and always restore the caller's database connection."""
    if not event_pairs:
        return
    original_database = neo_connector.database_name
    try:
        neo_connector.connect_database(get_setting('geocoding').get('type_database', 'geos-type-1'))
        for pair in event_pairs:
            cause_type, result_type = pair['geos_type']
            pair['type_path'] = (find_geo_type_path(cause_type, result_type)
                                 if cause_type != result_type else [])
            print(f"{pair['events_desc'][0]}-{pair['events_desc'][1]} : {pair['type_path']}")
    finally:
        neo_connector.connect_database(original_database)


def main_calculate_spatial_relations(neo_connector):
    """
    The main pipeline for calculating spatial relations. It geocodes disaster event pairs,
    computes topographical, distance, and basin relations, extracts semantic graph paths,
    and saves all computed spatial metrics to the relations between DisasterEvents.
    Pass the active database connector so type-path queries can restore its database.
    """
    geo_is_location = is_geo_location()

    # Reuse encoded pairs without repeating the LLM geocoding stage.
    event_pairs_encoding = event_pair_geo_encoding_parallel(geo_is_location, False)

    computed_pairs = 0
    skipped_pairs = 0
    for pair in event_pairs_encoding:
        ca_osm_ids = pair['osm_ids_pair'][0].replace('-', ',')
        re_osm_ids = pair['osm_ids_pair'][1].replace('-', ',')
        spatial = analyze_spatial_relation(ca_osm_ids, re_osm_ids)
        if spatial == {}:
            skipped_pairs += 1
            print(f"[Spatial] No computed relation saved for event pair {pair['event_pair']}.")
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
            computed_pairs += 1
        else:
            skipped_pairs += 1

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

    _populate_geo_type_paths(event_pair_all, neo_connector)

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

    summary = {
        'total_pairs': len(event_pair_all),
        'eligible_pairs': len(event_pairs_encoding),
        'computed_pairs': computed_pairs,
        'skipped_pairs': skipped_pairs,
        'missing_coordinate_pairs': max(0, len(event_pair_all) - len(event_pairs_encoding)),
    }
    print(f"[Spatial summary] {computed_pairs}/{len(event_pairs_encoding)} eligible pairs computed; "
          f"{skipped_pairs} skipped; {summary['missing_coordinate_pairs']} pairs lack valid OSM IDs.")
    return summary


def main_calculate_spatial_relations_only(neo_connector):
    """
    A streamlined version of the spatial relations pipeline that exclusively calculates
    and updates the geographic type-level paths ('type_path') between causally linked DisasterEvents.
    Pass the active database connector to preserve the caller's selected database.
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

    _populate_geo_type_paths(event_pair_all, neo_connector)

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
