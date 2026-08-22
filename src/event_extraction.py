from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from ontology.neo4jmodel import Neo4jDatabaseConnector, GeoObject, DisPhen, DisasterEvent, OriginText, ObservationEvent
import json
import ast
import re
from datetime import datetime
import textwrap

"""
Disaster and observation event extraction,
alongside disaster-causal and observation-triggering relationship identification.
"""

llm = ChatOpenAI(temperature=0.3,
                 model="deepseek-chat",
                 openai_api_base="https://api.deepseek.com",
                 openai_api_key="")


geo_types = [
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
impact_types = ['大气对流', '极端降水', '风力增强', '水位上涨', '水流湍急', '水体漫溢', '泄洪分洪', '壅水顶托',
                '水体倒灌', '排水不畅', '泥沙淤积', '水土流失', '堤坝决口', '洪水冲刷', '积水内涝', '侵入淹没',
                '淤积覆盖',
                '水流冒溢', '结构崩塌', '溃决断裂', '滑坡', '泥石流', '山洪', '堰塞阻塞', '塌陷沉降', '交通中断',
                '能源中断', '通信中断', '运行管制', '排涝失效', '人员伤亡', '人员受困', '撤离疏散', '资源告急',
                '生产损失', '水质污染', '漂浮物堆积', '植物病虫害', '疫病扩散']
sensor_types = ['主动光学传感器', '被动光学传感器', '主动微波传感器', '被动微波传感器', '其它传感器']
platform_types = ['航天观测平台', '航空观测平台', '地面固定平台', '地面移动平台']



def event_extraction_combined(text: str):
    """
    Extracts both disaster events and observation events from the provided text using a unified LLM prompt.
    It defines an ontology and uses few-shot prompting to instruct the LLM to output both `disaster_results`
    and `observation_results` instances.
    """

    ontology_prompt = f"""
        class Entity:
            # 所有实体的基类
            def __init__(self, name: str):
                self.name = name
    
    
        class Spatial_object(Entity):
            # 地理空间对象的通用抽象类
            pass
        
        
        class Geometry(Spatial_object):
            # 地理空间对象的空间范围形状类型，限定为'point'、'linestring'、'polygon'
            def __init__(self, name: str):
                super().__init__(name)
                if name not in ['point', 'linestring', 'polygon']:
                    self.name = ''
        
        
        class Feature(Spatial_object):
            # 定义：文本中提及的现实【静止】地理空间要素实体。
            # 范围：指向现实某一具体空间范围或为一类地物的统称，包括自然地貌（山、河）和人工设施（路、楼）。
            # 【重要约束】：
            # 1. 严禁抽取可移动物体（如：车辆、船只、行人）。
            # 2. 严禁抽取自然现象（如：洪水、积水、地震、由于降雨形成的临时水坑等）作为Feature，它们属于DisPhen。
            # 3. 严禁将“监测指标/因子”作为Feature名称（如：灾害范围）。
            def __init__(self,
                         name: str,  # 地理空间要素实体名称，如武汉市、洞庭湖
                         geo_type: str,  # 地理空间要素实体类型，如城市、湖泊
                         geometry: Geometry  # 地理空间要素空间范围形状类型
                         ):
                super().__init__(name)
                self.geo_type = geo_type
                self.geometry = geometry
                valid_types = {geo_types}
                if geo_type not in valid_types:
                    self.geo_type = ''
        
        
        class DisPhen(Entity):
            # 灾害场景下地理对象上发生的灾害现象，变化表现为遭受非人为的灾害破坏，或可能引发次级灾害的地理现象
            # 示例:降水量增加、积水、水位上升、淹没、堵塞、崩塌
            def __init__(self,
                         name: str,  # 文本中描述的灾害地理现象
                         impact_type: str,  # 所属灾害地理现象类型
                         ):
                super().__init__(name)
                self.name = name
                self.impact_type = impact_type
                valid_types = {impact_types}
                if impact_type not in valid_types:
                    self.impact_type = ''
        
        
        class Sensor(Entity):
            # 定义：执行地理环境观测任务的硬件设备或系统。
            # 【重要负面约束】：
            # 1. 严禁将“算法、数学模型、数据处理方法”识别为传感器。
            #    错误示例：'所优选的模型'、'时序分析法'、'深度学习算法'。
            # 2. 只有明确提到硬件（如：微波辐射计、SAR传感器、光学成像仪）时才抽取。
            def __init__(self,
                         name: str,  # 文本中描述的传感器名称
                         sensor_type: str,  # 传感器类型
                         sensor_platform: str  # 传感器所属平台
                         ):
                super().__init__(name)
        
                self.sensor_type = sensor_type
                valid_types = {sensor_types}
                if sensor_type not in valid_types:
                    self.sensor_type = ''
        
                self.sensor_platform = sensor_platform
                valid_platforms = {platform_types}
                if sensor_platform not in valid_platforms:
                    self.sensor_platform = ''
    
    
        class Observed_factor(Entity):
            # 定义：被传感器观测到的地理对象或环境的【特定属性、参数或状态】。
            # 【重要正向约束】：
            # 1. 必须是描述性的词汇，如“范围”、“深度”、“浓度”、“位移”、“植被指数”。
            # 2. 必须是可度量或可提取的状态。
            # 【重要负面约束】：
            # 1. 严禁将“灾害事件名称”识别为因子。
            #    错误示例：'洪涝灾害'、'森林火灾'、'地震'、'病虫害'。
            #    正确做法：从上述事件中提取属性，如 '洪水淹没范围'、'过火面积'、'地表形变量'。
            # 2. 如果文本只提到了监测灾害（如“监测洪涝灾害”），而没有提到具体指标，则 observed_factor 应为空。
            pass
        
        
        class Feature_factor(Observed_factor):
            # 可被传感器观测的地理空间对象本身的状态属性因子
            # 示例:积水水位、河流流速、边坡应力、建筑结构稳定性、道路车流量
            pass
        
        
        class Environment_factor(Observed_factor):
            # 地理空间对象所处环境中存在的可被传感器观测的属性因子
            # 示例:降水量、风向、地形高程、地表温度、空气湿度
            pass
        
        
        class Event:
            # 所有事件的基类，事件必须的主体为[主语(subject),谓语(predicate),宾语(object)]
            def __init__(self, subject, predicate, object):
                self.subject = subject
                self.predicate = predicate
                self.object = object
        
        
        from datetime import datetime
        from typing import List
        
        
        class Disaster(Event):
            # 灾害事件，事件主体定义为[某一地理空间对象(spatial_object),发生,某一现象(impact)]，事件论元包括若干可观测因子(observed_factors)和该灾害事件发生时间(time_occurrence)
            def __init__(
                    self,
                    spatial_object: Feature | Geometry,
                    impact: DisPhen,
                    observed_factors: List[Feature_factor | Environment_factor] = [],
                    time_occurrence: datetime = None
            ):
                super().__init__(subject=spatial_object, predicate='发生', object=impact)
                self.observed_factors = observed_factors
                self.time_occurrence = time_occurrence
        
        
        class Observation(Event):
            # 观测事件，事件主体定义为[某一传感器(sensor),观测,某对象的某个观测因子([observed_object, observed_factors])]，事件论元包括观测范围(observed_span)和观测发生时间(time_occurrence)
            def __init__(
                    self,
                    sensor: Sensor,
                    observed_object: Feature | Geometry = None,
                    observed_factors: List[Feature_factor | Environment_factor] = [],
                    observed_span: Geometry = None,
                    time_occurrence: datetime = None
            ):
                super().__init__(subject=sensor, predicate='观测', object=[observed_object, observed_factors])
                self.observed_span = observed_span
                self.time_occurrence = time_occurrence
    """

    system_prompt = (
        "这是一个面向对象的编程任务，上面定义了实体（Entity）及事件（Event）类。"
        "请分析文本，提取两类信息："
        "1. **Disaster**（灾害事件）：提取受灾体和灾害影响。"
        "2. **Observation**（观测事件）：提取传感器、观测对象及观测因子。"
        "请仅提取文本明确提及的信息。必须输出两个变量：`disaster_results` 和 `observation_results`。"
    )

    chat_example_1 = [
        "受持续强降雨影响，2016年5月18日鄱阳湖流域发生超警戒洪水，导致周边农田被淹。针对此险情，研究人员利用Sentinel-1A雷达对该区域洪水淹没范围进行了快速监测。",
        """
        disaster_results = [Disaster(subject=Feature(name='鄱阳湖流域',geo_type='流域',geometry=Geometry('polygon')), 
                                predicate='发生', 
                                object=DisPhen(name='超警戒洪水',impact_type='水位上涨'),
                                observed_factors=[Environment_factor(name='洪水淹没范围')],
                                time_occurrence=datetime(2016, 5, 18, 0, 0)),
                            Disaster(subject=Feature(name='农田',geo_type='田地',geometry=Geometry('polygon')), 
                                predicate='发生', 
                                object=DisPhen(name='被淹',impact_type='侵入淹没'),
                                observed_factors=[Environment_factor(name='洪水淹没范围')],
                                time_occurrence=datetime(2016, 5, 18, 0, 0))
        ]

        observation_results = [Observation(subject=Sensor(name='Sentinel-1A雷达',sensor_type='微波传感器', sensor_platform='航天观测平台'), 
                                    predicate='观测', 
                                    object=[Feature(name='鄱阳湖流域',geo_type='流域',geometry=Geometry('polygon')), [Environment_factor(name='洪水淹没范围')]],
                                    observed_span=Geometry('polygon'),
                                    time_occurrence=datetime(2016, 5, 18, 0, 0))
        ]
        """
    ]

    chat_example_2 = [
        "德国南部内卡河等河流水位超限，导致斯图加特市遭遇洪水倒灌，城区积水深度达0.5米，电力供应中断，数百人被疏散。",
        """
        disaster_results = [Disaster(subject=Feature(name='内卡河',geo_type='河流',geometry=Geometry('linestring')), 
                                predicate='发生', 
                                object=DisPhen(name='水位超限',impact_type='水位上涨'),
                                observed_factors=[Feature_factor(name='水位')],
                                time_occurrence=None),
                            Disaster(subject = Feature(name='斯图加特市',geo_type='行政区',geometry=Geometry('polygon')), 
                                predicate='发生', 
                                object=DisPhen(name='洪水倒灌',impact_type='洪水倒灌'),
                                observed_factors=[Feature_factor(name='积水深度')],
                                time_occurrence=None),
                            Disaster(subject = Feature(name='斯图加特市',geo_type='行政区',geometry=Geometry('polygon')), 
                                predicate='发生', 
                                object=DisPhen(name='电力供应中断',impact_type='能源中断'),
                                observed_factors=[],
                                time_occurrence=None),
                            Disaster(subject = Feature(name='斯图加特市',geo_type='行政区',geometry=Geometry('polygon')), 
                                predicate='发生', 
                                object=DisPhen(name='数百人被疏散',impact_type='撤离疏散'),
                                observed_factors=[],
                                time_occurrence=None)
                            ]
        observation_results = []
        """
    ]

    chat_example_3 = [
        "研究团队利用GNSS接收机对白水河滑坡体进行了连续监测，结果显示该坡体体表位移累计已达200mm，处于不稳定状态。",
        """
        disaster_results = []
        observation_results = [Observation(
                                    subject=Sensor(name='GNSS接收机', sensor_type='原位传感器', sensor_platform='地面固定平台'), 
                                    predicate='观测', 
                                    object=[Feature(name='白水河滑坡体', geo_type='山体', geometry=Geometry('polygon')), [Feature_factor(name='坡体体表位移')]],
                                    observed_span=Geometry('point'),
                                    time_occurrence=None)
        ]
        """
    ]

    prompt_template = ChatPromptTemplate(
        [
            ("system", ontology_prompt),
            ("system", system_prompt),
            ("human", chat_example_1[0]),
            ("ai", chat_example_1[1]),
            ("human", chat_example_2[0]),
            ("ai", chat_example_2[1]),
            ("human", chat_example_3[0]),
            ("ai", chat_example_3[1]),
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


def disaster_event_exaction(text: str):
    """
    Extracts only disaster events from the given text.
    Utilizes an LLM to parse the text and instantiate a list of `Disaster` objects, detailing the
    spatial object, impact, observed factors ,and time occurrence.
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
        
        class DisPhen(Entity):
            # 灾害场景下地理对象上发生的灾害现象，变化表现为遭受非人为的灾害破坏，或可能引发次级灾害的地理现象
            # 示例:降水量增加、积水、水位上升、淹没、堵塞、崩塌
            def __init__(self, 
                name:str,   #文本中描述的灾害地理现象
                impact_type:str,   #所属灾害地理现象类型
            ):
                self.name = name
                self.impact_type = impact_type
                valid_types = ['大气对流', '极端降水', '风力增强', '水位上涨', '水流湍急', '水体漫溢', '泄洪分洪', '壅水顶托',
                    '水体倒灌', '排水不畅', '泥沙淤积', '水土流失', '堤坝决口', '洪水冲刷', '积水内涝', '侵入淹没', '淤积覆盖',
                    '水流冒溢', '结构崩塌', '溃决断裂', '滑坡', '泥石流', '山洪', '堰塞阻塞', '塌陷沉降', '交通中断',
                    '能源中断', '通信中断', '运行管制', '排涝失效', '人员伤亡', '人员受困', '撤离疏散', '资源告急',
                    '生产损失', '水质污染', '漂浮物堆积', '植物病虫害', '疫病扩散']
                if impact_type not in valid_types:
                    self.impact_type = ''
        
        class Observed_factor(Entity):
            # 可被传感器观测得到的属性因子，该因子受到灾害事件影响所变化，或其变化会促进灾害事件的发生
            pass
        
        class Feature_factor(Observed_factor):
            # 可被传感器观测的地理空间对象本身的状态属性因子
            # 示例:积水水位、河流流速、边坡应力、建筑结构稳定性、道路车流量
            pass
        
        class Environment_factor(Observed_factor):
            # 地理空间对象所处环境中存在的可被传感器观测的属性因子
            # 示例:降水量、风向、地形高程、温度、湿度
            pass
            
        class Event:
            # 所有事件的基类，事件必须的主体为[主语(subject),谓语(predicate),宾语(object)]
            def __init__(self, subject:str, predicate:str, object:str):
                self.subject = subject
                self.predicate = predicate
                self.object = object
        
        from datetime import datetime
        
        class Disaster(Event):
            # 灾害事件，事件主体定义为[某一地理空间对象(spatial_object),发生,某一现象(impact)]，事件论元包括若干可观测因子(observed_factors)和该灾害事件发生时间(time_occurrence)
            def __init__(
                self,
                spatial_object:Feature | Geometry,
                impact:DisPhen,
                observed_factors:List[Feature_factor | Environment_factor] = []
                time_occurrence:datetime = None
            ):
                super().__init__(subject=spatial_object, predicate='发生', object=impact)
                self.observed_factors = observed_factors
                self.time_occurrence = time_occurrence
    """

    system_prompt = ("这是一个面向对象的编程任务，上面定义了一些实体（Entity）类及事件（Event）类，请将下列文本转换为若干个Disaster对象实例，要求仅提取文本中明确提及的信息，不添加推测内容,"
                     "一定只能输出results变量内容，输出格式必须参照给出的案例。")

    chat_example_1 = [
        "main_text = '德国南部内卡河等河流水位超限，导致斯图加特市遭遇洪水倒灌，城区积水深度达0.5米，电力供应中断，数百人被疏散。'",
        """
            results = [Disaster(subject=Feature(name='内卡河',geo_type='河流',geometry=Geometry('linestring')), 
                            predicate='发生', 
                            object=DisPhen(name='水位超限',impact_type='水位上涨'),
                            observed_factors=[Feature_factor(name='水位')],
                            time_occurrence=None),
                      Disaster(subject = Feature(name='斯图加特市',geo_type='行政区',geometry=Geometry('polygon')), 
                            predicate='发生', 
                            object=DisPhen(name='洪水倒灌',impact_type='洪水倒灌'),
                            observed_factors=[Feature_factor(name='积水深度')],
                            time_occurrence=None),
                      Disaster(subject = Feature(name='斯图加特市',geo_type='行政区',geometry=Geometry('polygon')), 
                            predicate='发生', 
                            object=DisPhen(name='电力供应中断',impact_type='能源中断'),
                            observed_factors=[],
                            time_occurrence=None),
                      Disaster(subject = Feature(name='斯图加特市',geo_type='行政区',geometry=Geometry('polygon')), 
                            predicate='发生', 
                            object=DisPhen(name='数百人被疏散',impact_type='撤离疏散'),
                            observed_factors=[],
                            time_occurrence=None)
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


def observation_event_exaction(text: str):
    """
    Extracts only observation events from the given text.
    Utilizes an LLM to parse the text and instantiate a list of `Observation` objects, detailing the
    sensor, observed object, observed factors, observed span, and time occurrence.
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
            # 【重要正向约束】：name必须是具体的地理名称（如：海南岛）或地物统称（如：水库）。
            # 【重要负面约束】：
            # 1. 严禁将“灾害事件名称”作为Feature名称。例如：“洪涝灾害”、“地震”不是Feature，其发生的地点（如：长江）才是Feature。
            # 2. 严禁将“监测指标/因子”作为Feature名称。例如：“灾害范围”是因子而非地物。
            # 3. 严禁抽取任何可移动物体，如：列车、车辆、汽车、船舶、飞机、行人、动物等。
            # 4. 严禁抽取自然现象，如：洪水、积水、由于降雨形成的临时水坑等。
            def __init__(self, 
                name:str,   #地理空间要素实体名称，如武汉市、洞庭湖、南京路
                geo_type:str,   #地理空间要素实体类型，如城市、湖泊、道路
                geometry:Geometry   #地理空间要素空间范围形状类型
            ):
                self.name = name
                self.geo_type = geo_type
                self.geometry = geometry
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

        class Sensor(Entity):
            # 定义：执行地理环境观测任务的硬件设备或系统。
            # 【重要负面约束】：
            # 1. 严禁将“算法、数学模型、数据处理方法”识别为传感器。
            #    错误示例：'所优选的模型'、'时序分析法'、'深度学习算法'。
            # 2. 只有明确提到硬件（如：微波辐射计、SAR传感器、光学成像仪）时才抽取。
            def __init__(self, 
                name:str,   #文本中描述的传感器名称
                sensor_type:str,   #传感器类型
                sensor_platform:str : #传感器所属平台
            ):
                self.name = name

                self.sensor_type = sensor_type
                valid_types = ['原位传感器', '光学传感器', 'LiDAR传感器', '微波传感器']
                if sensor_type not in valid_types:
                    self.sensor_type = ''

                self.sensor_platform = sensor_platform
                valid_platforms = ['航天观测平台', '航空观测平台', '地面固定平台', '地面移动平台']
                if sensor_platform not in valid_platforms:
                    self.sensor_platform = ''

        class Observed_factor(Entity):
            # 定义：被传感器观测到的地理对象或环境的【特定属性、参数或状态】。
            # 【重要正向约束】：
            # 1. 必须是描述性的词汇，如“范围”、“深度”、“浓度”、“位移”、“植被指数”。
            # 2. 必须是可度量或可提取的状态。
            # 【重要负面约束】：
            # 1. 严禁将“灾害事件名称”识别为因子。
            #    错误示例：'洪涝灾害'、'森林火灾'、'地震'、'病虫害'。
            #    正确做法：从上述事件中提取属性，如 '洪水淹没范围'、'过火面积'、'地表形变量'。
            # 2. 如果文本只提到了监测灾害（如“监测洪涝灾害”），而没有提到具体指标，则 observed_factor 应为空。
            pass

        class Feature_factor(Observed_factor):
            # 可被传感器观测的地理空间对象本身的状态属性因子
            # 示例:积水水位、河流流速、边坡应力、建筑结构稳定性、道路车流量
            pass

        class Environment_factor(Observed_factor):
            # 地理空间对象所处环境中存在的可被传感器观测的属性因子
            # 示例:降水量、风向、地形高程、地表温度、空气湿度
            pass

        class Event:
            # 所有事件的基类，事件必须的主体为[主语(subject),谓语(predicate),宾语(object)]
            def __init__(self, subject:str, predicate:str, object:str):
                self.subject = subject
                self.predicate = predicate
                self.object = object

        from datetime import datetime

        class Observation(Event):
            # 观测事件，事件主体定义为[某一传感器(sensor),观测,某对象的某个观测因子([observed_object, observed_factors])]，事件论元包括观测范围(observed_span)和观测发生时间(time_occurrence)
            def __init__(
                self,
                sensor: Sensor
                observed_object:Feature | Geometry = None,
                observed_factors:List[Feature_factor | Environment_factor] = []
                observed_span:Geometry = None
                time_occurrence:datetime = None
            ):
                super().__init__(subject=sensor, predicate='观测', object=[observed_object, observed_factors])
                self.observed_span = observed_span
                self.time_occurrence = time_occurrence
    """

    system_prompt = ("这是一个面向对象的编程任务，上面定义了一些实体（Entity）类及事件（Event）类，请将下列文本转换为若干个Observation对象实例，要求仅提取文本中明确提及的信息，不添加推测内容,"
                     "一定只能输出results变量内容，输出格式必须参照给出的案例。")

    chat_example_1 = [
        "基于Sentinel-1A雷达成像系统,对2016年5月18日长江中下游流域洪水灾害最为严重的区域进行了时序动态监测,快速提取了洪水淹没范围",
        """
            results = [Observation(subject=Sensor(name='Sentinel-1A雷达成像系统',sensor_type='微波传感器', sensor_platform='航天观测平台'), 
                            predicate='观测', 
                            object=[Feature(name='长江中下游流域',geo_type='流域',geometry=Geometry('polygon'), Environment_factor(name='洪水淹没范围')],
                            observed_span=Geometry('polygon'),
                            time_occurrence=datetime(2016, 5, 18, 0, 0))
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


def parse_disaster_results(results_raw):
    """
    Parses the raw string output from the LLM containing Disaster objects using regular expressions.
    It maps the extracted strings into a structured list of dictionaries representing disaster events.
    """

    results_raw = re.sub(r"```python|```", "", results_raw).strip()

    disaster_pattern = re.compile(
        r"Disaster\s*\(\s*"
        r"(?:(?:spatial_object|subject)\s*=\s*)?Feature\s*\(\s*(?:name\s*=\s*)?'(?P<name>[^']+)'\s*,\s*(?:geo_type\s*=\s*)?'(?P<geo_type>[^']+)'\s*,\s*(?:geometry\s*=\s*)?Geometry\s*\(\s*'(?P<geometry>[^']+)'\s*\)\s*\)\s*,\s*"

        r"(?:predicate\s*=\s*'(?P<predicate>[^']+)'\s*,\s*)?"

        r"(?:(?:impact|object)\s*=\s*)?(?:DisPhen|Change)\s*\(\s*(?:name\s*=\s*)?'(?P<obj_name>[^']+)'\s*,\s*(?:impact_type\s*=\s*)?'(?P<obj_type>[^']+)'\s*\)\s*,\s*"
        
        r"(?:observed_factors\s*=\s*)?\[(?P<factors>.*?)\]\s*,\s*"

        r"(?:time_occurrence\s*=\s*)?(?P<time>datetime\(.*?\)|None|'[^']+')\s*"
        r"\)",
        re.DOTALL
    )

    disasters = []

    for match in disaster_pattern.finditer(results_raw):
        res = match.groupdict()

        predicate = res.get('predicate') or "happen"

        factors_raw = res.get('factors', "")
        feature_factors = re.findall(r"Feature_factor\s*\(\s*(?:name\s*=\s*)?'([^']+)'\s*\)", factors_raw)
        env_factors = re.findall(r"Environment_factor\s*\(\s*(?:name\s*=\s*)?'([^']+)'\s*\)", factors_raw)

        time_raw = res.get('time', "None")
        time_occ = None if time_raw == "None" else time_raw.strip("'")

        disasters.append({
            "feature": {
                "name": res['name'],
                "geo_type": res['geo_type'],
                "geometry": res['geometry']
            },
            "predicate": predicate,
            "impact": {
                "name": res['obj_name'],
                "impact_type": res['obj_type']
            },
            "factors": {
                "feature_factors": feature_factors,
                "environment_factors": env_factors
            },
            "time": time_occ
        })

    return disasters


def preprocess_observation_data(raw_str: str) -> str:
    """
    Cleans and preprocesses the raw string output from the LLM prior to evaluation.
    It removes markdown code block markers, strips out the disaster_results portion,
    and renames the observation_results variable to 'results' for safe parsing.
    """

    if not raw_str:
        return ""

    cleaned_str = raw_str.replace('\xa0', ' ')
    cleaned_str = re.sub(r'```python\s*|```', '', cleaned_str)

    cleaned_str = re.sub(r'disaster_results\s*=\s*\[.*?\]', '', cleaned_str, flags=re.DOTALL)

    cleaned_str = cleaned_str.replace('observation_results', 'results')

    return cleaned_str.strip()


def parse_observation_results(results_raw):
    """
    Parses the preprocessed string containing Observation objects into a structured list of dictionaries.
    It sets up a restricted sandbox environment and safely evaluates the string using `eval()`.
    """

    results_raw = preprocess_observation_data(results_raw)

    class Geometry:
        def __init__(self, name): self.name = name

        def to_dict(self): return self.name

    class Feature:
        def __init__(self, name, geo_type, geometry):
            self.name, self.geo_type, self.geometry = name, geo_type, geometry

        def to_dict(self):
            return {"name": self.name, "geo_type": self.geo_type, "geometry": self.geometry.to_dict()}

    class Sensor:
        def __init__(self, name, sensor_type='', sensor_platform=''):
            self.name, self.sensor_type, self.sensor_platform = name, sensor_type, sensor_platform

        def to_dict(self):
            return {"name": self.name, "type": self.sensor_type, "platform": self.sensor_platform}

    class Feature_factor:
        def __init__(self, name): self.name = name

        def to_dict(self): return self.name

    class Environment_factor:
        def __init__(self, name): self.name = name

        def to_dict(self): return self.name

    def custom_datetime_formatter(*args):
        return f"datetime{args}"

    class Observation:
        def __init__(self, subject=None, sensor=None, predicate='观测', object=None, observed_span=None,
                     time_occurrence=None):
            self.sensor = subject if subject else sensor
            self.predicate = predicate

            self.observed_object = None
            self.factors = []

            if isinstance(object, list) and len(object) > 0:
                self.observed_object = object[0]

                if len(object) > 1:
                    raw_factors = object[1:] if len(object) > 2 else [object[1]]

                    for item in raw_factors:
                        if isinstance(item, list):
                            self.factors.extend(item)
                        else:
                            self.factors.append(item)
            # -----------------------

            self.observed_span = observed_span
            self.time_occurrence = time_occurrence

        def to_dict(self):
            def safe_to_dict(obj):
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                return str(obj) if obj is not None else None

            f_factors = [safe_to_dict(f) for f in self.factors if isinstance(f, Feature_factor)]
            e_factors = [safe_to_dict(e) for e in self.factors if isinstance(e, Environment_factor)]

            return {
                "sensor": safe_to_dict(self.sensor),
                "predicate": self.predicate,
                "observed_object": safe_to_dict(self.observed_object),
                "factors": {
                    "feature_factors": f_factors,
                    "environment_factors": e_factors
                },
                "observed_span": safe_to_dict(self.observed_span),
                "time": self.time_occurrence
            }

    clean_code = re.sub(r"```python|```", "", results_raw).strip()
    if "results =" in clean_code:
        clean_code = clean_code.split("results =", 1)[1].strip()

    safe_env = {
        'Observation': Observation,
        'Sensor': Sensor,
        'Feature': Feature,
        'Geometry': Geometry,
        'Feature_factor': Feature_factor,
        'Environment_factor': Environment_factor,
        'datetime': custom_datetime_formatter,
        'None': None
    }

    try:
        parsed_data = eval(clean_code, {"__builtins__": None}, safe_env)
        if isinstance(parsed_data, list):
            return [obs.to_dict() for obs in parsed_data]
        return []
    except Exception as e:
        print(f"解析出错: {e}")
        return []


def parse_disaster_cause_safe(llm_response: str):
    """
    Safely parses the LLM output containing cause relationships between disaster events[cite: 17].
    """

    class Disaster:
        def __init__(self, subject: str, predicate: str, object: str):
            self.subject, self.predicate, self.object = subject, predicate, object

    class CauseRelation:
        def __init__(self, head_disaster, tail_disaster, judgment_evidence):
            self.head_disaster, self.tail_disaster, self.judgment_evidence = head_disaster, tail_disaster, judgment_evidence

    cleaned_code = re.sub(r"```python|```", "", llm_response).strip()
    cleaned_code = cleaned_code.replace('（', '(').replace('）', ')').replace('。', '')

    local_scope = {'Disaster': Disaster, 'CauseRelation': CauseRelation}
    relations = []
    try:
        exec(cleaned_code, globals(), local_scope)
        for key in ['results', 'result', 'res']:
            if key in local_scope and isinstance(local_scope[key], list):
                relations = local_scope[key]
                break
    except Exception:
        pass

    return relations


def parse_disaster_trigger_safe(llm_response: str):
    """
    Safely parses the LLM output containing trigger relationships between disasters and observations[cite: 17].
    """

    class Disaster:
        def __init__(self, subject=None, predicate='发生', object=None):
            self.subject, self.predicate, self.object = subject, predicate, object

        def to_dict(self): return {"type": "Disaster", "subject": self.subject, "predicate": self.predicate,
                                   "object": self.object}

    class Observation:
        def __init__(self, subject=None, predicate='观测', object=None):
            self.subject, self.predicate, self.object = subject, predicate, object

        def to_dict(self): return {"type": "Observation", "subject": self.subject, "predicate": self.predicate,
                                   "object": self.object}

    class TriggerRelation:
        def __init__(self, head_disaster=None, tail_disaster=None, judgment_evidence=""):
            self.head_disaster, self.tail_disaster, self.judgment_evidence = head_disaster, tail_disaster, judgment_evidence

        def to_dict(self):
            return {
                "head": self.head_disaster.to_dict() if hasattr(self.head_disaster, 'to_dict') else str(
                    self.head_disaster),
                "tail": self.tail_disaster.to_dict() if hasattr(self.tail_disaster, 'to_dict') else str(
                    self.tail_disaster),
                "evidence": self.judgment_evidence
            }

    cleaned_code = re.sub(r"```python|```", "", llm_response).strip()
    cleaned_code = cleaned_code.replace('（', '(').replace('）', ')').replace('。', '')

    safe_env = {'TriggerRelation': TriggerRelation, 'Disaster': Disaster, 'Observation': Observation, 'None': None}
    local_scope = {}
    parsed_results = []
    try:
        exec(cleaned_code, safe_env, local_scope)
        target_list = None
        for key in ['result', 'results', 'res']:
            if key in local_scope and isinstance(local_scope[key], list):
                target_list = local_scope[key]
                break
        if target_list:
            parsed_results = [item.to_dict() for item in target_list if hasattr(item, 'to_dict')]
    except Exception:
        pass

    return parsed_results


def _worker_cause_relation(text, events_in, events_dict_in):
    """Worker to call LLM for causal relations[cite: 16]."""
    ontology_prompt = """
            class Event:
                def __init__(self, subject:str, predicate:str, object:str):
                    self.subject = subject
                    self.predicate = predicate
                    self.object = object
            class Disaster(Event):
                def __init__(self, spatial_object: str, impact: str):
                    super().__init__(subject=spatial_object, predicate='发生', object=impact)
            class Relation:
                def __init__(self, head_entity, tail_entity):
                    self.head_entity = head_entity
                    self.tail_entity = tail_entity
            class CauseRelation(Relation):
                def __init__(self, head_disaster: Disaster, tail_disaster: Disaster, judgment_evidence: str):
                    super().__init__(head_disaster, tail_disaster)
                    self.judgment_evidence = judgment_evidence
        """
    system_prompt = ("这是一个面向对象的编程任务，上面定义了事件（Event）及关系（Relation）类，请实例化所有CauseRelation对象。"
                     "必须且仅能使用输入`disasters`列表中提供的事件对象。阅读文本，判别显性或隐性因果关系，生成判定依据。仅输出results变量。")
    prompt_template = ChatPromptTemplate([
        ("system", ontology_prompt),
        ("system", system_prompt),
        ("human", "text = {news_text},\ndisasters = {news_disasters}")
    ])
    chain = prompt_template | llm

    try:
        output = chain.invoke({'news_text': text, 'news_disasters': events_in})
        parsed_relations = parse_disaster_cause_safe(output.content)

        batch_results = []
        for rel in parsed_relations:
            head_key = f"Disaster(subject='{rel.head_disaster.subject}', predicate='发生', object='{rel.head_disaster.object}')"
            tail_key = f"Disaster(subject='{rel.tail_disaster.subject}', predicate='发生', object='{rel.tail_disaster.object}')"
            batch_results.append({
                'head_id': events_dict_in.get(head_key),
                'tail_id': events_dict_in.get(tail_key),
                'evidence': getattr(rel, 'judgment_evidence', ''),
                'head_key': head_key,
                'tail_key': tail_key
            })
        return batch_results
    except Exception:
        return []


def _worker_trigger_relation(text, disaster_events, obs_events, events_dict):
    """Worker to call LLM for trigger relations[cite: 17]."""
    ontology_prompt = """
            class Event:
                def __init__(self, subject:str, predicate:str, object:str):
                    self.subject = subject
                    self.predicate = predicate
                    self.object = object
            class Disaster(Event):
                def __init__(self, spatial_object: str, impact: str):
                    super().__init__(subject=spatial_object, predicate='发生', object=impact)
            class Observation(Event):
                def __init__(self, sensor: str, observed_object:str, observed_factors:str):
                    super().__init__(subject=sensor, predicate='观测', object=[observed_object, observed_factors])
            class Relation:
                def __init__(self, head_entity, tail_entity):
                    self.head_entity = head_entity
                    self.tail_entity = tail_entity
            class TriggerRelation(Relation):
                def __init__(self, head_disaster: Disaster, tail_disaster: Observation, judgment_evidence: str):
                    super().__init__(head_disaster, tail_disaster)
                    self.judgment_evidence = judgment_evidence
        """
    system_prompt = ("这是一个面向对象的编程任务，请实例化所有TriggerRelation对象。"
                     "必须且仅能使用输入提供的事件对象。判别灾害触发观测任务的因果逻辑，生成判定依据。仅输出results变量。")
    prompt_template = ChatPromptTemplate([
        ("system", ontology_prompt),
        ("system", system_prompt),
        ("human", "text = {text},\ndisasters = {disasters}\nobservations = {observations}")
    ])
    chain = prompt_template | llm

    try:
        output = chain.invoke({'text': text, 'disasters': disaster_events, 'observations': obs_events})
        relations = parse_disaster_trigger_safe(output.content)

        batch_results = []
        for relation in relations:
            head = relation.get('head')
            tail = relation.get('tail')
            if not head or not tail or head.get('type') != 'Disaster' or tail.get('type') != 'Observation':
                continue

            head_key = f"Disaster(subject='{head.get('subject')}', predicate='发生', object='{head.get('object')}')"
            obs_obj = tail.get('object', [])
            geo_name = obs_obj[0] if isinstance(obs_obj, list) else ''
            factor_name = obs_obj[1] if isinstance(obs_obj, list) and len(obs_obj) > 1 else ''
            tail_key = f"Observation(subject='{tail.get('subject')}', predicate='观测', object=[{geo_name}, {factor_name}])"

            batch_results.append({
                'head_id': events_dict.get(head_key),
                'tail_id': events_dict.get(tail_key),
                'evidence': relation.get('evidence', '')
            })
        return batch_results
    except Exception:
        return []


def update_DO_argument_relations(disaster_node: DisasterEvent, observation_node: ObservationEvent):
    """
    Updates the specific argument relationships (e.g., associated geographic objects and observed factors)
    between a disaster node and an observation node in the Neo4j graph database.
    """

    d_geo_nodes = disaster_node.geo_argument.all()
    d_geo_node = None
    if d_geo_nodes:
        d_geo_node = d_geo_nodes[0]

    o_geo_nodes = observation_node.geo_argument.all()
    o_geo_node = None
    if o_geo_nodes:
        o_geo_node = o_geo_nodes[0]

    if o_geo_node and d_geo_node:
        if o_geo_node.node_id != d_geo_node.node_id:
            if not d_geo_node.associate.is_connected(o_geo_node):
                r = d_geo_node.associate.connect(o_geo_node)
                print(f"associate:{d_geo_node.geo_object_name} - {o_geo_node.geo_object_name}")

    o_factor_nodes = observation_node.factor_argument.all()
    for o_factor_node in o_factor_nodes:
        if d_geo_node:
            if o_factor_node.factor_type == "environment_factors":
                continue
            if not d_geo_node.has_factor.is_connected(o_factor_node):
                r = d_geo_node.has_factor.connect(o_factor_node)
                print(f"hasFactor: {d_geo_node.geo_object_name} - {o_factor_node.factor_name}")
        if not disaster_node.observed.is_connected(o_factor_node):
            r = disaster_node.observed.connect(o_factor_node)
            print(f"observed: {disaster_node.event_description} - {o_factor_node.factor_name}")


def update_D_argument_relations(disaster_node: DisasterEvent):
    """
    Updates the intrinsic argument relationships for a single disaster node, specifically linking
    the disaster's geographic object to its relevant feature factors.
    """

    d_geo_nodes = disaster_node.geo_argument.all()
    d_geo_node = None
    if d_geo_nodes:
        d_geo_node = d_geo_nodes[0]

    d_factor_nodes = disaster_node.observed.all()
    for d_factor_node in d_factor_nodes:
        if d_geo_node:
            if d_factor_node.factor_type == "environment_factors":
                continue
            if not d_geo_node.has_factor.is_connected(d_factor_node):
                r = d_geo_node.has_factor.connect(d_factor_node)
                print(f"hasFactor: {d_geo_node.geo_object_name} - {d_factor_node.factor_name}")


def main_update_DO_argument_relations():
    """
    Acts as the main execution loop to batch update all argument relations. It iterates through all
    disaster nodes and their connected observation nodes (via trigger relations) to enforce schema consistency.
    """

    disaster_nodes = DisasterEvent.nodes.all()
    for disaster_node in disaster_nodes:
        observation_nodes = disaster_node.trigger.all()
        for observation_node in observation_nodes:
            update_DO_argument_relations(disaster_node, observation_node)
        update_D_argument_relations(disaster_node)


if __name__ == '__main__':
    file_path = 'data/all_entity_relation_extraction.json'

    data_list = []

    # 2. 遍历列表，获取每个条目的 text 字段
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"--- 开始逐行读取文件：{file_path} ---")
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                # 每一行解析为一个独立的 JSON 对象
                item = json.loads(line)
                data_list.append(item)

                # 输出 text 字段
                text_content = item.get('text', '无文本内容')
                # print(f"数据条目 {line_number}:")
                # print(text_content)
                # print("-" * 30)
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_number} 行解析失败，已跳过。错误原因: {e}")

    count = 0
    for text in data_list:
        count += 1
        raw = observation_event_exaction(text)
        print(raw)
        parse = parse_observation_results(raw)
        print(parse)
        if count > 3:
            break
