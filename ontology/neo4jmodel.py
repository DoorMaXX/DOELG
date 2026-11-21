from neomodel import (
    config, db as neo4j_db,
    StructuredNode, StructuredRel,
    StringProperty, IntegerProperty, UniqueIdProperty, BooleanProperty,
    RelationshipTo, FloatProperty, ArrayProperty, JSONProperty
)
from src.construct_config import get_setting


# ========== Database Connection ==========
class Neo4jDatabaseConnector:
    DATABASE_URL = get_setting('neo4j.url')

    def __init__(self, database_name=None):
        """Connect to the selected database using the configured Neo4j server URL."""
        database_name = database_name if database_name is not None else get_setting('neo4j.default_database')
        self.DATABASE_URL = get_setting('neo4j.url')
        config.DATABASE_URL = self.DATABASE_URL
        self.db = neo4j_db
        self.database_name = database_name
        self.connect_database(database_name)

    def connect_database(self, database_name=None):
        """Switch the active database, falling back to the configured default when omitted."""
        database_name = database_name if database_name is not None else get_setting('neo4j.default_database')
        self.db.set_connection(self.DATABASE_URL + f"/{database_name}")


# ========== Relation Definition ==========
class SpatialRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="Spatial")
    relation_name = StringProperty(default="")


class HappenRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="Happen")
    relation_name = StringProperty(default="Happen")


class EventArgumentRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="Argument")
    relation_name = StringProperty(default="Argument")
    score = FloatProperty()


class FromRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="From")
    relation_name = StringProperty(default="From")


class ExistRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="Exist")
    relation_name = StringProperty(default="Exist")


class CauseRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="Cause")
    relation_name = StringProperty(default="Cause")
    affectType = StringProperty()
    matrix_relation_code = StringProperty()
    matrix_review_status = StringProperty()
    relation_count = IntegerProperty(default=1)
    evidence = StringProperty()
    cluster_pattern = StringProperty()
    confidence = FloatProperty(default=0)
    lift = FloatProperty(default=0)
    probability = FloatProperty(default=0)
    path_redundancy = FloatProperty(default=0)


class ObservedRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="Observed")
    relation_name = StringProperty(default="Observed Factor")


class OccurredAtRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="OccurredAt")
    relation_name = StringProperty(default="Occurred At")


class AlignedRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="Alignment")
    relation_name = StringProperty(default="Alignment")


class HasGeometryRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="HasGeometry")
    relation_name = StringProperty(default="Geometry")


class IncludeRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="Include")
    relation_name = StringProperty(default="Include")


class AssociateRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="Associate")
    relation_name = StringProperty(default="Associate")


class AssociateFactorRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="Associate")
    relation_name = StringProperty(default="Associate")

    type = StringProperty()
    score = FloatProperty(default=0)
    evidence = StringProperty()


class DisasterSpatialRelation(StructuredRel):
    rel_id = UniqueIdProperty()

    relations = JSONProperty(default={})


# ========== Node Definition ==========
class OSMFeature(StructuredNode):
    node_id = UniqueIdProperty()
    osm_id = StringProperty(unique_index=True)
    event_id = StringProperty()

    properties = JSONProperty(default={})
    importance = FloatProperty(default=0)
    priority = FloatProperty(default=999)
    topology = StringProperty()

    has_geometry = RelationshipTo("Geometry", "HASGEOMETRY", model=HasGeometryRelation)


class GeoObject(StructuredNode):
    node_id = UniqueIdProperty()
    geo_object_name = StringProperty(required=True)
    geo_type = StringProperty(required=False)
    geometry = StringProperty(required=False)
    is_location = BooleanProperty(required=False)

    happen = RelationshipTo("Impact", "HAPPEN", model=HappenRelation)
    origin = RelationshipTo("OriginText", "FROM", model=FromRelation)
    spatial = RelationshipTo("GeoObject", "SPATIAL", model=SpatialRelation)
    associate = RelationshipTo("GeoObject", "ASSOCIATE", model=AssociateRelation)
    has_geometry = RelationshipTo("Geometry", "HASGEOMETRY", model=HasGeometryRelation)
    has_factor = RelationshipTo("Factor", "HASFACTOR", model=ObservedRelation)

    aligned_to = RelationshipTo("GeoObject", "ALIGNED_TO", model=AlignedRelation)

    event_id = StringProperty()
    has_Feature = RelationshipTo("OSMFeature", "HAS_FEATURE", model=IncludeRelation)


class Geometry(StructuredNode):
    node_id = UniqueIdProperty()
    location = ArrayProperty(FloatProperty(), default=[])
    osm_id = StringProperty()

    geometry = StringProperty()


class DisPhen(StructuredNode):
    node_id = UniqueIdProperty()
    impact_name = StringProperty(required=True)
    impact_type = StringProperty(required=False)


class Impact(StructuredNode):
    node_id = UniqueIdProperty()
    impact_name = StringProperty(required=True)
    impact_type = StringProperty(required=False)


class Factor(StructuredNode):
    node_id = UniqueIdProperty()
    factor_type = StringProperty(required=True)
    factor_name = StringProperty(required=True)

    relate = RelationshipTo("StandardFactor", "MAPPING", model=AssociateFactorRelation)


class StandardFactor(StructuredNode):
    node_id = UniqueIdProperty()
    factor_name = StringProperty(required=True)
    relate_geos = ArrayProperty(default=[])


class Time(StructuredNode):
    node_id = UniqueIdProperty()
    time_value = StringProperty(required=True)


class EnvironmentDescription(StructuredNode):
    node_id = UniqueIdProperty()
    description = StringProperty()


class EnvironmentRule(StructuredNode):
    node_id = UniqueIdProperty()
    rules = JSONProperty(default={})


class DisasterEvent(StructuredNode):
    node_id = UniqueIdProperty()
    event_num = IntegerProperty()
    event_description = StringProperty()
    environment_description = StringProperty()

    depth = IntegerProperty()
    ori_event_desc = StringProperty()
    is_happen = BooleanProperty(default=False)
    parent_path = ArrayProperty(default=[])
    probability = FloatProperty(default=1)

    geo_argument = RelationshipTo("GeoObject", "ARGUMENT", model=EventArgumentRelation)
    impact_argument = RelationshipTo("Impact", "ARGUMENT", model=EventArgumentRelation)
    envDes_argument = RelationshipTo("EnvironmentDescription", "ARGUMENT", model=EventArgumentRelation)
    envRule_argument = RelationshipTo("EnvironmentRule", "ARGUMENT", model=EventArgumentRelation)
    origin = RelationshipTo("OriginText", "FROM", model=FromRelation)
    cause = RelationshipTo("DisasterEvent", "CAUSE", model=CauseRelation)
    occurred_at = RelationshipTo("Time", "OCCURRED_AT", model=OccurredAtRelation)
    observed = RelationshipTo("Factor", "OBSERVED", model=ObservedRelation)
    spatial_to = RelationshipTo("DisasterEvent", "SPATIAL_TO", model=DisasterSpatialRelation)
    trigger = RelationshipTo("ObservationEvent", "TRIGGER", model=CauseRelation)


class OriginText(StructuredNode):
    node_id = UniqueIdProperty()
    text_content = StringProperty(required=True)
    text_title = StringProperty()
    exist = RelationshipTo("DisasterEvent", "EXIST", model=ExistRelation)
    exist_obs = RelationshipTo("ObservationEvent", "EXIST", model=ExistRelation)
    hasGeo = RelationshipTo("GeoObject", "EXIST", model=ExistRelation)


class GlobalCauseRelation(StructuredRel):
    rel_id = UniqueIdProperty()
    relation_type = StringProperty(default="GlobalCause")
    relation_name = StringProperty(default="Cause")
    cluster_pattern = StringProperty()


class MergeDisasterEvent(StructuredNode):
    node_id = UniqueIdProperty()
    event_description = StringProperty()
    cluster_pattern = StringProperty()

    envDes_argument = RelationshipTo("EnvironmentDescription", "ARGUMENT", model=EventArgumentRelation)
    envRule_argument = RelationshipTo("EnvironmentRule", "ARGUMENT", model=EventArgumentRelation)
    include = RelationshipTo("DisasterEvent", "INCLUDE", model=IncludeRelation)
    cause = RelationshipTo("MergeDisasterEvent", "CAUSE", model=CauseRelation)
    spatial_to = RelationshipTo("MergeDisasterEvent", "SPATIAL_TO", model=DisasterSpatialRelation)
    trigger = RelationshipTo("MergeObservationEvent", "TRIGGER", model=CauseRelation)


class GlobalDisasterEvent(StructuredNode):
    node_id = UniqueIdProperty()
    event_description = StringProperty()

    geo_argument = RelationshipTo("GeoObject", "ARGUMENT", model=EventArgumentRelation)
    impact_argument = RelationshipTo("Impact", "ARGUMENT", model=EventArgumentRelation)
    include = RelationshipTo("MergeDisasterEvent", "INCLUDE", model=IncludeRelation)
    cause = RelationshipTo("GlobalDisasterEvent", "CAUSE", model=CauseRelation)
    observed = RelationshipTo("Factor", "OBSERVED", model=ObservedRelation)
    trigger = RelationshipTo("GeneralizedObservationEvent", "TRIGGER", model=CauseRelation)


class Sensor(StructuredNode):
    node_id = UniqueIdProperty()
    sensor_name = StringProperty(required=True)  # 对应instrument_name
    sensor_type = StringProperty(required=True)  # ['原位传感器', '光学传感器', 'LiDAR传感器', '微波传感器']其中之一
    platform = StringProperty(required=True)  # ['航天观测平台', '航空观测平台', '地面固定平台', '地面移动平台']其中之一
    space_res = FloatProperty()  # 对应best_resolution
    time_res = StringProperty()  # 对应coverage_cycle，最好可以计算出一个标准数值
    weather_effect = StringProperty()  # 用大模型判断或者自己根据传感器类型或者光谱范围设定下规则
    spectral_range = ArrayProperty(StringProperty(), required=False, default=[])  # 对应spectral_range

    observe_geo = RelationshipTo("GeoObject", "OBSERVED", model=ObservedRelation)  # 不用考虑
    observe_factor = RelationshipTo("Factor", "OBSERVED",
                                    model=ObservedRelation)  # 对应primary_mission_objectives和secondary_mission_objectives，需要转成中文，‘Factor’节点factor_type为空字符串，每个指标一个节点
    belong_platform = RelationshipTo("Platform", "BELONG", model=FromRelation)  # 对应satellite_name，每个卫星一个节点
    detail_type = RelationshipTo("Type_of_sensor", "TYPE", model=FromRelation)  # 对应type_of_sensor，需要转成中文


class Platform(StructuredNode):
    node_id = UniqueIdProperty()
    platform_name = StringProperty(required=True)


class Type_of_sensor(StructuredNode):
    node_id = UniqueIdProperty()
    type = StringProperty()


class ObservationPattern(StructuredNode):
    node_id = UniqueIdProperty()
    space_res = FloatProperty(default=-1)
    time_res = FloatProperty(default=-1)
    obs_position = StringProperty(default='')
    weather_effect = StringProperty(default='')
    disaster_ids = ArrayProperty(StringProperty(), default=[])


class GeneralizedObservationPattern(StructuredNode):
    node_id = UniqueIdProperty()
    space_res = StringProperty(default='High Resolution')
    time_res = StringProperty(default='High Frequency')
    obs_position = StringProperty(default='')
    weather_effect = StringProperty(default='')
    disaster_ids = ArrayProperty(StringProperty(), default=[])

    include = RelationshipTo("ObservationPattern", "INCLUDE", model=IncludeRelation)


class ObservationEvent(StructuredNode):
    node_id = UniqueIdProperty()
    event_num = IntegerProperty()
    event_description = StringProperty()
    observe_pattern = StringProperty()
    observe_span = StringProperty()

    disaster_id = StringProperty()

    sensor_argument = RelationshipTo("Sensor", "ARGUMENT", model=EventArgumentRelation)
    ins_sensor_argument = RelationshipTo("Sensor", "INS_SENSOR", model=EventArgumentRelation)
    geo_argument = RelationshipTo("GeoObject", "ARGUMENT", model=EventArgumentRelation)
    factor_argument = RelationshipTo("Factor", "OBSERVED", model=ObservedRelation)
    obsPat_argument = RelationshipTo("ObservationPattern", "ARGUMENT", model=EventArgumentRelation)
    gen_obsPat_argument = RelationshipTo("GeneralizedObservationPattern", "ARGUMENT", model=EventArgumentRelation)
    origin = RelationshipTo("OriginText", "FROM", model=FromRelation)
    occurred_at = RelationshipTo("Time", "OCCURRED_AT", model=OccurredAtRelation)


class MergeObservationEvent(StructuredNode):
    node_id = UniqueIdProperty()
    event_description = StringProperty()
    cluster_pattern = StringProperty()

    obsPat_argument = RelationshipTo("ObservationPattern", "ARGUMENT", model=EventArgumentRelation)
    include = RelationshipTo("ObservationEvent", "INCLUDE", model=IncludeRelation)


class GlobalObservationEvent(StructuredNode):
    node_id = UniqueIdProperty()
    event_description = StringProperty()

    sensor_argument = RelationshipTo("Sensor", "ARGUMENT", model=EventArgumentRelation)
    geo_argument = RelationshipTo("GeoObject", "ARGUMENT", model=EventArgumentRelation)
    factor_argument = RelationshipTo("Factor", "OBSERVED", model=ObservedRelation)
    obsPat_argument = RelationshipTo("ObservationPattern", "ARGUMENT", model=EventArgumentRelation)

    include = RelationshipTo("MergeObservationEvent", "INCLUDE", model=IncludeRelation)


class GeneralizedObservationEvent(StructuredNode):
    node_id = UniqueIdProperty()
    event_description = StringProperty()
    is_public = BooleanProperty(default=False)

    sensor_argument = RelationshipTo("Sensor", "ARGUMENT", model=EventArgumentRelation)
    geo_argument = RelationshipTo("GeoObject", "ARGUMENT", model=EventArgumentRelation)
    factor_argument = RelationshipTo("Factor", "OBSERVED", model=ObservedRelation)
    std_factor_argument = RelationshipTo("StandardFactor", "OBSERVED", model=ObservedRelation)
    obsPat_argument = RelationshipTo("GeneralizedObservationPattern", "ARGUMENT", model=EventArgumentRelation)

    include = RelationshipTo("ObservationEvent", "INCLUDE", model=IncludeRelation)
