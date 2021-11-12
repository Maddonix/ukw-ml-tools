# Interventions
FIELDNAME_FRAMES = "frames"
FIELDNAME_VIDEO_PATH = "video_path"
FIELDNAME_TOKENS = "tokens"
FIELDNAME_TOKENS_PATHO = "patho"
FIELDNAME_TOKENS_REPORT = "report"
FIELDNAME_EXTERNAL_ID = "_id_michael"
FIELDNAME_LABELS_FRAMES = "labels_frames"
FIELDNAME_INTERVENTION_TYPE = "intervention_type"
FIELDNAME_FREEZES = "freezes"
FIELDNAME_INTERVENTION_DATE = "intervention_date"
FIELDNAME_VIDEO_METADATA = "metadata"
FIELDNAME_FPS = "fps"
FIELDNAME_FRAMES_TOTAL = "frames_total"
FIELDNAME_PATHO_RAW = "patho_raw"
FIELDNAME_REPORT_RAW = "report_raw"

# Images
FIELDNAME_INTERVENTION_ID = "intervention_id"
FIELDNAME_LABELS = "labels_new"
FIELDNAME_IMAGE_PATH = "path"
FIELDNAME_FRAME_NUMBER = "n"
FIELDNAME_LABELS_VALIDATED = "labels_validated"
FIELDNAME_LABELS_UNCLEAR = "labels_unclear"
FIELDNAME_IMAGE_TYPE = "image_type"
LABEL_UNCLEAR = "unclear"

# Shared
FIELDNAME_ORIGIN = "origin"
FIELDNAME_VIDEO_KEY = "video_key"

# AI
FIELDNAME_AI_VERSION = "version"
FIELDNAME_IMAGE_SCALING = "image_scaling"
FIELDNAME_PREDICTION_VALUE = "value"
FIELDNAME_PREDICTIONS = "predictions"
FIELDNAME_PREDICTION_LABEL = "label"

# OTHER STRING CONSTANTS
IMAGETYPE_FRAME = "frame"
IMAGETYPE_FREEZE = "freeze"

# TERMINOLOGY
URL_POST_ONTOLOGY = "/PostOntology"
URL_TEXT_TO_TOKEN = "/PostTerminologyTokens"
URL_TEXT_TO_XML = "/PostTerminologyXml"

# PREDICTION_RESULT_DF
COLNAME_AI_NAME = "ai_name"


# CFG

# STATS
"""
Components in the name are not allowed to contain . or ,
Stats Result names for dict are defined as:
{db}.{value_type}.{att1}.{value_1}.{att2}.{value2}.....

if an attribute has more than one value, they are to be comma separated

"""

PREFIX_COUNT = "count"
PREFIX_INTERVENTION = "interventions"
PREFIX_EXISTS = "exists"

DF_COL_ENTITY = "entity"
DF_COL_VALUE = "value"
DF_COL_VALUE_TYPE = "value_type"
DF_COL_DATE = "date"
