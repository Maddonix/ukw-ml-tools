# Interventions
FIELDNAME_FRAMES = "frames"
FIELDNAME_VIDEO_PATH = "video_path"
FIELDNAME_TOKENS = "tokens"
FIELDNAME_TOKENS_PATHO = "patho"
FIELDNAME_TOKENS_REPORT = "report"
FIELDNAME_TOKENS_VALUE = "value"
FIELDNAME_EXTERNAL_ID = "_id_michael"
FIELDNAME_INTERVENTION_IMAGE_ANNOTATIONS = "image_annotations"
FIELDNAME_INTERVENTION_TYPE = "intervention_type"
FIELDNAME_FREEZES = "freezes"
FIELDNAME_INTERVENTION_DATE = "intervention_date"
FIELDNAME_VIDEO_METADATA = "metadata"
FIELDNAME_FPS = "fps"
FIELDNAME_FRAMES_TOTAL = "frames_total"
FIELDNAME_PATHO_RAW = "patho_raw"
FIELDNAME_REPORT_RAW = "report_raw"
FIELDNAME_INTERVENTION_IMAGE_PREDICTIONS = "image_predictions"
FIELDNAME_VIDEO_DURATION = "duration"
FIELDNAME_N_EXTRACTED_FRAMES = "n_extracted_frames"
FIELDNAME_PREDICTIONS_SMOOTH = "predictions_smooth"
FIELDNAME_PREDICTION_FLANKS = "prediction_flanks"
FIELDNAME_ANNOTATION_FLANKS = "annotation_flanks"
FIELDNAME_FLANK_RANGES = "ranges"
FIELDNAME_FLANK_FRAMES = "frames"
FIELDNAME_VIDEO_ANNOTATION_TIMESTAMP = "latest_video_annotation_date"

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
COLNAME_RESULT_TYPE = "type"
RESULT_TYPE_PREDICTION = "prediction"
RESULT_TYPE_ANNOTATION = "annotation"
RESULT_TYPE_PREDICTION_SMOOTH = "prediction_smooth"
COLNAME_PREDICTION_VALUE = "value"
FIELDNAME_PREDICTIONS = "predictions"
COLNAME_RESULT_LABEL = "label"
COLNAME_RESULT_NAME = "name"

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
PREFIX_IMAGE = "images"
PREFIX_EXISTS = "exists"

DF_COL_ENTITY = "entity"
DF_COL_VALUE = "value"
DF_COL_VALUE_TYPE = "value_type"
DF_COL_DATE = "date"
