from .multilabel_googlenet import MultilabelGoogleNet
from ..classes.config import AiLabelConfig
from pathlib import Path

LOOKUP_MODELS = {
    "multilabel_googlenet": MultilabelGoogleNet
}

def get_model(ai_config:AiLabelConfig, ckpt_path:Path, eval = True, cuda = True):
    model = LOOKUP_MODELS[ai_config.ai_settings.base_model_name]
    model = MultilabelGoogleNet.load_from_checkpoint(ckpt_path.as_posix())
    if cuda:
        model.to(0)
    if eval:
        model.eval()
        model.freeze()

    return model