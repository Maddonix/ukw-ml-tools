from .multilabel_googlenet import MultilabelGoogleNet
from .binary_resnext import BinaryResnext
from ..classes.config import AiLabelConfig
from pathlib import Path
from .regnet import RegNet
from .sim_clr import SimCLR

LOOKUP_MODELS = {
    "multilabel_googlenet": MultilabelGoogleNet,
    "binary_resnext": BinaryResnext,
    "regnet": RegNet,
    "sim_clr": SimCLR
}


def get_model(name: str, ckpt_path: Path, eval=True, cuda=True):
    model = LOOKUP_MODELS[name]
    model = model.load_from_checkpoint(ckpt_path.as_posix())
    if cuda:
        model.to(0)
    if eval:
        print("Load in Evaluation Mode")
        model.eval()
        model.freeze()

    return model
