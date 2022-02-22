from .rotation_model import DeskewNet
from .rotation_model_v2 import DeskewNetV2
from .rotation_model_v3 import DeskewNetV3
from .rotation_model_fft import DeskewNetV4
models={
    "DeskewNet":DeskewNet,
    "DeskewNetV2":DeskewNetV2,
    "DeskewNetV3":DeskewNetV3,
    "DeskewNetV4":DeskewNetV4,
}
def build_model(type, args):
    return models[type](**args)
