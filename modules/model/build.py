from .rotation_model import DeskewNet
from .rotation_model_v2 import DeskewNetV2
models={
    "DeskewNet":DeskewNet,
    "DeskewNetV2":DeskewNetV2,
}
def build_model(type, args):
    return models[type](**args)
