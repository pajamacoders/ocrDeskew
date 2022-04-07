from .rotationnet import RotationNet
from .rotationnet_v2 import RotationNet_v2
from .rotation_model_v2 import DeskewNetV2
from .rotation_model_v3 import DeskewNetV3
from .rotation_model_fft import DeskewNetV4
from .rotation_model_fft_v2 import DeskewNetV5
from .rotation_model_fft_v3 import DeskewNetV6
from .rotation_model_fft_v4 import DeskewNetV7
from .rotation_model_fft_v5 import DeskewNetV8
from .mobilevit import MobileViT
from .craft import CRAFT
from .stn_model import STDirNet
models={
    "RotationNet":RotationNet,
    "RotationNet_v2":RotationNet_v2,
    "DeskewNetV2":DeskewNetV2,
    "DeskewNetV3":DeskewNetV3,
    "DeskewNetV4":DeskewNetV4,
    "MobileViT":MobileViT,
    "CRAFT":CRAFT,
    "DeskewNetV5":DeskewNetV5,
    "DeskewNetV6":DeskewNetV6,
    "DeskewNetV7":DeskewNetV7,
    "DeskewNetV8":DeskewNetV8,
    "STDirNet":STDirNet,
}
def build_model(type, args):
    return models[type](**args)
