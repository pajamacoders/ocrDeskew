from torchvision import transforms
#from .transform import RandomRotation, Normalize, RandomCrop, Resize, Shaper, RandomOrientation, RandomRotationTest, RandomLineErasing
from .transform import *
from .custom_lr_scheduler import CosineAnnealingWarmUpRestarts
transfactory={
    'Normalize':Normalize,
    'Resize':Resize,
    'ResizeAspectRatio':ResizeAspectRatio,
    'RandomCropAndPad':RandomCropAndPad,
    'RandomRotation':RandomRotation,
    'RandomOrientation':RandomOrientation,
    'Shaper':Shaper,
    'RandomRotationTest':RandomRotationTest,
    'RandomLineErasing':RandomLineErasing,
    'NormalizeMeanStd':NormalizeMeanStd
}

lr_scheduler_factory={
    'CosineAnnealingWarmUpRestarts':CosineAnnealingWarmUpRestarts,
}

def build_transformer(cfg):
    elem_list = [transfactory[type](**args) for type, args in cfg.items()]
    return transforms.Compose(elem_list)

def build_lr_scheduler(type, args, opt):
    return lr_scheduler_factory[type](**args, optimizer=opt)