from torchvision import transforms
from .transform import RandomRotation, Normalize, RandomCrop, Resize, Shaper
transfactory={'Resize':Resize,
'RandomCrop':RandomCrop,
'RandomRotation':RandomRotation,
'Normalize':Normalize,
'Shaper':Shaper
}

def build_transformer(cfg):
    elem_list = [transfactory[type](**args) for type, args in cfg.items()]
    return transforms.Compose(elem_list)