from torchvision import transforms
from .transform import RandomRotation, Normalize, ToTensor,RandomCrop, Resize
transfactory={'Resize':Resize,
'RandomCrop':RandomCrop,
'RandomRotation':RandomRotation,
'Normalize':Normalize,
'ToTensor':ToTensor
}

def build_transformer(cfg):
    elem_list = [transfactory[type](**args) for type, args in cfg.items()]
    return transforms.Compose(elem_list)