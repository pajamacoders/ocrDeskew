from torchvision import transforms
from .transform import RandomRotation, Normalize, ToTensor

def build_transformer():
    return transforms.Compose([RandomRotation()])