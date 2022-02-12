from torch.utils.data import DataLoader, RandomSampler
from .ocrDataset import OCRDataset

ocrFactory = {'OCRDataset':OCRDataset}

def build_dataloader(type, train, valid):
    train_dataset = ocrFactory[type](**train)
    valid_dataset = ocrFactory[type](**valid)
    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, shuffle=(sampler is None), sampler=sampler)
    valid_loader = DataLoader(valid_dataset, shuffle=False, sampler=sampler)
    return train_loader, valid_loader
