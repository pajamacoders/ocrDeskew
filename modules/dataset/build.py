from torch.utils.data import DataLoader, RandomSampler
from .ocrDataset import OCRDataset
from .ocrDataset3ch import OCRDataset3ch
from .fontDataset import FontDataSet

ocrFactory = {'OCRDataset':OCRDataset,
'OCRDataset3ch':OCRDataset3ch,
'FontDataSet':FontDataSet}

def build_dataloader(type, train_batch, valid_batch, train, valid, augment_fn):
    train_dataset = ocrFactory[type](**train, transformer=augment_fn)
    valid_dataset = ocrFactory[type](**valid, transformer=augment_fn)

    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=(sampler is None), sampler=sampler, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    return train_loader, valid_loader
