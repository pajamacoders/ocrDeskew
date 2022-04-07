from .ocrDataset import OCRDataset
from .ocrDataset3ch import OCRDataset3ch
from .fontDataset import FontDataSet
from .build import build_dataloader
__all__=['OCRDataset', 'build_dataloader', 'OCRDataset3ch','FontDataSet']