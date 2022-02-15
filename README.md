# ocrDeskew

## requirements
mlflow==1.23.1
opencv-python==3.4.11


## run command
train:
```bash
python3 train.py --config config/resnet_ocr.json --run_name {RUN_NAME_YOU_WANT}
```

## run tracking ui
```bash
mlflow ui
```