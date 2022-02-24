# ocrDeskew

## requirements
docker image : nvcr.io/nvidia/pytorch:22.01-py3
mlflow==1.23.1
opencv-python==3.4.11


## run command
train:
```bash
python3 train.py --config config/resnet_ocr.json --run_name {RUN_NAME_YOU_WANT}
```
test:
```bash
python3 test.py --config config/rotmodel_fft_version.json --run_name {RUN_NAME_YOU_WANT}
```

## checkpoints
1. [cls_model_deg_range_30](https://drive.google.com/file/d/1Q0mxqBSPREJYYjbcerorVv94v9pe2syC/view?usp=sharing)

tag: cls_model_deg_range_30

데이터셋: 이미지는 -30~+30 도 범위 내에서 임의로 회전 시키고 0.5 도 단위로 class를 구분해 회전의 정도를 classification 문제로 정의 하고 푼모델( ocropus3-ocrorot 모델 참조함)

테스트 결과:

1.1 입력 이미지가 -30~+30 내에 있을 경우 testset 에 대해 아래 결과를 얻음:

cls_loss:0.0887, precision:0.9731, recall:0.9810, f1_score:0.9730

1.2 입력 이미지가 -60~+60 내에 있을 경우 (bin size 1degree) testset 에 대해 아래 결과를 얻음:

ls_loss:11.7469, precision:0.0043, recall:0.0172, f1_score:0.0065 ( 사용 불가.)


## run tracking ui
```bash
mlflow ui
```