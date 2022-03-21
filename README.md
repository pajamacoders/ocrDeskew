# ocrDeskew

## requirements
도커 파일 참조
```bash
docker build -t ocr_deskew:latest .
```


## run command(회전 보정)
```bash
python3 train.py --task deskew --config config/resnet_ocr.json --run_name {RUN_NAME_YOU_WANT}
```
test:
```bash
python3 test.py --task deskew --config config/rotmodel_fft_version.json --run_name {RUN_NAME_YOU_WANT}
```

## run command(상하 반전 classification)
train:
```bash
python3 train.py --task orientation --config config/resnet_ocr.json --run_name {RUN_NAME_YOU_WANT}
```
test:
```bash
python3 test.py --task orientation --config config/rotmodel_fft_version.json --run_name {RUN_NAME_YOU_WANT}
```


## 회전 보정 모델 체크 포인트
1. [cls_model_deg_range_89](https://drive.google.com/file/d/1P_fj-hDsW4TJkUCo-jKMVQEPrTPsy7M0/view?usp=sharing)

tag: cls_model_deg_range_89

데이터셋: 학습시 입력 이미지를 -89~+89 도 범위 내에서 임의로 회전 시키고 0.5도 단위로 class를 구분해 회전의 정도를 classification 
문제로 정의 하고 학습시킨 checkpoint

테스트 결과:
1.1 입력 이미지가 -89~+89 도 이내의 회전을 보일 경우 testset 에 대해 아래 결과를 얻음:

cls_loss:0.0898, precision:0.9495, recall:0.9454, f1_score:0.9441

model config file: config/rotmodel_fft_version_large.json

테스트시 위 모델 다운 받고 'config/rotmodel_fft_version_large.json' 의 model_cfg['args']['pretrained'] 의 path를 다운 받은 모델의 
path로 설정 후 test.py 의 --config 파라미터 값으로 이 파일을 지정.


2. [deskew_model_aug_rot_180_state_dict](https://drive.google.com/file/d/1tXntxFk5KXfYfQS70FsCcGnsqkwfgXhF/view?usp=sharing)
데이터셋: 학습시 입력 이미지를 -89~+89 도 범위 내에서 임의로 회전 시키고 0.5도 단위로 class를 구분해 회전의 정도를 classification 
문제로 정의 하고 학습시킨 checkpoint

실험 결과:
cls_loss:0.0842, precision:0.9540, recall:0.9527, f1_score:0.9485

model config file: config/rotmodel_fft_version_small.json

테스트시 위 모델 다운 받고 'config/rotmodel_fft_version_small.json' 의 model_cfg['args']['pretrained'] 의 path를 다운 받은 모델의 
path로 설정 후 test.py 의 --config 파라미터 값으로 이 파일을 지정.


## 상하 반전 보정 모델 체크 포인트
1. [upside_down_mobilevit_v1.0](https://drive.google.com/file/d/1ecFc8iMWZl4H3a4NTsLeKKvng6a8jubK/view?usp=sharing)
tag:
설명: 상하 반전된 문서를 보정하기 위한 네트워크의 체크 포인트

데이터셋: aihub 데이터셋을 상하 반전 및 -5~5degree 사이에서 회전시켜 학습 시킴

테스트 결과:테스트 데이터셋 기준(이미지 약 3600장)

cls_loss:0.041, precision: 0.992, recall: 0.992, f1 score0.992

model config file: config/upside_down_vit.json


## serve file 사용 방법
예)
1. 회전 모델 체크포인트 deskew_model_aug_rot_180_state_dict를  ./checkpoints 에 다운로드
2. 해당 체크포인트의 model config 파일(config/rotmodel_fft_version_small.json)의 model_cfg->args에 pretrained 파라미터 추가
  -> "model_cfg":{
        "type":"DeskewNetV4",
        "args":{"buckets":356, "last_fc_in_ch":128, "pretrained":"checkpoints/deskew_model_aug_rot_180_state_dict.pth"}
    }
3. 상하 반전 보정 모델 체크 포인트 upside_down_mobilevit_v1.0를 .checkpoints/에 다운로드
4. 상하 반전 보정 모델의 model config 파일(config/upside_down_vit.json)의  model_cfg->args에 pretrained 파라미터 추가
  ->  "model_cfg":{
        "type":"MobileViT",
        "args":{
            "image_size":[512,512],
            "dims":[64, 80, 96],
            "channels":[16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
            "num_classes":1,
            "expansion":2,
            "pretrained":"checkpoints/upside_down_v1.0.pth"
        }
    }
5. 실행
```bash
python3 serve.py --deskew_config config/rotmodel_fft_version_small.json --orientation_config config/upside_down_vit.json --run_name your_run_name
```

## run tracking ui
```bash
mlflow ui
```