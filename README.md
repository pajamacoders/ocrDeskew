# ocrDeskew

## requirements
도커 파일 참조
```bash
docker build -t ocr_deskew:latest .
docker run -it --gpus all -p 0.0.0.0:5000:5000 -v /path/to/project:/code -v /path/to/data:/home/train_data_v2 ocr_deskew:latest
```

## run command
train:
```bash
python3 train.py --config config/resnet_ocr.json --run_name {RUN_NAME_YOU_WANT}
```

test:
```bash
python3 test.py --config config/rotmodel_fft_version.json --run_name {RUN_NAME_YOU_WANT}
```

serve:
```bash
python3 serve.py --deskew_config config/rotmodel_fft_version_small.json --orientation_config config/upside_down_vit.json --run_name {RUN_NAME_YOU_WANT} 
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

model config file: config/rotmodel_fft_version_small_range_89.json

테스트시 위 모델 다운 받고 'config/rotmodel_fft_version_small_range_89.json' 의 model_cfg['args']['pretrained'] 의 path를 다운 받은 모델의 
path로 설정 후 test.py 의 --config 파라미터 값으로 이 파일을 지정.
config file 내의 아래 설정값을 주어진 값으로 바꿀것


3. [rot_range_90_correction_checkpoint.pth](https://drive.google.com/file/d/1FpCyAc3vpTpMR-0oYjz1hSdkBfPkQ1HT/view?usp=sharing)
tag: rot_range_90
데이터셋: 학습시 입력 이미지를 -90~+90 도 범위 내에서 임의로 회전 시키고 0.5도 단위로 class를 구분해 회전의 정도를 classification 
문제로 정의 하고 학습시킨 checkpoint

실험 결과:
cls_loss:0.0842, precision:0.9336, recall:0.9326, f1_score:0.9281
model config file: config/rotmodel_fft_version_small_range_90.json

인퍼런스 결과 해석:
회전되어 들어오는 이미지는 보정되어 0,180,-90,90 도 회전된 이미지로 바뀐다. 
예) 시계 방향으로 79도 회전된 이미지가 입력으로 들어오면 네트워크는 이 이미지가 회전된 정도를 79도 로 예측해 반시계반향으로 79도 회전시켜  0도 회전된 이미지로 만들 수도 있고 -11도 회전된 이미지로 판단하고 시계방향으로 11도 회전시켜 (79+11)=90 도 회전된 이미지로 보정할 수도 있다. 
이러한 경향성이 있으므로 정확한 보정을 위해서는 0,180,90,-90 도 4개 방향으로 회전된 모델을 학습 시켜 모든 이미지를 0도 회전된 이미지로 만드는 2 stage 접근이 필요 한것으로 보인다. 


테스트시 위 모델 다운 받고 'config/rotmodel_fft_version_small_range_90.json' 의 model_cfg['args']['pretrained'] 의 path를 다운 받은 모델의 
path로 설정 후 test.py 의 --config 파라미터 값으로 이 파일을 지정.

## 방향 보정 모델 체크 포인트[-90,0,90,180 도 회전 된 이미지의 방향을 예측 하는 모델]
1. [upside_down_mobilevit_v1.0](https://drive.google.com/file/d/1ecFc8iMWZl4H3a4NTsLeKKvng6a8jubK/view?usp=sharing)

설명: 상하 반전된 문서를 보정하기 위한 네트워크의 체크 포인트

데이터셋: aihub 데이터셋을 상하 반전 및 -5~5degree 사이에서 회전시켜 학습 시킴

테스트 결과:테스트 데이터셋 기준(이미지 약 3700장)

cls_loss:0.041, precision: 0.992, recall: 0.992, f1 score0.992

model config file: config/upside_down_vit.json

2. [direction_prediction_model_v2.pth](https://drive.google.com/file/d/1oO6zCR1POSW7pkScmb5t5cVqar-U2LEg/view?usp=sharing)

설명: -90도, 0도, 90도, 180도 방향전환 된 문서를 보정하기 위한 네트워크의 체크 포인트

데이터셋: aihub 데이터셋의 이미지를 -90도,0도, 90도, 180도 회전 및 -3~3degree 사이에서 회전시켜 학습 시킴

테스트 결과:테스트 데이터셋 기준(이미지 약 3700장)

cls_loss:0.041, precision: 0.977, recall: 0.977, f1 score: 0.977

model config file: config/direction_prediction_model_vit_v2.json



## serve file 사용 방법
예)
1. 회전 모델 체크포인트 rot_range_90_correction_checkpoint.pth를  ./checkpoints 에 다운로드
2. 해당 체크포인트의 model config 파일(config/rotmodel_fft_version_small_range_90.json)의 model_cfg->args에 pretrained 파라미터 추가
```
    -> "model_cfg":{
        "type":"DeskewNetV4",
        "args":{"buckets":361, "last_fc_in_ch":128, "pretrained":"checkpoints/rot_range_90_correction_checkpoint.pth"}
    }
```

3. 상하 반전 보정 모델 체크 포인트 direction_prediction_model_vit_v2.json 를 .checkpoints/에 다운로드
4. 상하 반전 보정 모델의 model config 파일(config/direction_prediction_model_vit_v2.json )의  model_cfg->args에 pretrained 파라미터 추가
```
  ->  "model_cfg":{
        "type":"MobileViT",
        "args":{
            "image_size":[512,512],
            "dims":[64, 80, 96],
            "channels":[16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
            "num_classes":1,
            "expansion":2,
            "pretrained":"checkpoints/direction_prediction_model_vit_v2.json"
        }
    }
```
5. 실행
```bash
python3 serve.py --deskew_config config/rotmodel_fft_version_small_range_90.json --orientation_config config/direction_prediction_model_vit_v2.json --run_name your_run_name
```

## run tracking ui
```bash
mlflow ui
```