# ocrDeskew
본 프로젝트에서는 [CRAFT](https://github.com/clovaai/CRAFT-pytorch) 를 character detection 을 위해 사용 한다.<br/>  
Direction Estimation model은 한글 문서에 대해 작동한다. 영문, 숫자로 이루어진 문서에 대한 작동을 보장하지 않는다. <br/><br/>

## requirements
도커 파일 참조
```bash
docker build -t ocr_deskew:latest .
docker run -it --gpus all -p 0.0.0.0:5000:5000 -v /path/to/project:/code -v /path/to/data:/home/train_data_v2 ocr_deskew:latest
```

## Model description
1. Deskew model: <br/>
Deskewing is a process whereby skew is removed by rotating an image by the same amount as its skew.<br/>
It estimates angle between -90 to 90 degree.<br/>
디스큐 모델은 -90~90 도 사이의 임의의 각도로 회전된 문서가 회정된 정도를 예측한다. <br/>
각도 추정 문제를 classification 문제로 정의 하고 -90~90 도 사이의 각도를 0.5도 단위로 클래스로 정의해 어느 클래스에 속하는지 분류 한다. <br/>

2. direction estimation model: 
Deskew model is not perpect so to make up deskew model.<br/>
Direction estimation model classifies input image to 4 class. <br/>
class 0: image is not rotated.<br/>
class 1: image is rotated by 90 degree.<br/>
class 2: image is rotated by 180 degree.<br/>
class 3: image is rotated by 270 degree.<br/>
디렉션 추정 모델은 입력 이미지가 0,90,180,270 도 중 어느 각도로 회전 되어있는지 4 클래스 분류 문제를 푼다.<br/> 

회전된 문서가 입력으로 들어왔을때 <br/>
input_img -> deskew model -> direction estimation model -> output image <br/>
의 순으로 파이프 라인을 구성해 최종 output image가 회전없는 문서가 되도록 하는 것을 목표로 한다. <br/><br/><br/>


## Deskew model train 
train:
```bash
python3 train.py --config config/rotmodel_fft_version_mid_range_90.json --run_name {RUN_NAME_YOU_WANT}
```

test:
```bash
python3 test.py --config config/rotmodel_fft_version_mid_range_90_test.json --run_name {RUN_NAME_YOU_WANT}
```
<br/><br/>

## Direction Estimation Model train
train:
```bash
python3 train.py --config config/STDirNet_config.json --run_name {RUN_NAME_YOU_WANT}
```

test:
```bash
python3 inference_direction.py --config config/STDirNet_config_valid.json --run_name {RUN_NAME_YOU_WANT}
```
<br/><br/>

## Serve model
serve:
```bash
python3 serve_patch_base.py --config config/deskew_and_direction_correction_v2.json --run_name {RUN_NAME_YOU_WANT} --input_data_dir {path/to/image_dir/*}
```
<br/><br/>

## Checkpoints

1. Deskew model checkpoint <br/>
 [deskew_model_checkpoint.pth](https://drive.google.com/file/d/1Btgi_taAgAdqAvvzaELI8OrVt0eJZI_6/view?usp=sharing) <br/>

학습 방식: 학습시 입력 이미지를 -90~+90 도 범위 내에서 임의로 회전 시키고 0.5도 단위로 class를 구분해 <br/>
회전의 정도를 classification 문제로 정의 하고 학습시킨 checkpoint.<br/>
아래 augmentation 적용: <br/>
    resize aspect ratio <br/>
    random line erasing <br/>
    random rotation <br/>

모델 입력 이미지 size: 1x512x512 (ch x height x width) <br/>

테스트 결과: <br/>
1.1 입력 이미지가 -90~+90 도 이내의 회전을 보일 경우 testset 에 대해 아래 결과를 얻음: <br/>
cls_loss:0.057, precision:0.946, recall:0.945, f1_score:0.938 <br/><br/>
model config file: config/rotmodel_fft_version_mid_range_90_test.json<br/>
테스트시 위 모델 다운 받고 'config/rotmodel_fft_version_mid_range_90_test.json' 의 <br/>
model_cfg['args']['pretrained'] 의 path를 다운 받은 모델의 path로 설정 후 test.py 의 --config 파라미터 값으로 이 파일을 지정. <br/>


2. Direction estimation model checkpoint <br/>
[STNet_kor_eng_num.pth](https://drive.google.com/file/d/172yx5bHs7C6i5EM_6n47F5fjne7e64lt/view?usp=sharing) <br/>
학습 방식: 입력 이미지를 0,90,180,270 4가지 중 한 방향으로 회전 시켜 입력하고 <br/>
  degree  class <br/>
    0   ->  0   <br/>
    90  ->  1 <br/>
    180 ->  2 <br/>
    270 ->  3 <br/>
위와 같이 클래스를 부여해 분류 하도록 학습 시킴.  <br/>

모델 입력 이미지 size: 1x40x40 (ch x height x width) <br/>

실험 결과: <br/>
precidion:0.9603, recall:0.9583, f1_score:0.9584 <br/>
model config file: config/STDirNet_config_valid.json <br/>
테스트시 위 모델 다운 받고 'config/STDirNet_config_valid.json' 의 model_cfg['args']['pretrained'] 의 path를 다운 받은 모델의  <br/>
path로 설정 후 test.py 의 --config 파라미터 값으로 이 파일을 지정. <br/>
config file 내의 아래 설정값을 주어진 값으로 바꿀것 <br/>


3. [craft_mlt_25k.pth](https://drive.google.com/file/d/1YEYHzt4sD7LVH-HB0mfnrxlcrCmRHDh9/view?usp=sharing) <br/>
pretrained craft model checkpoint <br/><br/>
모델 입력 이미지 size: 3 x height x width 


## serve_patch_base.py 사용 방법
예)
1. deskew model 체크포인트 deskew_model_checkpoint.pth를  './checkpoints' 에 다운로드 <br/>
2. direction correction model  체크 포인트 STNet_kor_eng_num.pth를 './checkpoints' 에 다운로드 <br/>
3. craft model의 체크 포인트 craft_mlt_25k.pth 를 './checkpoints' 에 다운로드  <br/>
4. serve_patch_base.py 를 위한 configuration file 설정 <br/>
configuration file location : config/deskew_and_direction_correction_v2.json <br/>
위 config 파일 내에 다음과 같이 각 모델의 정보와 checkpoint 위치 설정 <br/>
ex)
```
  "craft_model_cfg":{
        "type": "CRAFT", 
        "args":{"craft_pretrained":"checkpoints/craft_mlt_25k.pth"}
    },
    "deskew_model_cfg":{
        "type":"DeskewNetV4",
        "args":{"buckets":360, "last_fc_in_ch":256, "pretrained":"checkpoints/deskew_model_checkpoint.pth"}
    },
    "direction_model_cfg":{
        "type":"STDirNet",
        "args":{"pretrained":"checkpoints/STNet_kor_eng_num.pth"}
    }
```
5. 실행
```bash
python3 serve_patch_base.py --config config/deskew_and_direction_correction_v2.json --run_name {RUN_NAME_YOU_WANT} --input_data_dir path/to/image_dir
```

## run tracking ui
```bash
docker-compose up -d
```