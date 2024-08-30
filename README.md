# 2023 바이오헬스 데이터 경진대회 - 치의학 분야 (바이오헬스 혁신융합대학 7개 대학 재학생 부문)
## [컴퓨터비전] 사랑니 발치 수술 후 위험도 예측 모델 개발

### 코드 구조

```
$USER/RESULT
├────TRAIN/
│   ├── TL_AdamW.py
│   └── FT_RMSprop.py 
├────TEST/
│  ├── predict_AdamWRMSprop.py
├── README.md
├── FT_RMSdragon.pth
├── BEST_TL_AdamWdragon.pth
└── fin_tuned_model_AdamWRMSdragon2.csv

```
```
├── TRAIN : 학습에 필요한 TL_AdamW.py, FT_RMSprop.py 파일을 담은 폴더
│    ├── TL_AdamW.py :
│    │		AdamW optimizer를 이용해 Transfer learning을 진행(공식문서참고 및 실험적으로 파라미터 결정)**
│    │				1. classification layer에 이진분류 결과를 도출하도록 layer를 추가
│    │				2. 150 epoch 중 f1-score가 가장 높을 때의 pth 저장
│    └── FT_RMSprop.py :
│		TL_AdamW.py에서 얻은 pth와 RMSprop optimizer를 이용해 미세조정**
│				1. densenet121의 모든 파라미터 동결(freeze) 후 classification layer만 동결해제
│				2. 공식문서를 참고하여 실험적으로 얻은 파라미터와 학습률을 통해 scheduler 세팅
│				3. 최종적으로 얻은 pth 생성
│
├── TEST : 추론에 필요한 predict_AdamWRMSprop.py 파일을 담은 폴더
│    └──  predict_AdamWRMSprop.py:
│		baseline predict.py와 동일. (경로만 다름)**
│			1. TRAIN의 두 파일을 통해 최종적으로 얻은 pth로 새로운 데이터에 대한 예측 수행
│			2. 예측 결과 0/1을 low/high로 바꾸어 csv 형태로 저장
│  				(fin_tuned_model_AdamWRMSdragon2.csv)
│
├── FT_RMSdragon.pth : 최종적으로 학습된 모델
├── BEST_TL_AdamWdragon.pth : 최적 성능일때 저장된 모델
└── fin_tuned_model_AdamWRMSdragon2.csv : 추론을 돌린 결과
```
---

### 학습(TRAIN) 순서 및 방법 
1. 'python3.9 TL_AdamW.py' 파일 실행(먼저 수행)
2. 'python3.9 FT_RMSprop.py' 파일 실행
3. 'USER/RESULT' 내에 결과가 저장됨


### 추론(PREDICT) 순서 및 방법
1. 'python3.9 predict_AdamWRMSprop.py' 실행
2. 'USER/RESULT/' 내에 결과 파일(fin_tuned_model_AdamWRMSdragon2.csv)이 저장됨

### 공식 URL
* densenet121 : https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html
* RMSprop : https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
* AdamW : https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

### 사용 라이브러리 및 환경
* 라이브러리
	- torch
	- torchvision
	- torch.optim의 RMSprop, lr_scheduler
	- PIL
	- tqdm
	- sklearn
	- pandas
	- numpy

* 환경
	- python : 3.9.18
	- cuda : runtimeAPI 11.3, driverAPI 11.7
	- os : Ubuntu 20.04 LTS
	
hi
