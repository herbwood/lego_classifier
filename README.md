
# Lego Classifier

## Overview

Lego Classifier는 AlexNet을 활용하여 10종의 레고 이미지를 분류하는 프로젝트입니다. 

## Requirements and run
```
pip install -r requirements.txt
git clone https://github.com/herbwood/crops_doctor.git
python train.py
```

## Project Details
<p align="center"><img src="https://user-images.githubusercontent.com/35513025/76194797-4260a580-622a-11ea-9af3-a2eea75d26ea.png"></p>

- [kaggle LEGO bricks image dataset](https://www.kaggle.com/joosthazelzet/lego-brick-images) 데이터셋을 활용하여 프로젝트 진행(10종의 lego만 사용)
- 4만장의 이미지 중 32000장을 train set으로, 8000장을 test set으로 사용
- [Image normalization in pytorch](https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7)를 참고하여 최적의 평균값 찾아 정규화 진행
- pretrained되지 않은 AlexNet을 사용하여 모델 학습 진행
- epoch=40, optimizer=Adam, learning rate=0.001 사용하여 최종 **87.31%** accuracy를 보임
- [pytorch project template](https://github.com/victoresque/pytorch-template)을 참고하여 전체 프로젝트 구조 구성

