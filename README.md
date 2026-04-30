# Brain Tumor Segmentation with Clinical-Aware Evaluation

## 📌 프로젝트 개요
본 프로젝트는 뇌종양 분할(segmentation)을 수행하고, 기존의 Dice score 중심 평가를 넘어 **임상적으로 의미 있는 분석**을 수행하는 것을 목표로 함
일반적으로 segmentation 성능은 Dice score로 평가되지만, 본 프로젝트에서는 Dice가 유사하더라도 임상적으로는 전혀 다른 결과를 낼 수 있음을 보여줌

## 📌 목표
- nnUNet, DynUNet 기반 뇌종양 segmentation 수행
- 단순 정확도(Dice)가 아닌 다음 지표를 포함한 분석:
  - Volume error (종양 크기 차이)
  - Spatial error (위치 오차)
  - Boundary error (HD95)
- MNI atlas 기반 해부학적 위치 분석

## 📌 방법
### 모델
- nnUNet (self-configuring baseline)
- DynUNet (MONAI 기반)
- 이외 UNet, AttentionUNet, SwinUNERT

### 데이터셋
- MSD Brain Tumor dataset (BRATS 형식)

### 전처리
- nnUNet preprocessing pipeline
- ANTs 기반 MNI152 공간 정합

## 📌 평가 지표
| 지표 | 설명 |
|------|------|
| Dice | segmentation overlap |
| HD95 | 경계 정확도 |
| Volume Error | 종양 크기 차이 |
| Center Distance | 위치 오차 |
| Atlas Overlap | 해부학적 위치 분석 |

## 📌 주요 결과
### 1. Dice는 비슷하지만 임상 결과는 다름
- nnUNet vs DynUNet → Dice 유사
- 하지만:
  - HD95 차이 큼
  - Volume 오차 큼
  - 위치 오차 발생 (~27 voxel)

### 2️. 과도한 segmentation (Over-segmentation)
| Class | GT (ml) | Pred (ml) |
|------|--------|----------|
| WT | 36.49 | 240.74 |
| TC | 27.40 | 188.12 |
| ET | 3.37 | 51.02 |

👉 실제보다 최대 6배 이상 크게 예측

### 3️. 위치 오차
- Center distance: 약 27 voxel
- Dice가 높아도 실제 위치는 크게 어긋날 수 있음

### 4️. Atlas 기반 분석 (MNI 공간)
- 중심 위치: Right Occipital (Inferior)
- 주요 포함 영역:
1. Superior Temporal Gyrus, anterior division (18.2%)
2. Parahippocampal Gyrus, posterior division (16.8%)
3. Temporal Occipital Fusiform Cortex (14.9%)
4. Temporal Fusiform Cortex, anterior division (11.5%)
5. Lingual Gyrus (7.9%)

👉 중심 위치와 실제 분포 영역이 불일치

## 📌 핵심 인사이트
> Dice score가 높다고 해서 임상적으로 신뢰할 수 있는 결과는 아니다.

- Volume error가 매우 중요
- 위치 오차 존재 가능
- 큰 종양에서는 center 기반 위치 추정 한계
- Atlas 기반 분석이 더 의미 있음

## 📌 개선 시도
- Connected component filtering 적용
- 과도한 segmentation 일부 개선
- 재학습 없이 성능 개선 가능

## 📌 결론
본 프로젝트는 기존 segmentation 평가의 한계를 분석하고, 다음과 같은 **임상 중심 평가 방법**을 제안한다:
- Volume 기반 평가
- 위치 기반 평가
- 해부학적 해석

## 📌 향후 계획
- WT / TC / ET 클래스별 위치 분석
- 모델 calibration
- 자동 임상 리포트 생성

## 🧩 사용 기술
- PyTorch
- MONAI
- nnUNet
- ANTs
- Nilearn / Nibabel
- FSL (MNI atlas)
