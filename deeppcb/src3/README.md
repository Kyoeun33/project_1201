# EfficientNet-B0와 ResNet18의 PCB 정상·불량 분류 비교

`scr3`는 ImageNet으로 사전학습한 EfficientNet-B0의 마지막 선형층을 출력 2개로 바꾸고 전체 네트워크를 학습한다. A·B·C 각각을 `src2`에 저장한 ResNet18의 같은 데이터 조합과 비교한다. 입력은 PCB 사진 한 장이고 클래스는 `0 normal`, `1 defect`다. [EfficientNet-B0 설정](config.json) · [공유 모델 정의](../src2/model.py)

이진 분류층으로 교체한 뒤 실제로 학습되는 파라미터는 EfficientNet-B0 4,010,110개, ResNet18 11,177,538개다.

## 실행 순서

기존 프로젝트 루트에서 가상환경을 활성화하고 `deeppcb`로 이동한다. 아래 명령은 이 폴더를 기준으로 실행한다.

```powershell
.\.venv\Scripts\Activate.ps1
cd .\data_example\DeepPCB\deeppcb
python -m pip install -r scr3/requirements.txt
python -m scr3.01_prepare
python -m scr3.02_train --experiments A B C
python -m scr3.03_evaluate --experiments A B C
python -m scr3.04_predict scr3/artifacts/runs/C/best.pt data/group00041/00041/00041000_test.jpg
```

| 파일 | 실행하면 얻는 결과 |
| --- | --- |
| [01_prepare.py](01_prepare.py) | 원본 그룹 분할, 타일 라벨, 누수 검사, 정답 미리보기. 기존 ResNet18 원본·타일 목록의 해시와 대조한다. |
| [02_train.py](02_train.py) | EfficientNet-B0의 A·B·C 체크포인트, 학습 로그와 검증 결과. 특정 조합만 학습하려면 예를 들어 `--experiments C`를 사용한다. |
| [03_evaluate.py](03_evaluate.py) | 고정된 체크포인트의 원본별 시험 예측과 두 아키텍처의 비교 CSV |
| [04_predict.py](04_predict.py) | 입력 사진의 클래스 번호·이름, 정상·불량 확률, 운영 임계값, 사용한 타일 수 |
| [05_pcb_binary_analysis.ipynb](05_pcb_binary_analysis.ipynb) | 두 아키텍처 비교, 데이터·라벨 예시, EfficientNet-B0 학습 곡선, 혼동행렬, 출처별 결과와 오탐·미검출 사례 |

`04_predict`는 개별 사진을 판정할 때 사용한다. 노트북은 03단계에서 저장한 평가 결과를 읽는다. PyCharm에서 노트북을 열거나 다음 명령으로 다시 실행할 수 있다.

```powershell
python -m jupyter nbconvert --execute --to notebook --inplace scr3/05_pcb_binary_analysis.ipynb
```

## 두 모델이 공유하는 조건

| 조합 | 가중치 학습에 사용하는 출처 |
| --- | --- |
| A | DeepPCB |
| B | DeepPCB + PKU-Market-PCB |
| C | DeepPCB + PKU-Market-PCB + VisA PCB1~PCB4 |

이번 분할의 원본 8,119장은 학습 5,559장, validation 1,344장, test 1,216장으로 나뉜다. A·B·C 모두 세 출처를 포함한 공통 validation과 test를 사용한다. PKU는 `../out_data/PCB_DATASET`의 Pascal VOC XML과 정상 템플릿을 사용한다. [분할별 실제 집계](artifacts/prepared/counts.csv) · [타일·그룹 분할의 세부 기준](../src2/README.md)

- seed 42, RGB, 기본 타일 640×640, 겹침 128픽셀, 결함 포함 비율 0.5를 사용한다. DeepPCB는 전체 640×640을 한 장으로 사용한다. PKU·VisA는 자른 뒤에 224×224로 조정한다.
- 각 아키텍처의 ImageNet 사전학습 가중치로 시작한다. 배치 48, 3 epoch, AdamW, 학습률 0.0002, 같은 학습 증강과 클래스 가중치를 적용한다. CUDA를 사용할 수 있으면 GPU에서 실행하고 CPU에서도 실행할 수 있다.
- validation 원본의 로그 손실이 가장 낮은 epoch를 저장한다. 원본 불량 점수는 모든 타일에서 나온 불량 확률의 최댓값이다. 원본을 판정할 때는 정답이 일부만 걸친 경계 타일까지 점수를 계산한다.
- 임계값 0.5의 결과와 각 모델이 validation에서 고른 운영 임계값의 결과를 함께 계산한다. 운영 목표는 불량 recall 90% 이상, 정상 FPR 5% 이하이다. 동시 충족이 어려우면 FPR 상한 안에서 recall을 최대화하며 선택 사유를 저장한다.

## 비교 결과 읽기

[model_comparison.csv](artifacts/model_comparison.csv)에서 `experiment`, `scope`, `mode`가 같은 두 행을 보면 된다. 예를 들어 `C / VisA/pcb2 / default`는 세 출처로 학습한 두 모델을 VisA PCB2 원본에 임계값 0.5로 적용한 결과다. 같은 수치 임계값에서 비교하려면 `default`, 각 모델의 운영 기준에서 비교하려면 `operating`을 선택한다.

표에는 정상·불량 수, TN·FP·FN·TP, accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, FPR, FNR이 들어간다. 전체·DeepPCB·PKU·VisA와 VisA의 PCB1~PCB4를 제공한다. 분모가 없어 계산할 수 없는 지표는 빈 값으로 저장하고 `undefined`에 이름을 남긴다. PKU test의 정상 원본은 2장이어서 오탐 한 장이 FPR을 50%p 바꾼다.

[비교 코드](comparison.py)는 저장 당시의 모델명, 분할 목록, 학습 조건, 검증 임계값 정책, 시험 결과 파일 해시를 확인하고 두 예측의 원본 ID·정답·출처·타일 수를 대조한다. 두 모델의 결과가 모두 있고 조건이 일치하는 조합만 CSV에 들어간다. 완료된 시험 추론은 재실행 시 저장 결과를 사용한다. 노트북의 `DETAIL_EXPERIMENT`와 `DETAIL_MODE`로 출처별 비교표를 바꿀 수 있다.

### 이번 3 epoch 실험의 시험 F1

| 조합 | ResNet18 기본 0.5 | EfficientNet-B0 기본 0.5 | ResNet18 운영 | EfficientNet-B0 운영 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.420 | 0.568 | 0.619 | 0.774 |
| B | 0.613 | 0.550 | 0.821 | 0.622 |
| C | 0.947 | 0.969 | 0.903 | 0.960 |

공통 시험 원본은 정상 820장, 불량 396장이었다. C의 기본 0.5에서 ResNet18은 오탐 28장·미검출 15장, EfficientNet-B0는 오탐 16장·미검출 9장이었다. VisA에서 오탐이 21장→2장, 미검출이 12장→6장으로 줄었다. 같은 조건에서 DeepPCB 오탐은 5장→12장으로 늘었다.

EfficientNet-B0 C는 validation 손실이 가장 낮은 2번째 epoch를 사용했다. validation에서 선택한 운영 임계값 0.805096은 validation recall 95.65%, FPR 4.98%로 두 목표를 충족했다. 이 임계값에서 시험 오탐은 8장, 미검출은 23장이었고, PKU 정상 2장은 모두 오탐이었다. 이 수치는 seed 42로 한 번 학습한 결과다. [C의 검증 선택 근거](artifacts/runs/C/threshold.json) · [C의 시험 지표](artifacts/runs/C/test/metrics.csv)

## 새 실험 설정과 파일 위치

데이터와 출력 경로는 설정 파일 위치를 기준으로 해석한다. `output_dir`은 EfficientNet-B0의 산출물, `resnet_output_dir`은 비교할 ResNet18 산출물이다. 기본 경로는 각각 `scr3/artifacts`, `src2/artifacts`다. 다른 epoch 수나 학습률을 시도할 때는 설정을 복사해 새 `output_dir`을 지정한 뒤 01~03단계에 `--config scr3/새설정.json`을 전달한다. 같은 조건으로 학습한 ResNet18을 비교하려면 그 결과 위치를 `resnet_output_dir`로 지정한다.

가중치는 `artifacts/runs/{A,B,C}/best.pt`, 검증·시험 원본별 예측과 지표는 각 실험의 `validation/`, `test/`에 저장한다. 각 번호 스크립트는 같은 프로젝트의 `src2` 데이터 처리·훈련·추론 함수를 사용한다.
