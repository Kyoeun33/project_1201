# PCB 정상·불량 이진 분류

PCB 사진 한 장을 받아 `0 normal` 또는 `1 defect`와 두 클래스의 확률을 출력한다. 학습 모델은 RGB 타일 하나를 입력받는다. 원본의 불량 점수는 모든 타일의 불량 softmax 확률 중 최댓값이며, 정상 점수는 1에서 그 값을 뺀 값이다. 고해상도 원본을 판정할 때 템플릿이나 주석을 제공할 필요가 없다. 원본 점수는 타일 점수를 집계한 값이므로 생산 라인의 불량 발생 빈도로 해석하려면 별도 확률 보정이 필요하다.

## 실행

Python 3.11을 사용해 검증했다. PowerShell에서 현재 프로젝트 루트의 가상환경을 활성화하고 **`src2`의 부모인 `deeppcb`** 로 이동한다. PyCharm에서 같은 인터프리터를 선택해도 된다.

```powershell
.\.venv\Scripts\Activate.ps1
cd .\data_example\DeepPCB\deeppcb
python -m pip install -r src2/requirements.txt
python -m src2.01_prepare
python -m src2.02_train --experiments A B C
python -m src2.03_evaluate --experiments A B C
python -m src2.04_predict src2/artifacts/runs/C/best.pt data/group00041/00041/00041000_test.jpg
```

마지막 명령은 클래스 번호, 이름, `prob_normal`, `prob_defect`, 저장된 임계값과 사용한 타일 수를 JSON으로 출력한다. 임의 사진 경로도 같은 자리에 넣을 수 있다. 기본 추론은 640×640 입력을 한 장으로 처리하고 나머지 입력에는 저장된 타일 설정을 적용한다. 타일 크기를 변경한 실험에서 640×640 PKU·VisA 이미지를 처리할 때는 `--source PKU` 또는 `--source VisA`로 원래 정책을 지정한다. `--device cpu`로 CPU 추론을 확인할 수 있다.

저장된 분석은 [05_pcb_binary_analysis.ipynb](05_pcb_binary_analysis.ipynb)에서 볼 수 있다. 처음부터 다시 읽으려면 아래 명령을 사용한다. 노트북은 학습이나 test 추론을 실행하지 않는다.

```powershell
python -m jupyter nbconvert --execute --to notebook --inplace src2/05_pcb_binary_analysis.ipynb
python -m unittest src2.test_contract
```

## 데이터 경로와 실제 파일 확인

경로는 [config.json](config.json) **파일이 있는 디렉터리**를 기준으로 해석한다. 절대 경로도 사용할 수 있다. 누락된 소스나 잘못 연결된 XML·마스크는 해당 경로와 함께 오류로 표시한다. 일부 데이터만 있을 때는 설정의 `sources`를 줄이고 가능한 실험만 실행한다. A/B/C 전체 비교에는 세 출처 모두가 필요하다.

| 출처 | 기본 루트 | 사용하는 원본과 정답 | 분할 그룹 |
| --- | --- | --- | --- |
| DeepPCB | `../data` | `trainval.txt`와 `test.txt`에 실린 기본 경로에서 `_test.jpg`, `_temp.jpg`, `*_not/*.txt`를 연결 | 같은 `groupNNNNN`의 모든 검사 이미지와 템플릿 |
| PKU-Market-PCB | `../out_data/PCB_DATASET` | `images/<유형>/*.jpg`에 같은 상대 경로의 `Annotations/*.xml`을 연결. `PCB_USED/01.JPG` 등은 정상 템플릿 | 검사 파일명 앞 숫자 및 템플릿 파일명의 보드 번호 |
| VisA | `../out_data` | `pcb1`~`pcb4`의 `Data/Images/{Normal,Anomaly}`와 `Data/Masks/Anomaly`. 제공된 `image_anno.csv`와도 대조 | PCB 종류·폴더·파일 ID를 합친 원본 이미지 ID |

- DeepPCB TXT 각 행은 두 꼭짓점 `x1 y1 x2 y2`와 결함 클래스 ID다. 프로젝트의 공식 목록·라벨 리더를 재사용한다. 상자 하나 이상이면 검사 이미지는 불량, 빈 공식 TXT는 정상, 템플릿은 정상이다. 원본 데이터 자체에 이진화된 회로 영상이 포함돼 있다. [DeepPCB 공식 설명](https://github.com/tangsanli5201/DeepPCB)
- PKU 결함 이미지 693장의 XML 파일명과 크기가 실제 이미지와 일치했다. VOC의 1부터 시작하는 양끝 포함 좌표는 `[xmin-1, ymin-1, xmax, ymax)`로 변환한다. DeepPCB의 0부터 시작하는 끝점도 내부에서는 오른쪽·아래쪽을 제외하는 좌표로 통일한다. 두 출처 모두 결함 종류는 분류 타깃에서 합친다. [PKU 공식 데이터 목록](https://robotics.pkusz.edu.cn/resources/dataset/) · [데이터 제작 논문](https://arxiv.org/abs/1901.08204)
- 이 작업 공간의 `PCB-Defect An Annotated Dataset for Surface Defect`는 230장에 COCO·회전 박스를 제공하는 별도 배포본이다. B·C에는 요청한 PKU XML과 10장의 정상 템플릿이 있는 `PCB_DATASET`을 사용했다. [로컬 배포 설명](<../out_data/PCB-Defect An Annotated Dataset for Surface Defect/PCB-Defect/README.txt>)
- VisA는 정상 폴더를 0, 이상 폴더를 1로 읽는다. 원본 마스크의 0은 배경이며 1을 비롯한 모든 양수는 이상이다. 마스크 크기 불일치, 누락, 중복 연결, 빈 이상 마스크를 검사한다. [VisA 공식 데이터 구조](https://github.com/amazon-science/spot-diff)

## 타일과 분할

먼저 원본을 seed와 보드 그룹으로 70%·15%·15% 목표 비율에 배정한다. VisA는 PCB 종류와 정상·불량별로 원본 ID를 나눈다. DeepPCB의 공식 두 목록은 원본 확인에 사용하며, 보드 전체를 함께 보관하기 위해 이번 프로젝트의 분할을 새로 고정한다. 그룹 수와 크기에 따라 실제 비율은 달라진다. 동일 원본 ID, 보드 그룹, 원본 파일 SHA-256의 분할 간 중복, 중복 타일, 타일과 부모 분할 불일치를 자동 검사한다. 회전 복제본은 수집에서 제외하고 무작위 수평·수직 반전은 학습 배치에서만 만든다. 해시 검사는 바이트가 같은 파일을 판별한다. 알려지지 않은 보드 계보 및 재인코딩된 유사 사진은 별도의 식별 정보가 필요하다.

DeepPCB는 640×640 원본 전체가 한 샘플이다. PKU와 VisA는 기본 640×640 창을 512픽셀씩 이동한다. 마지막 창을 가장자리에 맞추므로 끝부분의 겹침은 128픽셀보다 커질 수 있다. 짧은 축은 실제 이미지 범위만 자른다. **네이티브 해상도에서 먼저 자른 다음** RGB로 구성한 타일을 모델 입력 크기로 리사이즈한다. 회색조는 3채널로 복제한다. 원본에 그린 박스나 마스크는 모델에 들어가지 않는다.

각 결함에 대해 타일 안에 포함된 결함 면적을 그 결함의 전체 면적으로 나눈다. 박스는 직사각형 픽셀 면적, 마스크는 8방향으로 연결된 각 이상 성분의 실제 전경 픽셀 수를 사용한다. 결함이 여러 개면 가장 높은 비율을 택하므로 작은 결함 한 개가 온전히 들어오면 불량 타일이 된다.

| 조건 | 타일 처리 |
| --- | --- |
| 정상 원본 또는 겹친 결함 픽셀 0개 | normal |
| 겹친 픽셀이 있고 최대 포함 비율 ≥ `min_defect_coverage` (기본 0.5) | defect |
| 겹친 픽셀이 있고 최대 비율이 기준 미만 | 타일 지도학습과 타일 loss에서 제외 |

`tiles.csv`는 모든 창의 부모, 분할, 좌표, `overlap_px`, `region_px`, `overlap_fraction`을 보관한다. `label=-1`은 제외 표시다. 원본 validation·test·사용자 추론에서는 **모든 창**을 평가해 최대 점수를 구한다. 정답 위치로 평가 타일을 골라내면 실제 추론과 다른 점수가 생긴다.

## 확인한 표본 수

| 출처 | 정상 원본 | 불량 원본 | 분할 그룹 | 생성한 창 | 정상 타일 | 불량 타일 | 부분 겹침 제외 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepPCB | 1,500 | 1,500 | 11 | 3,000 | 1,500 | 1,500 | 0 |
| PKU | 10 | 693 | 10 | 16,990 | 11,856 | 4,374 | 760 |
| VisA PCB1~PCB4 | 4,016 | 400 | 4,416 | 26,496 | 25,156 | 1,185 | 155 |

기본 설정으로 원본 8,119장에서 창 46,486개를 만들었다. 이 중 45,571개에 지도학습용 이진 라벨을 부여했다. 공식 목록에 없는 DeepPCB 템플릿 1장과 PKU 회전 이미지 693장은 원본 모집단에서 제외했다. DeepPCB에는 바이트가 같은 템플릿의 추가 사본 2장이 같은 그룹·같은 분할에 있다. [분할별 집계](artifacts/prepared/counts.csv) · [누수·제외 보고서](artifacts/prepared/summary.json)

시각 검증에서 PKU 결함 박스가 3,276/4,221픽셀 들어온 창은 불량, 1,265/2,970픽셀 들어온 창은 제외였다. VisA의 부분 마스크 예시는 387/4,769픽셀로 제외됐다. 미리보기의 청록색은 창, 분홍색은 정답이다. [DeepPCB 샘플](artifacts/prepared/previews/DeepPCB.png) · [PKU 샘플](artifacts/prepared/previews/PKU.png) · [VisA 샘플](artifacts/prepared/previews/VisA.png)

## 학습과 평가

| 실험 | 가중치 업데이트에 쓰는 출처 | 체크포인트·임계값 선택 | 최종 평가 |
| --- | --- | --- | --- |
| A | DeepPCB | 공통 validation 원본 | 공통 test 원본 |
| B | DeepPCB + PKU | 동일 | 동일 |
| C | DeepPCB + PKU + VisA PCB1~PCB4 | 동일 | 동일 |

이 분할의 validation은 1,344장, test는 1,216장이다. A·B도 체크포인트와 임계값을 고를 때 PKU·VisA를 포함한 validation을 본다. 따라서 미학습 출처에 대한 결과를 해석할 때 이 조건을 고려해야 한다.

각 모델은 검증 원본의 이진 로그 손실이 가장 낮은 epoch를 저장한다. 학습 loss는 선택한 타일 가중치를 반영한다. 검증 타일 loss는 라벨이 있는 타일의 비가중 로그 손실이고, 검증 원본 loss는 전체 창을 집계한 원본당 손실이다. 클래스 가중치·샘플러는 각 실험의 학습 타일 빈도에서만 계산한다.

운영 임계값은 최적 체크포인트의 validation 원본에서 선택한다. 기본 목표는 불량 recall ≥ 0.90, 정상 FPR ≤ 0.05다. 가능한 후보 중 recall이 높고 FPR이 낮은 순으로, 동률이면 0.5에 가까운 순으로 선택한다. 두 목표를 동시에 달성할 수 없으면 FPR 상한 안에서 recall을 최대화하고 미달 사유를 기록한다. 상한 자체를 달성할 수 없으면 최소 FPR을 사용한다. 검증에 한 클래스만 있으면 0.5를 유지한다. 최종 모델 파일에는 클래스 매핑, 입력·타일 설정, 운영 임계값, 선택 근거가 함께 들어 있다.

`evaluate`는 저장된 체크포인트마다 test를 한 번 추론하고 그 예측으로 0.5와 운영 임계값 결과를 모두 계산한다. 재실행하면 저장 파일·모델·분할 식별자를 검증한 뒤 결과를 재사용한다. 노트북 또한 저장된 예측만 읽는다.

양성은 불량이다. 혼동행렬 행은 실제, 열은 예측이며 순서는 normal, defect다. FPR은 FP/(TN+FP), 미검출률 FNR은 FN/(TP+FN)이다. PR-AUC는 recall을 가로축, precision을 세로축으로 한 **사다리꼴 면적**이며 시작점은 (0, 1)이다. 두 AUC는 정상·불량 모두가 있을 때만 계산한다. 분모가 0인 지표는 CSV에서 빈칸으로 두고 `undefined` 열에 지표명을 남긴다. 각 표에 실제 정상·불량 수를 함께 기록한다. 이번 test의 DeepPCB는 보드 그룹 1개, PKU 정상은 2장뿐이라 출처별 수치의 변동이 클 수 있다.

### 이번 실행의 시험 결과

공통 test에는 정상 820장과 불량 396장이 들어 있다. 아래 수치는 세 모델의 실제 저장 예측에서 계산했다.

| 모델 | 기본 0.5 F1 | 기본 recall | 기본 FPR | 운영 임계값 | 운영 F1 | 운영 recall | 운영 FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.420 | 67.7% | 74.5% | 0.975065 | 0.619 | 47.7% | 3.17% |
| B | 0.613 | 93.4% | 53.8% | 0.998379 | 0.821 | 70.5% | 0.61% |
| C | 0.947 | 96.2% | 3.41% | 0.948927 | 0.903 | 83.1% | 0.49% |

C의 validation에서 두 운영 목표를 동시에 달성하는 임계값은 없었다. FPR 상한을 지킨 선택점은 validation recall 85.4%, FPR 4.98%였다. test에서는 기본 임계값의 오탐 28장·미검출 15장이 운영 임계값에서 오탐 4장·미검출 67장으로 바뀌었다. 이 선택은 test를 열기 전에 확정했다.

C의 운영 임계값에서 불량 검출은 DeepPCB 196/216장, PKU 107/120장, VisA 26/60장이었다. VisA의 PCB1·PCB2·PCB3·PCB4는 각각 15장 중 5·11·5·5장을 검출했다. PKU는 정상 2장 중 1장을 오탐했다. [A 전체·출처별 지표](artifacts/runs/A/test/metrics.csv) · [B 전체·출처별 지표](artifacts/runs/B/test/metrics.csv) · [C 전체·출처별 지표](artifacts/runs/C/test/metrics.csv)

## 설정과 산출물

- `seed`, `split_fractions`, `tiling.size`, `tiling.overlap`, `tiling.min_defect_coverage`로 데이터 분할과 라벨 정책을 설정한다.
- `training`에서 모델(`resnet18`, `efficientnet_b0`), 사전학습, 입력 크기, 배치, epoch, 학습률, AdamW/SGD, 증강, 가중치 감쇠, CPU/CUDA, 혼합 정밀도와 로더 워커 수를 바꿀 수 있다. 기본값은 ImageNet 사전학습 ResNet18, 224×224 **타일 입력**, 배치 48, 3 epoch다. CUDA가 없으면 CPU로 실행한다. 사전학습 가중치가 캐시에 없고 오프라인이면 `pretrained`를 `false`로 설정한다.
- `training.imbalance`는 `none`, `class_weight`, `weighted_sampler` 중 하나다. RGB가 기본이다. 이진화 비교는 설정을 복사해 `preprocessing`을 `binary`로, `output_dir`을 예를 들어 `artifacts_binary`로 정한 뒤 같은 seed로 실행한다. 별도 설정은 각 명령에 `--config src2/파일명.json`으로 전달한다.
- 준비 결과와 완료된 모델은 재실행 시 검증 후 재사용한다. 데이터 내용·설정이 바뀐 새 실험은 별도 `output_dir`에서 `01_prepare → 02_train → 03_evaluate`를 실행한다. `cache_inputs=true`는 타일을 자른 뒤 리사이즈한 RGB를 시스템 임시 폴더 `deeppcb-src2-rgb-v1`에 무손실 배열로 보관한다. 현재 전체 데이터·224 입력에서 약 6.5 GiB다. `false`로 두면 같은 좌표에서 즉시 잘라 읽는다.

| 위치 | 내용 |
| --- | --- |
| `artifacts/prepared/` | 원본·타일 목록, 실제 사용 경로·seed·분할 제약, 클래스별 집계, 누수 검사, 원본/타일/정답 미리보기 |
| `artifacts/runs/{A,B,C}/` | `best.pt`, `config.json`, `classes.json`, `splits.csv`, `run.json`, `history.csv`, `learning_curves.png`, `threshold.json` |
| 각 실험의 `validation/` | 원본별 예측, 두 임계값의 출처별 지표와 검증 임계값 후보표 |
| 각 실험의 `test/` | 동일 test 원본 목록에 대한 확률·예측, 출처별/PCB별 지표·혼동행렬 원소, 분할·모델·결과 해시 |
