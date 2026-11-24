# DeepPCB 정상 템플릿 쌍입력 불량 검출

정렬된 640×640 정상 템플릿과 검사 이미지를 받아 불량의 박스, 종류, 점수를 예측합니다. 같은 공식 분할을 쓰는 **속도형(XOR 후보 + CNN)**과 **정밀형(쌍입력 Faster R-CNN)**을 비교합니다. Python 3.10 이상, PyTorch와 torchvision의 호환 버전이 필요합니다.

## 데이터 배치와 설치

`DeepPCB.zip`을 풀면 `deeppcb/` 폴더가 생깁니다. 별도로 받은 DeepPCB 데이터 ZIP(`data.zip` 또는 `PCBData.zip`)을 **프로젝트 루트에서** 풀어 아래처럼 놓으세요. 데이터 원본은 이 소스 ZIP에 들어 있지 않습니다.

```text
deeppcb/
  data/
    trainval.txt
    test.txt
    group00041/...
  configs/
  src/
  README.md
  requirements.txt
```

데이터 ZIP에 최상위 `data1` 폴더가 들어 있으므로, `deeppcb/`에서 압축을 풀면 됩니다. 공식 목록의 첫 필드는 접미사 없는 기본 JPG 경로이고, 두 번째 필드는 다른 폴더에 있는 라벨 TXT 경로입니다. 01은 각각 `_temp.jpg`, `_test.jpg`, `.txt`를 연결합니다. 이 프로젝트는 첨부된 실제 구조(`data1`, `data1`, `group*/<id>/*_temp.jpg`, `*_test.jpg`, `group*/<id>_not/*.txt`)에 맞춰 작성됐습니다. 불필요한 추가 JPG는 공식 목록에 없으면 무시합니다.

프로젝트 루트에서 다음을 실행합니다. Windows 명령 프롬프트 기준입니다.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CPU 환경에서는 `device: cpu`로 설정하거나 `auto`를 그대로 둡니다. CUDA를 지원하는 PyTorch 빌드와 GPU가 설치된 환경에서 `device: cuda` 또는 `auto`를 사용합니다. PyTorch 설치 방식은 장치와 운영체제에 맞는 빌드를 선택하세요. CPU 및 CUDA 모두 동일한 소스 코드로 실행합니다.

## 전체 실행 순서

반드시 `deeppcb/`에서 실행하세요. 01과 02는 학습 전에 실행하며, 03·04가 각각 체크포인트를 생성합니다.

```bat
python src/01_prepare_data.py
python src/02_check_candidates.py
python src/03_train_fast.py
python src/04_train_accurate.py
python src/05_calibrate_on_val.py --max-fp-per-image 1
python src/06_evaluate_on_test.py
```

03·04의 실행 순서는 서로 바꿔도 됩니다. `--epochs 1 --limit-train 2 --limit-val 2`는 짧은 연결 점검에 쓸 수 있습니다. 05에는 `--limit-val 2`, 06에는 `--limit-test 2 --benchmark-pairs 2 --warmup 1` 옵션이 있습니다. **일부 val로 보정했다면 전체 test 평가는 막힙니다.** 최종 비교 전에는 제한 옵션 없이 03~06을 실행하세요. 작은 학습 결과는 모델 성능을 뜻하지 않습니다.

## 데이터와 모델의 규칙

- 공식 `trainval.txt` 1,000쌍을 시드 42로 train 800 / val 200으로 분리하고, 공식 `test.txt` 500쌍은 마지막에만 평가합니다. 두 모델은 `runs/splits.csv`를 공유합니다. 01은 파일 누락, 크기, 중복 ID, `x1 y1 x2 y2 class_id` 좌표 및 클래스 값을 모두 확인하고 검사 영상의 박스 미리보기를 만듭니다. 라벨 구분자는 공백과 쉼표 모두 허용합니다. 박스는 `[x1, y1, x2, y2]` 반열림 좌표로 처리합니다.
- 공식 클래스는 `1=open`, `2=short`, `3=mousebite`, `4=spur`, `5=spurious copper`, `6=pin hole`입니다. 속도형의 `0`은 배경입니다. train과 val에서 여섯 클래스가 모두 존재하는지 검사합니다. 템플릿 계열 그룹 분할 및 분할 간 유사 템플릿 검사는 이번 구현에 포함되지 않습니다.
- 원본 전체 영상은 640×640, 회색조 값을 유지합니다. 블러나 전체 영상 축소는 없습니다. 후보용 이진화에만 임계값을 적용합니다. 분류 CNN에는 가변 크기 **후보 패치만** 96×96으로 맞춰 넣습니다. 증강 기본값은 `false`이며, 켜면 양쪽 영상과 박스에 같은 90도 단위 회전·뒤집기를 적용합니다.
- 02는 val의 정답 박스가 **분류 패치**에 온전히 들어간 비율, **출력 후보**와 IoU > 0.33을 만족하는 비율을 전체와 클래스별로 기록합니다. 이미지당 후보 수 평균과 p95도 기록합니다. 추가 후보 병합은 하지 않습니다. 출력 박스는 연결 요소에 `output_margin`을 더하고, 분류 패치는 그 박스에 `context_margin`을 추가합니다. 후보 단계에서 구리 추가·소실 마스크를 따로 만듭니다. `copper_polarity: dark`는 어두운 쪽을 구리로 간주하며, 반대 영상은 `light`로 바꿉니다.
- 속도형은 `[검사 영상, 템플릿, 추가 마스크, 소실 마스크]` 패치로 7종(배경 + 6종) 확률을 예측합니다. 학습 라벨은 출력 후보 박스와 GT 박스의 최고 IoU가 **0.33 초과**일 때 해당 클래스입니다. 최종 좌표는 더 큰 분류 패치 좌표가 아닌 출력 후보 좌표입니다. 클래스별 NMS를 사용합니다.
- 정밀형은 두 전체 영상의 특징을 **공유 ResNet18**로 각각 뽑고 `[검사, 템플릿, 절댓값 차이]`를 결합합니다. stride 4의 고해상도 특징을 포함한 FPN과 8픽셀부터 시작하는 작은 앵커를 쓰는 Faster R-CNN입니다. 사전학습 가중치 다운로드나 XOR 입력 조건은 없습니다.

## 설정

`configs/fast.yaml`과 `configs/accurate.yaml`에서 데이터 위치, 분할 CSV, 결과 폴더, 장치, 학습률·에폭·증강 여부 등을 바꿀 수 있습니다. 초기 후보 설정은 임계값 **128**, 최소 연결 요소 면적 **16픽셀**, 출력 박스 여백 **8픽셀**, 추가 문맥 여백 **12픽셀**입니다. 후보 설정이나 분할 CSV를 바꾸었다면 02를 다시 실행한 뒤 03을 재학습하세요. 체크포인트에는 학습 당시 모델·후보 설정과 분할 해시를 기록하고, 05·06은 해당 설정으로 추론합니다.

## 검증, 비교, 결과 파일

05는 val의 **같은 이미지당 오탐 상한**(기본 1)에 대해 각 모델의 0.00~1.00 점수 임계값을 0.01 간격으로 시험합니다. 임계값 1에서도 포화 점수로 오탐이 남을 때를 대비해 1.000001을 포함합니다. 상한을 만족하는 값 중 F1이 가장 높은 값을 선택합니다. 06은 **그 임계값을 고정**해 공식 test를 평가합니다. val 상한을 test에 재적합하지 않으므로 실제 test 오탐은 상한과 다를 수 있습니다.

운영 지표인 TP/FP/FN, 재현율, F1의 매칭은 같은 클래스의 IoU ≥ 0.33이며 한 정답은 한 예측에만 대응합니다. 작은 박스는 **면적 < 32×32 픽셀**입니다. 클래스별 재현율과 작은 박스 재현율도 기록합니다. mAP@0.33과 mAP@0.50:0.95는 **운영 임계값 적용 전** 점수 순위, 클래스별 AP, 101점 재현율 보간을 사용합니다. AP와 NMS는 이 프로젝트에서 구현했으므로 공식 DeepPCB/COCO 평가 스크립트와 구현 세부사항에 따라 결과가 다를 수 있습니다.

속도 측정은 같은 장치에서 test의 동일한 순서 이미지 쌍을 한 번에 한 쌍씩 처리합니다. 기본 3회 워밍업 후 20쌍을 **미리 메모리에 로드**하고, 이진화·후보·패치 또는 전체 영상 전처리·모델 추론·후처리까지 포함해 중앙값과 p95(ms)를 기록합니다. 디스크 읽기는 제외하며 CUDA는 전후 동기화합니다. `--benchmark-pairs`, `--warmup`, `--device`로 실행을 조절할 수 있습니다.

```text
runs/
  splits.csv                 # 공통 800/200/500 분할
  data_report.json           # 분할별 클래스 수
  previews/*.png             # 01: 검사 영상 + 정답 박스
  candidate_report.json      # 02: 전체/클래스별 후보 커버리지
  candidate_previews/*.png   # 02: 청록 후보 박스, 빨강 정답 박스
  fast.pt, accurate.pt       # 03, 04: 모델 체크포인트
  val_raw_*.csv              # 05: 임계값 이전 val 예측
  calibration.json           # 05: 각 모델의 val 임계값
  test_raw_*.csv             # 06: 임계값 이전 test 예측(AP 계산용)
  test_predictions_*.csv     # 06: 임계값 적용 후 최종 검출
  test_metrics.json          # 06: 오탐·재현율·F1·mAP·속도 비교
```

모든 예측 CSV의 열 순서는 `sample_id,x1,y1,x2,y2,class_id,score`입니다. 테스트 전에 `runs/`의 작은 연결 점검 결과를 남겨두었다면, 전체 학습 및 val 보정을 다시 실행하면 덮어써집니다.
