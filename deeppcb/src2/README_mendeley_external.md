# Mendeley PCB-Defect v2 외부 평가

기존 `src2` ResNet18 체크포인트로 불량 PCB를 얼마나 검출하는지 원본 이미지 단위로 평가합니다. 기본 설정은 실험 C의 `artifacts/runs/C/best.pt`와 그 체크포인트의 validation 임계값을 사용합니다.

## 데이터와 주석

기본 경로는 `out_data/PCB-Defect An Annotated Dataset for Surface Defect` 아래 재주석판의 `train/images`와 `train/labels`입니다. 배포본의 폴더명이 `train`이며, 이 프로젝트에서는 해당 폴더를 외부 시험 데이터로 사용합니다. v2 YOLO OBB의 각 TXT 행에는 클래스 번호와 정규화된 네 모서리 좌표 8개가 있습니다. 같은 배포 폴더의 `PCB_Defect/PCB_Defect` 하위 COCO JSON은 v1 주석이므로, COCO 옵션을 사용한다면 해당 JSON과 짝이 맞는 v1 이미지 폴더를 함께 지정하세요. v2 공개 수치인 이미지 230장, 주석 2,506개와의 차이는 실제 감사 결과에 표시됩니다. [Mendeley v2](https://data.mendeley.com/datasets/vdj74sngvn/2)

## 실행

기존 프로젝트의 Python 환경을 활성화하고 `src2/requirements.txt` 의존성을 사용하세요. `deeppcb` 폴더에서 다음 명령을 실행합니다.

```powershell
python -m src2.06_mendeley_external_evaluate
```

기본 결과는 `src2/artifacts/external/mendeley_v2/C`에 저장됩니다. 다른 실험 체크포인트나 CPU가 필요하면 다음과 같이 지정합니다.

```powershell
python -m src2.06_mendeley_external_evaluate --checkpoint src2/artifacts/runs/A/best.pt --device cpu --output-dir src2/artifacts/external/mendeley_v2/A
```

데이터 감사만 수행하는 명령입니다.

```powershell
python -m src2.06_mendeley_external_evaluate --audit-only --output-dir src2/artifacts/external/mendeley_v2/audit
```

- `mendeley_external_config.json`의 상대 경로는 설정 파일 위치 기준입니다. 명령행 경로는 현재 작업 폴더 기준입니다.
- `--image-dir`, `--annotations`, `--annotation-format obb|coco`, `--config`, `--checkpoint`, `--device auto|cuda|cpu`, `--output-dir`, `--batch-size`로 변경할 수 있습니다. OBB 클래스 번호가 다른 배포본은 설정 파일의 `obb_classes`를 수정합니다.
- `threshold`를 `null`로 두면 체크포인트에 저장한 validation 임계값을 사용합니다. 이미 다른 내부 validation 기록에 고정된 임계값을 사용해야 할 때만 `--threshold`와 `--threshold-source`를 함께 지정합니다.
- 기존 전처리 메타데이터가 빠진 체크포인트를 사용할 경우 `summary.json`의 `missing_checkpoint_settings`를 확인합니다. 원래 학습 설정을 확인한 뒤 `checkpoint_metadata_overrides`에 해당 값을 넣을 수 있습니다. 키는 `training.image_size`, `training.preprocessing`, `tiling.size`처럼 점으로 구분합니다.
- 실행마다 비어 있는 출력 폴더를 사용합니다. 다른 폴더에 저장했다면 노트북 상단의 `RESULT_DIR`를 그 경로로 지정하세요.

## 결과

| 파일 | 내용 |
|---|---|
| `audit.json`, `audit_images.csv`, `audit_issues.csv` | 실제 이미지·주석 수, 연결 상태, 이미지 크기와 해시, 손상 파일 및 주석 문제 |
| `image_predictions.csv` | 이미지별 최고 타일 점수, 타일 수, 판정, 원본 주석 요약 |
| `tile_predictions.csv` | 타일 번호, 원본 좌표, 정상·불량 확률과 판정 |
| `summary.json` | 체크포인트 해시와 로드 상태, 전처리, 임계값, 재현율, 제외 수, 장치, 실행 시간 |
| `figures/` | 점수 분포, 검출·미검출 비교, 해당 그룹이 있을 때 원본 및 겹침 예시 |

`src2/mendeley_external_evaluation.ipynb`를 열어 저장된 결과를 확인합니다. 이 노트북은 CSV와 그래프를 읽습니다.

평가 분모는 읽을 수 있는 불량 이미지 수입니다. 누락·손상 파일은 감사에 기록되며, 주석이 누락된 읽을 수 있는 PCB도 데이터셋 정의에 따라 불량 이미지로 평가합니다. 주석은 위치를 읽는 데 사용합니다. 정상 이미지가 없으므로 TN, FP, specificity, 정상 오탐률은 산출할 수 없습니다. 모든 이미지를 불량으로 판정하는 기준선도 재현율 100%이므로, 결과는 불량 미검출을 확인하는 데 해석 범위를 둡니다.
