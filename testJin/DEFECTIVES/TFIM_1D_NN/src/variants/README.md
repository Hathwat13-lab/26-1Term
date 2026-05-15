# 실험 분기 설명

이 폴더는 troubleshooting 과정을 “실행 가능한 파일 분기” 형태로 보존합니다.

## `baseline_derivative_blind`

첫 번째 신경망 시도입니다.

- GELU MLP를 사용해서 ReLU의 2차 미분 문제는 피했습니다.
- 하지만 학습은 거의 자유에너지 `F` 값 자체에 집중됩니다.
- `chi` Hessian 패널티가 매우 약합니다.
- 예상 실패: `F` MSE는 좋아 보이지만, 상전이 근처에서 `chi = -d2F/dh2`가 너무 평평하거나 피크 위치가 어긋날 수 있습니다.

## `sobolev_critical`

문제 원인을 보고 수정한 troubleshooting 분기입니다.

- `h = 1` 근처 smooth critical feature를 추가했습니다.
- Swish activation을 사용했습니다.
- 낮은 온도 임계 영역을 derivative batch에서 더 자주 보게 했습니다.
- `chi`에 더 강한 Sobolev loss를 걸고, 약한 `Cv` penalty도 추가했습니다.
- 평가는 MSE뿐 아니라 MAE, normalized RMSE, correlation, 피크 위치 오차, 피크 높이 상대오차, 임계영역 metric을 함께 봅니다.

설명이나 비교가 필요할 때는 이 스냅샷들을 기준으로 보면 됩니다. 상위 `src/` 파일들은 앞으로 계속 바뀔 수 있습니다.
