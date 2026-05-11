# Balanced peak guard 분기

## 목적

`sobolev_critical` 분기에서 생긴 문제를 줄이기 위한 분기입니다.

관찰된 문제:

- 낮은 온도에서 `chi` 피크 위치는 `h = 1` 근처로 좋아졌습니다.
- 하지만 `1/L -> 0` 열역학 극한 외삽에서 피크 높이가 과하게 커지는 overshoot가 생겼습니다.

## 수정 내용

1. critical feature를 조금 완만하게 만들었습니다.
   - rounding floor를 키웠습니다.
   - `log(r2)` feature의 스케일을 줄였습니다.
   - softplus sharpness를 낮췄습니다.
2. `chi_weight`를 `8e-2`에서 `4e-2`로 낮췄습니다.
3. 임계영역 derivative sampling 비율을 `0.70`에서 `0.55`로 낮췄습니다.
4. `inv_L = 0` 외삽에서 `chi`가 finite-size label보다 과하게 솟으면 패널티를 주는 guardrail loss를 추가했습니다.

## 의도

피크 위치를 완전히 포기하지 않으면서, 피크 높이 overshoot를 줄이는 균형점을 찾는 것입니다.
