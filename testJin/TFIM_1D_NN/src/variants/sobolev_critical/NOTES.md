# Sobolev critical 분기

## 수정 내용

1. `h = 1` 근처를 잘 표현하기 위한 smooth critical feature를 추가했습니다.
   예: `h - 1`, `(h - 1)^2`, thermal/finite-size rounding scale, `log((h - 1)^2 + rounding^2)`
2. activation은 Swish 계열을 사용해서 미분 가능성을 유지했습니다.
3. `chi` Sobolev loss를 `2e-3`에서 `8e-2`로 키웠습니다.
4. 낮은 온도 임계 영역을 derivative batch에 더 많이 샘플링했습니다.
5. 열역학 곡률을 더 일반적으로 학습하도록 작은 `Cv` penalty를 추가했습니다.

## 기대한 개선

- `chi` 피크 위치가 exact critical fan 쪽으로 가까워져야 합니다.
- holdout `chi MAE`와 임계영역 `chi` metric이 좋아져야 합니다.

## 남은 문제

- 낮은 온도의 열역학 극한에서 피크 높이가 overshoot될 수 있습니다.
- 다음 튜닝에서는 피크 위치뿐 아니라 피크 높이까지 동시에 안정화해야 합니다.
