# 개발자용 초기 아이디어 메모 (Developer Initial Memo)

본 문서는 초기 브레인스토밍 채팅 기록을 바탕으로 작성된 프로젝트 기획 문서입니다.

## 핵심 아이디어 (Core Idea)
**Transverse Quantum Ising Model**의 물리량(헬름홀츠 자유 에너지(분배 함수), 자화, 비열, 상전이 등)을 확인하고, 이를 빠르고 정확하게 예측할 수 있는 **대리 모델(Surrogate Model)**을 구축하여 기존 정통 계산 방식(Exact Method)과 성능을 비교한다.

## 1. 정답지 (Ground Truth) 구축 방법론
대리 모델 학습 및 검증을 위한 정답 데이터는 시스템의 차원에 따라 다음과 같은 물리적 방법론으로 계산한다.
* **1차원 시스템**: 요르단-위그너 변환 (Jordan-Wigner Transformation)
* **2차원 시스템**: 양자 몬테카를로 (Quantum Monte Carlo)

## 2. 대리 모델 (Surrogate Model) 구조
* **모델 아키텍처**: MLP (Multi-Layer Perceptron) 기반 딥러닝 모델
* **입력 파라미터 (Input)**: 온도 (Temperature), 자기장 (Magnetic Field)
* **출력 파라미터 (Output)**: 자유 에너지 (Free Energy) 등 주요 물리량
* **검증 방식**: 구축된 테스트 데이터셋을 통해 기존 Exact Method 대비 **(1) 연산 성능(속도)** 및 **(2) 정확도(Accuracy)** 비교

## 3. 확장 아이디어 (Optional Extensions)
기본 대리 모델이 성공적으로 구축되거나, 난이도가 너무 낮을 경우 다음 방향 중 하나를 선택해 고도화한다.
* **A. 파라미터 스캔 확장**: 입력(Input) 파라미터를 추가하여 전체 파라미터 공간(Parameter Space)에 대한 스캔 진행.
* **B. 아키텍처 다각화**: 단순 지도학습(Supervised Learning) 구조와 물리 법칙을 내재화한 **PINN (Physics-Informed Neural Network)** 구조 간의 성능 및 학습 효율 비교.
* **C. 물질 역설계 (Inverse Design)**: 타겟 상전이 특성(Target Phase Transition)을 가지는 물질의 파라미터를 역추적하는 설계 진행.
