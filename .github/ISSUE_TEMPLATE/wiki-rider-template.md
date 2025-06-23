---
name: Wiki-Rider Template
about: 금일 학습한 AI 이론 내용을 정리해 주세요!
title: "[날짜] [모듈명] (예시 : 250630 Machine Learning Basic)"
labels: ''
assignees: ''

---

## 📅 학습 날짜  
*예시: 2025-06-30*


## 📌 키워드 및 요약 정리
_키워드 예시: Tensor, Gradient Descent, Transformer, CNN 등_

- _용어 정의: 예) **Tensor**는 다차원 배열로, PyTorch에서 데이터를 표현하는 기본 단위입니다._
- _핵심 원리: 예) **Autograd**는 연산 그래프를 자동으로 생성하고 역전파로 미분값을 계산합니다._
- _시각적/직관적 설명: *예) 가중치가 클수록 그래디언트가 가팔라져 학습 속도가 빨라짐_
- _실습 예시/코드 간략 요약: 예) `torch.tensor()`를 사용해 2차원 텐서를 생성하고 `.backward()`로 미분 수행_


## 📘 개념 정리  

_예시 :_
### 1. Autograd의 동작 원리

PyTorch의 Autograd는 텐서 연산을 추적하는 **동적 계산 그래프**를 생성합니다.  
순전파(forward) 과정에서 연산이 기록되며, `.backward()` 호출 시 자동으로 그래디언트가 계산됩니다.

- 수식:

  $$
  y = 3x^2 + 2x + 1 \Rightarrow \frac{dy}{dx} = 6x + 2
  $$

- PyTorch 코드 예시:
  ```python
  import torch

  x = torch.tensor(2.0, requires_grad=True)
  y = 3 * x**2 + 2 * x + 1
  y.backward()
  print(x.grad)  # tensor(14.)
  ```
  
### 2. Autograd가 비선형 함수에서 그래디언트를 계산하는 방식

비선형 함수 예: ReLU(x) = max(0, x)
이 경우에도 Autograd는 각 구간에서의 도함수를 계산하고 연결하여 연산합니다.

시각적 설명: ReLU는 0 이하에서는 기울기가 0, 0 초과에서는 1로 작용

역전파 시, 0인 입력에 대해서는 그래디언트가 흐르지 않음 → 죽은 뉴런 문제



## ❓ 어려웠던 부분  
_예시:_
- _Autograd 연산 그래프에서 비선형 함수의 그래디언트 계산 방식이 헷갈렸습니다._  
- _Softmax와 CrossEntropyLoss가 왜 함께 쓰이는지 직관적으로 잘 이해되지 않았습니다._



## 💬 질문 또는 토론 유도  
_예시:_
- _파라미터 초기화 방식이 학습 성능에 어떤 영향을 주나요?_
- _CNN의 커널 사이즈를 줄이면 어떤 효과가 생길까요?_



## 🔗 참고 자료 (선택)  
_예시:_
- _[PyTorch 공식 문서 - Autograd](https://pytorch.org/docs/stable/autograd.html)_
- _CS231n 강의 노트 - Backpropagation 섹션_
