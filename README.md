# DeepLearning_Project
# 🚀 CIFAR-10 이미지 분류 성능 한계 돌파 프로젝트

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)

## 📌 프로젝트 소개
본 프로젝트는 CIFAR-10 데이터셋을 활용하여 기본 CNN 모델의 한계를 분석하고, 데이터 증강(Data Augmentation) 및 정규화(Regularization) 기법을 단계적으로 적용하여 분류 정확도를 90% 이상으로 끌어올리는 것을 목표로 한 실험 저장소입니다.

* **개발 기간:** 202X.XX.XX ~ 202X.XX.XX
* **주요 목표:** 1. 과적합(Overfitting) 발생 원인 분석 및 해결
  2. 체계적인 Git Branch 전략을 통한 실험 이력 관리
  3. AI 어시스턴트(LLM)와의 협업을 통한 문제 해결 루틴 구축

---

## 🌿 Git Branch 전략 및 워크플로우
안전한 실험과 체계적인 버전 관리를 위해 목적에 따라 브랜치를 분리하여 작업했습니다.

* `main` : 항상 실행 가능하고 검증된 최종 베이스라인 코드
* `feature/data-aug` : 이미지 증강 기법(RandomFlip, Rotation 등) 도입 및 테스트
* `experiment/batch-norm` : 배치 정규화(Batch Normalization) 적용에 따른 가중치 분포 안정화 실험
* `feature/lr-scheduler` : Cosine Decay 학습률 스케줄러 적용

> **💡 협업 및 기록:** 각 `feature`, `experiment` 브랜치에서 의미 있는 성능 향상이 검증되면 Pull Request(PR)를 통해 `main` 브랜치로 병합(Merge)하며 개발 이력을 문서화했습니다.

---

## 📊 핵심 실험 결과 및 트러블슈팅

### 1. 베이스라인 모델의 한계 (과적합 발생)
* **문제:** 초기 CNN 모델 학습 결과, Train Accuracy는 95%에 달했으나 Validation Accuracy는 70% 부근에서 정체되며 `epoch_loss` 그래프가 심하게 진동함.
* **원인:** 모델이 훈련 데이터의 노이즈까지 암기하는 전형적인 과적합 상태.

### 2. 해결 과정 및 성능 변화
| 적용 기법 | Validation Accuracy | 주요 효과 및 분석 |
| :--- | :---: | :--- |
| **Baseline** | 70.2% | 심각한 과적합 발생 |
| **+ Data Augmentation** | 82.5% | 훈련/검증 Loss 간극 축소. 일반화 성능 향상 |
| **+ Batch Normalization** | 86.1% | `bias/histogram` 안정화, 학습 속도 가속 |
| **+ Dropout & LR Decay** | **91.3%** | 최종 90% 목표 달성. 최적점 부근에서 안정적 수렴 |

*(💡 팁: 여기에 텐서보드 그래프 스크린샷 이미지를 첨부하면 매우 좋습니다!)*
`![텐서보드 결과](이미지_링크주소)`

---

## 💡 회고 및 배운 점 (Retrospective)
1. **데이터의 중요성:** 복잡한 모델 구조를 설계하기 전, 데이터 증강을 통해 모델에게 다양한 패턴을 학습시키는 것이 성능 향상에 더 결정적인 역할을 한다는 것을 깨달았습니다.
2. **모니터링의 힘:** 텐서보드를 통해 가중치 분포와 Loss 개형을 분석하지 않았다면, 과적합의 원인을 정확히 짚어내기 어려웠을 것입니다. '감'이 아닌 '데이터' 기반의 튜닝을 경험했습니다.

---

## ⚙️ 실행 방법 (How to Run)
본 프로젝트를 로컬 환경에서 실행하는 방법입니다.

```bash
# 1. 저장소 클론
git clone [https://github.com/사용자이름/저장소이름.git](https://github.com/사용자이름/저장소이름.git)

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 모델 학습 및 평가 실행
python train.py
