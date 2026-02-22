import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

# 1. 임의의 신용카드 데이터셋 생성 (X1, X2)
# X1: 거래 금액 점수 (정규화된 값)
# X2: 평소 거래 위치와의 거리 점수 (정규화된 값)
np.random.seed(42)
num_samples = 1000

# 기본 특징 2개 생성 (-1 ~ 1 사이의 임의의 값)
X1 = np.random.uniform(-1, 1, num_samples)
X2 = np.random.uniform(-1, 1, num_samples)

# 타겟 데이터 (Y): 신용카드 사기 여부 (0: 정상, 1: 불법 사용)
# 임의의 비선형 규칙 생성 (X1과 X2가 모두 높거나, 복잡한 패턴일 때 사기로 간주)
Y = (X1**2 + X2**2 > 0.5).astype(int)

# 2. 특징 공학 (Feature Engineering) - 이미지의 입력 노드 4개 구현
# 이미지에서 선택된 입력 노드: X1, X2, X1^2, X1*X2
X1_sq = X1 ** 2
X1_X2 = X1 * X2

# 모델에 넣기 위해 4개의 특징을 하나의 배열로 합침 (shape: [1000, 4])
X_train = np.column_stack((X1, X2, X1_sq, X1_X2))

# 3. 이미지와 동일한 신경망 구조 구축
def build_playground_model():
    model = models.Sequential()
    
    # 입력층 (4개의 특징) 및 첫 번째 은닉층 (뉴런 2개, ReLU)
    model.add(layers.Dense(2, activation='relu', input_shape=(4,), name="Hidden_Layer_1"))
    
    # 두 번째 은닉층 (뉴런 1개, ReLU)
    # 참고: 이미지상 2번째 은닉층이 1개의 뉴런을 가지며 ReLU를 사용합니다.
    model.add(layers.Dense(1, activation='relu', name="Hidden_Layer_2"))
    
    # 실무적 보정 (출력층): 
    # 플레이그라운드에서는 맵의 색상으로 분류를 보여주지만, 
    # 실제 코딩에서는 0~1 사이의 확률값으로 만들기 위해 Sigmoid 출력층이 하나 더 필요합니다.
    model.add(layers.Dense(1, activation='sigmoid', name="Output_Layer"))
    
    return model

model = build_playground_model()
model.summary()

# 4. 모델 컴파일 (이미지 설정 반영)
# 학습률 0.03 설정
optimizer = optimizers.SGD(learning_rate=0.03) 
# 또는 최적화를 위해 optimizers.Adam(learning_rate=0.03) 사용 가능

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. 모델 학습 (테스트)
print("\n--- 학습 시작 ---")
history = model.fit(X_train, Y, epochs=100, batch_size=10, verbose=1)

# revized 260222: test

# branch commit test
