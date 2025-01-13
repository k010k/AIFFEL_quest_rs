import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
#############################
# (1) 데이터 가져오기
# sklearn.datasets의 load_diabetes에서 데이터를 가져 온다.
diabetes = load_diabetes()
# diabetes의 data를 df_X에, target을 df_y에 저장 한다.
df_X = diabetes.data
df_y = diabetes.target
#############################
#############################
# (2) 모델에 입력할 데이터 X 준비하기
# df_X에 있는 값들을 numpy array로 변환해서 저장 한다.
df_X = np.array(df_X)
#############################
#############################
# (3) 모델에 예측할 데이터 y 준비하기
# df_y에 있는 값들을 numpy array로 변환해서 저장 한다.
df_y = np.array(df_y)
#############################
#############################
# (4) train 데이터와 test 데이터로 분리하기
# train_test_split 함수로 이용 하여 훈련, 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
#############################
#############################
# (5) 모델 준비하기
# 가중치와 절편을 무작위로 초기화
W = np.random.rand(10)
b = np.random.rand()

def model(X, W, b):
    """
    선형 회귀 모델 함수
    :param X: 입력 데이터 (N x 10 array)
    :param W: 가중치 벡터 (10-element array)
    :param b: 절편 (scalar)
    :return: 예측값 (N-element array)
    """
    predictions = X.dot(W) + b
    return predictions
#############################
#############################
# (6) 손실함수 loss 정의하기
def MSE(a, b):
    """
    평균 제곱 오차를 계산하는 함수
    :param a: 예측값 (N-element array)
    :param b: 실제값 (N-element array)
    :return: 평균 제곱 오차 (scalar)
    """
    mse = ((a - b) ** 2).mean()
    return mse

def loss(X, W, b, y):
    """
    손실 함수를 계산하는 함수
    :param X: 입력 데이터 (N x 10 array)
    :param W: 가중치 벡터 (10-element array)
    :param b: 절편 (scalar)
    :param y: 실제값 (N-element array)
    :return: 손실 값 (scalar)
    """
    predictions = model(X, W, b)
    L = MSE(predictions, y)
    return L
#############################
#############################
# (7) 기울기를 구하는 gradient 함수 구현하기
def gradient(X, W, b, y):
    """
    경사 하강법을 위한 그래디언트를 계산하는 함수
    :param X: 입력 데이터 (N x 10 array)
    :param W: 가중치 벡터 (10-element array)
    :param b: 절편 (scalar)
    :param y: 실제값 (N-element array)
    :return: 가중치와 절편에 대한 그래디언트 (dW, db)
    """
    N = len(y)  # 데이터 포인트의 개수
    y_pred = model(X, W, b)  # 예측값 계산

    # 가중치 W에 대한 그래디언트 계산
    dW = 1/N * X.T.dot(y_pred - y)
    # 절편 b에 대한 그래디언트 계산
    db = 2 * (y_pred - y).mean()
    return dW, db
#############################
#############################
# (8) 하이퍼 파라미터인 학습률 설정하기
LEARNING_RATE = 0.05
#############################
#############################
# (9) 모델 학습하기
losses = []
MAX_ITER = 10000

for i in range(1, MAX_ITER + 1):
    dW, db = gradient(X_train, W, b, y_train)
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(X_train, W, b, y_train)
    losses.append(L)

    # iteration마다 loss값 확인하기
    if i % 100 == 0:
        print('Iteration %d : Loss %.4f' % (i, L))
#############################
#############################
# (10) test 데이터에 대한 성능 확인하기
prediction = model(X_test, W, b)
mse = loss(X_test, W, b, y_test)
print("MSE:", mse)
#############################
#############################
# (11) 정답 데이터와 예측한 데이터 시각화하기
plt.figure(figsize=(12, 6))

# 첫 번째 특성에 대한 실제값과 예측값을 그래프로 그림
plt.scatter(X_test[:, 0], y_test, label="y_test", color='blue', alpha=0.5)
plt.scatter(X_test[:, 0], prediction, label="prediction", color='red', alpha=0.5)

plt.xlabel("Feature 1")
plt.ylabel("Target Value")
plt.title("Diabetes Prediction vs Actual")
plt.legend()
plt.grid(True)
plt.show()

# 추가: 손실 함수의 변화 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(losses, label='Training Loss', color='green')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.legend()
plt.grid(True)
plt.show()