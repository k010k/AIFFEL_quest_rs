import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import minmax_scale
import calendar
from datetime import datetime
#############################
# (1) 데이터 가져오기
# 터미널에서 ~/data/data/bike-sharing-demand 경로에 train.csv 데이터를 train 변수로 가져 옵니다.
# train = pd.read_csv('~/datasets/bike-sharing-demand/train.csv')
train = pd.read_csv('~/data/data/bike-sharing-demand/train.csv')
#############################
#############################
# (2) datetime 컬럼을 datetime 자료형으로 변환하고 연, 월, 일, 시, 분, 초까지 6가지 컬럼 생성하기
# datetime 컬럼이 문자열인 경우 변환
train['datetime'] = pd.to_datetime(train['datetime'])
train['year']=train[ "datetime"].dt.year
train['month']=train[ "datetime"].dt.month
train['day']=train[ "datetime"].dt.day
train['hour']=train[ "datetime"].dt.hour
train['minute']=train[ "datetime"].dt.minute
train['second']=train[ "datetime"].dt.second
#############################
#############################
# (3) year, month, day, hour, minute, second 데이터 개수 시각화하기
# sns.countplot 활용해서 시각화하기
# subplot을 활용해서 한 번에 6개의 그래프 함께 시각화하기
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
sns.countplot(x='year', data=train, ax=axes[0, 0])
sns.countplot(x='month', data=train, ax=axes[0, 1])
sns.countplot(x='day', data=train, ax=axes[1, 0])
sns.countplot(x='hour', data=train, ax=axes[1, 1])
sns.countplot(x='minute', data=train, ax=axes[2, 0])
sns.countplot(x='second', data=train, ax=axes[2, 1])

plt.show()
#############################
#############################
# X, y 컬럼 선택 및 train/test 데이터 분리
# 불 필요한 특성인 'datetime', 'casual', 'registered', 'count', 'minute', 'second' 제거 하여 pandas Dataframe 생성
# 2) 불 필요한 변환 제거 및 명시 적인 컬럼 지정
X_scaled = minmax_scale(train.drop(['datetime', 'casual', 'registered', 'count', 'minute', 'second'], axis=1))
feature_cols = train.drop(['datetime', 'casual', 'registered', 'count', 'minute', 'second'], axis=1).columns
X = pd.DataFrame(X_scaled, columns=feature_cols)


X_scaled = minmax_scale(train.drop(['datetime', 'casual', 'registered', 'count', 'minute', 'second'], axis=1))
X = pd.DataFrame(X_scaled, columns=train.drop(['datetime', 'casual', 'registered', 'count', 'minute', 'second'], axis=1).columns)
# count 값을 맞추고자 하므로, y 변수에 count 컬럼의 데이터 넣기
y = train['count']
# train_test_split 함수로 이용 하여 훈련, 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#############################
#############################
# (5) LinearRegression 모델 학습
# sklearn의 LinearRegression 모델 불러오기 및 학습하기
model = LinearRegression(positive=True)
model.fit(X_train, y_train)
#############################
#############################
# (6) 학습된 모델로 X_test에 대한 예측값 출력 및 손실함수값 계산
# 학습된 모델에 X_test를 입력해서 예측값 출력하기
y_pred = model.predict(X_test)
print("예측값:", y_pred)
# mse로 손실함수 값 계산하기 맟 츌룍
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
# rmse 손실함수 값 계산하기 맟 츌력
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, y_pred)
#############################
#############################
# (7) x축은 temp 또는 humidity로, y축은 count로 예측 결과 시각화하기
# Seaborn whitegrid 스타일 설정
sns.set(style="whitegrid")

plt.figure(figsize=(18, 6))

# 첫 번째 서브플롯: 온도(temp)에 대한 실제값과 예측값 시각화
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test['temp'], y=y_test, label='Actual', color='blue', alpha=0.6)
sns.scatterplot(x=X_test['temp'], y=y_pred, label='Predicted', color='red', alpha=0.6)
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.title('Actual vs Predicted Count by Temperature')
plt.legend()

# 두 번째 서브플롯: 습도(humidity)에 대한 실제값과 예측값 시각화
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test['humidity'], y=y_test, label='Actual', color='blue', alpha=0.5)
sns.scatterplot(x=X_test['humidity'], y=y_pred, label='Predicted', color='red', alpha=0.5)
plt.xlabel('Humidity')
plt.ylabel('Count')
plt.title('Actual vs Predicted Count by Humidity')
plt.legend()

plt.tight_layout()
plt.show()
