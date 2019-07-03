import xgboost
import pandas as pd

raw = inputs[0]

# 종속 / 독립 골라내
X = raw[['KM','AGE']]
y = raw[['PRICE']]

# 모델 세팅
model = xgboost.XGBRegressor(learning_rate=0.1,
                             max_depth=5,
                             n_estimators=100) 

# 모델훈련
model.fit(X,y)

# model을 밖으로 빼낸다.
model.save_model("model.xgb")

# 결과 뽑기 
result = pd.DataFrame({ 'PRICE':raw['PRICE'], 'PREDICT' :model.predict(X) })