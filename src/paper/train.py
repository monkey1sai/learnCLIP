from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# 示例數據（指稱相似度特徵）
features = np.array([
    [0.8, 0.6],  # 正樣本
    [0.2, 0.1],  # 負樣本
    [0.7, 0.5],  # 正樣本
    [0.3, 0.2]   # 負樣本
])
labels = np.array([1, 0, 1, 0])  # 標籤：1為正樣本，0為負樣本

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 訓練分類器
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")
