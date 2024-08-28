import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用微软雅黑字体

# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = 'C:\\Users\\admin\\Desktop\\ZSRF.csv'  # 替换成自己的文件路径
data = pd.read_csv(file_path, encoding='gb18030')

# 分割特征和标签
X = data.iloc[:, 1:]  # 特征列
y = data.iloc[:, 0]  # 标签列

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化模型
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True,max_iter=1000),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}


# 执行10折交叉验证并打印评分
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10)
    print(f'{name} 10折交叉验证的平均准确率: {np.mean(scores)}')

# 绘制ROC曲线
plt.figure(figsize=(6, 6))
for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_proba = model.predict_proba(X_test)[:, 1]  # 仅针对有predict_proba方法的模型

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# 绘制随机猜测线
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')

# 设置图形属性
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 绘制混淆矩阵
for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 绘制混淆矩阵图
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 移除cbar=False，以显示颜色条形图
    plt.xlabel('Prediction label')
    plt.ylabel('Real label')
    plt.title(f'{name} ')
    plt.show()