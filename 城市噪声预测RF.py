import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 读取数据
file_path = 'C:\\Users\\admin\\Desktop\\ZSRF.csv'  # 替换成自己的文件路径
data = pd.read_csv(file_path, encoding='gb18030')

# 数据预处理

# 第一列是标签
y = data.iloc[:, 0]
# 剩下的列是特征
X = data.iloc[:, 1:]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 查看训练集和测试集的大小
print("训练集特征矩阵的大小:", X_train.shape)
print("测试集特征矩阵的大小:", X_test.shape)
print("训练集目标变量的大小:", y_train.shape)
print("测试集目标变量的大小:", y_test.shape)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100,random_state=42)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_classifier.predict(X_test)

#######模型在测试集上的评价指标值
print('验证集准确率：', accuracy_score(y_test, y_pred))
#####此处是三分类，因此需要加上average='micro'
#####print('\t验证集查准率：', precision_score(y_test, y_pred,average='micro'))
#####print('\t验证集召回率：', recall_score(y_test, y_pred,average='micro'))
#####print('\t验证集F1：', f1_score(y_test, y_pred,average='micro'))
# 计算Precision Score
precision = precision_score(y_test, y_pred, average='binary')  # 使用'macro'来计算未加权的均值
print(f"Precision Score: {precision}")
# 计算Recall Score
recall = recall_score(y_test, y_pred, average='binary')  # 使用'macro'来计算未加权的均值
print(f"Recall Score: {recall}")
# 计算F1 Score
f1 = f1_score(y_test, y_pred, average='binary')  # 使用'macro'来计算未加权的均值
print(f"F1 Score: {f1}")

# 预测结果图
y_test = y_test.to_numpy()
plt.figure(figsize=(5, 4))
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Pred')
plt.xlabel('预测样本')
plt.ylabel('预测结果')
plt.title('预测结果对比')
plt.legend()
plt.show()

# 获取特征的重要性
feature_importances = rf_classifier.feature_importances_

# 获取特征名称
feature_names = X_train.columns

# 对特征重要性进行排序
indices = np.argsort(feature_importances)[::-1]

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
bars = plt.bar(range(X_train.shape[1]), feature_importances[indices], color="r", align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])

# 在条形图上添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, round(yval, 4), ha='center', va='bottom')

plt.show()

# 计算准确率和生成分类报告
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(class_report)

# 假设 y_test 和 y_pred 已经定义
conf_matrix = confusion_matrix(y_test, y_pred)

# 使用Seaborn绘制混淆矩阵的热力图
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_curve, auc

# 首先，使用predict_proba方法获取测试样本属于各个类别的概率
y_scores = rf_classifier.predict_proba(X_test)

# 对于二分类问题，我们只关注正类的概率
y_scores = y_scores[:, 1]

# 计算ROC曲线的TPR和FPR
fpr, tpr, threshold = roc_curve(y_test, y_scores)

# 计算AUC值
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='navy', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([-0.05, 1.05]) # ROC曲线横坐标
plt.ylim([-0.05, 1.05]) # ROC曲线纵坐标
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print(f"ROC AUC: {roc_auc}")

import pandas as pd

# 读取新数据文件
X_new = pd.read_csv('C:\\Users\\admin\\Desktop\\NEW_ZSRF.csv', encoding='gb18030')  # 或其他适合你数据的编码

# 使用模型对新数据的分类概率进行预测
y_new_proba = rf_classifier.predict_proba(X_new)

# 打印预测概率
print(y_new_proba)

# 首先，将预测概率转换成DataFrame
proba_df = pd.DataFrame(y_new_proba, columns=rf_classifier.classes_)

# 然后，使用to_csv方法保存DataFrame到CSV文件
proba_df.to_csv('predicted_probabilities.csv', index=False)