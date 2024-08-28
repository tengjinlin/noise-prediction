# noise-prediction
# This study introduces various machine learning methods and applies the Random Forest algorithm, which performed best, to investigate noise suitability in the central urban area of Nanchang City. 
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
# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用微软雅黑字体
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
# 初始化评分结果存储，增加 'Accuracy'
scores = {
    'Model': [],
    'Set': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'Accuracy': []  # 新增准确率存储
}

# 训练每个模型，并计算训练集和测试集的评分
for name, model in models.items():
    # 创建一个包含预处理步骤的管道
    pipeline = make_pipeline(StandardScaler(), model)
    # 训练模型
    pipeline.fit(X_train, y_train)

    # 训练集预测
    y_train_pred = pipeline.predict(X_train)
    # 测试集预测
    y_test_pred = pipeline.predict(X_test)

    # 计算训练集和测试集的准确率
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # 计算训练集的其他分数
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    # 计算测试集的其他分数
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # 存储所有分数
    scores['Model'].extend([name] * 2)
    scores['Set'].extend(['Training', 'Test'])
    scores['Precision'].extend([train_precision, test_precision])
    scores['Recall'].extend([train_recall, test_recall])
    scores['F1 Score'].extend([train_f1, test_f1])
    scores['Accuracy'].extend([train_accuracy, test_accuracy])  # 存储准确率

# 将结果转换为 DataFrame 以方便显示和绘图
score_df = pd.DataFrame(scores)

# 使用 Pandas 的功能输出扩展的表格结果
print(score_df.pivot(index='Model', columns='Set', values=['Precision', 'Recall', 'F1 Score', 'Accuracy']))

# 执行10折交叉验证并计算Precision、Recall、F1分数和Accuracy
for name, model in models.items():
    y_pred = cross_val_predict(model, X, y, cv=10)
    cv_precision = precision_score(y, y_pred)
    cv_recall = recall_score(y, y_pred)
    cv_f1 = f1_score(y, y_pred)
    cv_accuracy = accuracy_score(y, y_pred)  # 计算准确率
    print(f'{name} 模型的Precision: {cv_precision}')
    print(f'{name} 模型的Recall: {cv_recall}')
    print(f'{name} 模型的F1分数: {cv_f1}')
    print(f'{name} 模型的Accuracy: {cv_accuracy}')  # 输出准确率


# 用于绘制学习曲线的函数
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(5, 5))

    axes.set_title(title, pad=5)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Number of training samples", labelpad=5)
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, color="gray", alpha=0.1)
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, color="gray", alpha=0.1)
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross validation score")
    axes.legend(loc="best")

    return plt

# 设定交叉验证的折数
cv = 10

# 绘制每个模型的学习曲线
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
for i, (name, model) in enumerate(models.items()):
    ax = axes.flatten()[i]
    plot_learning_curve(model, name, X, y, axes=ax, ylim=(0, 1.01), cv=cv, n_jobs=4)

plt.tight_layout()
plt.subplots_adjust(hspace=0.25)  # 增加垂直空间
plt.show()

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
