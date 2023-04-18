#!/usr/bin/env python
# coding: utf-8

# 一.导入数据，数据预处理

# In[9]:


#读取7万的数据

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

loan_data = pd.read_excel("LendingClubLoans1.xlsx")
print(loan_data.shape)


# In[10]:


#选取属性
#1.主观分析：19个维度+1grade（标签）
new_loan_data = pd.DataFrame(loan_data, columns=["loan_amnt","term","home_ownership","annual_inc",
                                                 "delinq_2yrs","open_acc","pub_rec","total_acc",
                                                 "total_rev_hi_lim","avg_cur_bal","mort_acc",
                                                 "inq_last_6mths","installment","out_prncp",
                                                 "pub_rec_bankruptcies","pct_tl_nvr_dlq","loan_status",
                                                "num_actv_bc_tl","dti","grade"])
new_loan_data.shape


# 1.1  处理缺失值
# 
#      首先计算缺失值比例，若缺失值很多，删除列（该属性）；很少，删除行。
#      差不多的，根据数据类型和属性的业务含义具体情况具体分析进行 缺失值填充

# In[11]:


#计算特征的缺失值比例


# In[12]:



##定义计算函数 calculate the missing value percent of features
def draw_missing_data_table(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.shape[0]).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    missing_data.reset_index(inplace=True)
    missing_data.rename(columns={"index": "feature_name"}, inplace=True)

    return missing_data


# In[13]:


##计算比例
missing_data_count = draw_missing_data_table(new_loan_data)
missing_data_count.to_csv("missing_data_count.csv")
missing_data_count = pd.read_csv("missing_data_count.csv", header=0, index_col=0)
missing_data_count = missing_data_count[missing_data_count["Percent"] > 0.0]
print(missing_data_count.head())


# In[14]:


msno.matrix(new_loan_data, labels=True)


# In[15]:


c=[new_loan_data['dti'],new_loan_data['avg_cur_bal']]


# 1.1的结果分析：
# 

#  1.剩余缺失值比例较小，我们删除缺失比例小于0.001的行，由于样本并不太多，为了保持信息全面，因此，我们考虑对dti缺失的行进行填补

#  2.对缺失数据进行分析，决定如何填补

# 业务理解：dti：借款人负债比

# 1.都是 离散 数值型 2.每个特征在不同样本之间没有规律，不能用平均值填充，0/众数比较合适
# 缺失原因：可能是有实际意义的缺失，就是负债率和从未拖欠率为0，因此我们考虑用0来填充

# 下面是具体的缺失值处理步骤分析：

# In[16]:


##对于缺失值比例小于0.01的特征，删除含有这些特征的缺失值的行数据 删除行:
# delete rows which contain missing value for features that  missing value precent less than 0.04
for index, feature_count_null in missing_data_count.iterrows():
	if feature_count_null["Percent"] < 0.01:
		drop_feature_name = feature_count_null["feature_name"]
		drop_index = new_loan_data[new_loan_data[drop_feature_name].isnull().values == True].index
		new_loan_data.drop(index=drop_index, axis=0, inplace=True)

print(new_loan_data.shape)


# In[17]:


#对负债率，用0来填充
new_loan_data["dti"].fillna(value=0, inplace=True)


# 1.2 选择标签grade，将其转化为二分类，并计算样本比例

# In[18]:


#缺失数据处理结束
#将信用等级二分类划分：1良好 2较差，我们发现 样本较为均衡，因此不需要均衡化处理
grade_dict = {"A": 1, "B": 1, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0}
# 用1表示信用状况良好，用0表示信用较差
new_loan_data["grade"] = new_loan_data["grade"].map(grade_dict)
new_loan_data["grade"] = new_loan_data["grade"].astype("float")
new_loan_data_grade_count = new_loan_data["grade"].value_counts().to_dict()
sum_value = 0.0
for key, value in new_loan_data_grade_count.items():
	sum_value += value
for key, value in new_loan_data_grade_count.items():
	new_loan_data_grade_count[key] = value / sum_value
print(new_loan_data_grade_count)
#new_loan_data.drop(["grade"], axis=1, inplace=True)
print(new_loan_data.shape)


# 111可视化

# In[20]:


fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x="grade",data=new_loan_data,ax=axs[0])
axs[0].set_title("Frequency of each grade")
new_loan_data["grade"].value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each grade")
plt.show()
plt.close()
fig.savefig("1.Frequency and Percentage of each grade.jpg", dpi=200, bbox_inches="tight")


# In[21]:


#美观
import plotly as py
import plotly.graph_objs as go
pyplt=py.offline.plot
labels=['0','1']
values=[0.5127,0.4873]
trace=[go.Pie(
    labels=labels,
    values=values,
    hole=0.7,   #控制环形中心空白大小
    hoverinfo='label+percent'       #hoverinfo属性用于控制当用户将鼠标指针放到环形图上时，显示的内容
)]
layout=go.Layout(
    title='训练集的标签分布',
)
fig=go.Figure(data=trace,layout=layout)
pyplt(fig,filename='./环形饼图.html')


# 1.2 结果分析：样本较均衡，因此不需要均衡化处理

# 1.3 特征转换

# In[22]:


#1.将非数值型的数据进行数字化转换。
objectColumns = new_loan_data.select_dtypes(include=["object"]).columns
objectColumns


# In[23]:


new_loan_data['home_ownership'].value_counts()


# In[24]:


new_loan_data['loan_status'].value_counts()


# In[25]:


new_loan_data['term'].value_counts()


# In[26]:


#看分类型变量贡献度
import plotly.express as px
fig = px.parallel_categories(new_loan_data, dimensions=['term', 'home_ownership', 'loan_status','grade'],
                color="grade", color_continuous_scale=px.colors.sequential.Emrld,width=1000,height=618
                )
fig.show()


# In[27]:


import plotly.express as px
fig = px.parallel_categories(new_loan_data, dimensions=['term', 'home_ownership', 'loan_status','grade'],
                color="grade", color_continuous_scale=px.colors.sequential.Emrld,width=1000,height=618
                )
fig.show()


# 发现很均匀

# loan_status是贷款状态，是多分类，变成二分类
# term和home_owner也是多分类，且是多值无序变量 但是这里我们用了onehot编码，因为他们对维度的扩展不会很大

# In[28]:


#将三种分类变量都变成数值型，其中多分类贷款状态变成二分类
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
# 1 is good loan,0 is bad loan
new_loan_data["loan_status"] = new_loan_data["loan_status"].map(loan_status_dict)
new_loan_data["loan_status"] = new_loan_data["loan_status"].astype("float")
##term和home是多值无序，对多值无序变量进行独热编码
n_columns = ["home_ownership", "term"] 
dummy_df = pd.get_dummies(new_loan_data[n_columns])# 用get_dummies进行one hot编码
new_loan_data = pd.concat([new_loan_data, dummy_df],axis=1) #当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并


# In[29]:


new_loan_data.loc[:,new_loan_data.columns.str.contains("home_ownership")].head()


# In[30]:


new_loan_data = new_loan_data.drop(n_columns, axis=1)  #清除原来的分类变量


# In[31]:


col = new_loan_data.select_dtypes(include=['int64','float64']).columns
#选择非onehot编码的进行标准化处理
#len(col)


# 1.4 标准化处理：

# In[32]:


col = col.drop('grade') #剔除目标变量
new_loan_data_ml_df = new_loan_data # 复制数据至变量loans_ml_df
###################################################################################
from sklearn.preprocessing import StandardScaler # 导入模块
sc =StandardScaler() # 初始化缩放器
new_loan_data_ml_df[col] =sc.fit_transform(new_loan_data_ml_df[col]) #对数据进行标准化
new_loan_data_ml_df.head() #查看经标准化后的数据


# 特征进行进行抽象化处理，使得变量的数据类型让算法可以理解，同时也将不同变量的规格缩放至同一规格

# 1.5 确定第一组属性值和标签值

# In[33]:


col_new=new_loan_data.columns
col_new=col_new.drop('grade')
X1=new_loan_data[col_new]#已经缩放完的
#X1


# In[34]:


y=new_loan_data['grade']


# 设置第一组训练集和测试集，用随机森林进行训练

# ！！【本实验测试集训练集的比都为0.23】

# In[35]:


from sklearn.model_selection import train_test_split
X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y,test_size=0.23,random_state=3)	#这里


# In[36]:


# use randomforest model to train and predict
print("use randomforest model to train and predict")
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(X1_train, y1_train)
rf_y_pred = rf.predict(X1_test)
rf_test_acc = accuracy_score(y1_test, rf_y_pred)
rf_classification_score = classification_report(y1_test, rf_y_pred)
print("Rf model test accuracy:{:.4f}".format(rf_test_acc))
print("rf model classification_score:\n", rf_classification_score)
rf_confusion_score = confusion_matrix(y1_test, rf_y_pred)
# print(rf_confusion_score)
f_rf, ax_rf = plt.subplots(1, 3, figsize=(15, 10))
# 混淆矩阵的y轴为true label,x轴为pred label
# 精确率,如对正类 ,所有预测为正类样本中中真实的正类占所有预测为正类的比例
# 召回率,如对正类,所有真实的正类样本中有多少被预测为正类的比例
# 分别计算预测预测的正样本数和负样本数以及真实的正样本数和负样本数
rf_cm_pred_label_sum = rf_confusion_score.sum(axis=0)
rf_cm_true_label_sum = rf_confusion_score.sum(axis=1)
# 计算正样本和负样本的精确率和召回率
rf_model_precision, rf_model_recall = np.empty([2, 2], dtype=float), np.empty([2, 2], dtype=float)
rf_model_precision[0][0], rf_model_precision[1][0] = rf_confusion_score[0][0] / rf_cm_pred_label_sum[0],                                                      rf_confusion_score[1][0] / rf_cm_pred_label_sum[0]
rf_model_precision[0][1], rf_model_precision[1][1] = rf_confusion_score[0][1] / rf_cm_pred_label_sum[1],                                                      rf_confusion_score[1][1] / rf_cm_pred_label_sum[1]
rf_model_recall[0][0], rf_model_recall[0][1] = rf_confusion_score[0][0] / rf_cm_true_label_sum[0],                                                rf_confusion_score[0][1] / rf_cm_true_label_sum[0]
rf_model_recall[1][0], rf_model_recall[1][1] = rf_confusion_score[1][0] / rf_cm_true_label_sum[1],                                                rf_confusion_score[1][1] / rf_cm_true_label_sum[1]
sns.heatmap(rf_confusion_score, annot=True, fmt="d", cmap="Blues", ax=ax_rf[0], square=True, linewidths=0.5)
sns.heatmap(rf_model_precision, annot=True, fmt=".5f", cmap="Blues", ax=ax_rf[1], square=True, linewidths=0.5)
sns.heatmap(rf_model_recall, annot=True, fmt=".5f", cmap="Blues", ax=ax_rf[2], square=True, linewidths=0.5)
ax_rf[0].set_title("rf confusion matrix", fontsize=16)
ax_rf[1].set_title("rf model precision", fontsize=16)
ax_rf[2].set_title("rf model recall", fontsize=16)
ax_rf[0].set_xlabel("Predicted label", fontsize=16)
ax_rf[0].set_ylabel("True label", fontsize=16)
ax_rf[1].set_xlabel("Predicted label", fontsize=16)
ax_rf[1].set_ylabel("True label", fontsize=16)
ax_rf[2].set_xlabel("Predicted label", fontsize=16)
ax_rf[2].set_ylabel("True label", fontsize=16)
plt.show()
plt.close()
f_rf.savefig("./pictures/2.1 rf model confusion matrix.jpg", dpi=200, bbox_inches="tight")


# 第一组属性的准确率为0.82

# 正常情况下，影响目标变量的因数是多元性的；
# 但不同因数之间会互相影响（共线性），或相重叠，进而影响到统计结果的真实性。
# 下一步，我们在第一次降维的（主观）选择基础上，通过皮尔森相关性图谱找出冗余特征并将其剔除；
# 同时，可以通过相关性图谱进一步引导我们选择特征的方向。
# 降维！！

# In[37]:


#看第一次变量间的相关性
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(new_loan_data[col_new].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[38]:


#plotly美观
#import plotly.io as pio
#import plotly.express as px
#import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from plotly.subplots import make_subplots
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)
#pio.templates.default = "none"
corr = new_loan_data[col_new].corr()
fig = ff.create_annotated_heatmap(
    z=corr.to_numpy().round(2),
    x=list(corr.index.values),
    y=list(corr.columns.values),       
    xgap=3, ygap=3,
    zmin=-1, zmax=1,
    colorscale='Mint',
    colorbar_thickness=30,
    colorbar_ticklen=3,
)
fig.update_layout(title_text='<b>Correlation Matrix (cont. features)<b>',
                  title_x=0.5,
                  titlefont={'size': 24},
                  width=1000, height=1000,
                  xaxis_showgrid=False,
                  xaxis={'side': 'bottom'},
                  yaxis_showgrid=False,
                  yaxis_autorange='reversed',                   
                  paper_bgcolor=None,
                  )
fig.show()


# 由图可发现，install和loan_amnt之间正相关性很强，0.94，
# open_acc和total_acc相关性较强0.74
# d2yrs和out完全不相关
# 因此我们删除install或loan中的一个，主观分析，instal比loan的对个人信用等级的影响更大。因此我们删除loan

# In[39]:


feature_names = X1.columns.tolist()
feature_names = np.array(feature_names)
print(feature_names.shape)


# 1.6 设置第二组属性值X2=X1删除loan_amnt

# In[40]:


drop_col = ["loan_amnt"]
col_new2 = col_new.drop(drop_col) #
X2=new_loan_data[col_new2]


# In[41]:


X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y,test_size=0.23,random_state=3)


# In[42]:


# use randomforest model to train and predict
print("use randomforest model to train and predict")
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(X2_train, y2_train)
rf_y_pred = rf.predict(X2_test)
rf_test_acc = accuracy_score(y2_test, rf_y_pred)
rf_classification_score = classification_report(y2_test, rf_y_pred)
print("Rf model test accuracy:{:.4f}".format(rf_test_acc))
print("rf model classification_score:\n", rf_classification_score)
rf_confusion_score = confusion_matrix(y2_test, rf_y_pred)
# print(rf_confusion_score)
f_rf, ax_rf = plt.subplots(1, 3, figsize=(15, 10))
# 混淆矩阵的y轴为true label,x轴为pred label
# 精确率,如对正类 ,所有预测为正类样本中中真实的正类占所有预测为正类的比例
# 召回率,如对正类,所有真实的正类样本中有多少被预测为正类的比例
# 分别计算预测预测的正样本数和负样本数以及真实的正样本数和负样本数
rf_cm_pred_label_sum = rf_confusion_score.sum(axis=0)
rf_cm_true_label_sum = rf_confusion_score.sum(axis=1)
# 计算正样本和负样本的精确率和召回率
rf_model_precision, rf_model_recall = np.empty([2, 2], dtype=float), np.empty([2, 2], dtype=float)
rf_model_precision[0][0], rf_model_precision[1][0] = rf_confusion_score[0][0] / rf_cm_pred_label_sum[0],                                                      rf_confusion_score[1][0] / rf_cm_pred_label_sum[0]
rf_model_precision[0][1], rf_model_precision[1][1] = rf_confusion_score[0][1] / rf_cm_pred_label_sum[1],                                                      rf_confusion_score[1][1] / rf_cm_pred_label_sum[1]
rf_model_recall[0][0], rf_model_recall[0][1] = rf_confusion_score[0][0] / rf_cm_true_label_sum[0],                                                rf_confusion_score[0][1] / rf_cm_true_label_sum[0]
rf_model_recall[1][0], rf_model_recall[1][1] = rf_confusion_score[1][0] / rf_cm_true_label_sum[1],                                                rf_confusion_score[1][1] / rf_cm_true_label_sum[1]
sns.heatmap(rf_confusion_score, annot=True, fmt="d", cmap="Blues", ax=ax_rf[0], square=True, linewidths=0.5)
sns.heatmap(rf_model_precision, annot=True, fmt=".5f", cmap="Blues", ax=ax_rf[1], square=True, linewidths=0.5)
sns.heatmap(rf_model_recall, annot=True, fmt=".5f", cmap="Blues", ax=ax_rf[2], square=True, linewidths=0.5)
ax_rf[0].set_title("rf confusion matrix", fontsize=16)
ax_rf[1].set_title("rf model precision", fontsize=16)
ax_rf[2].set_title("rf model recall", fontsize=16)
ax_rf[0].set_xlabel("Predicted label", fontsize=16)
ax_rf[0].set_ylabel("True label", fontsize=16)
ax_rf[1].set_xlabel("Predicted label", fontsize=16)
ax_rf[1].set_ylabel("True label", fontsize=16)
ax_rf[2].set_xlabel("Predicted label", fontsize=16)
ax_rf[2].set_ylabel("True label", fontsize=16)
plt.show()
plt.close()
f_rf.savefig("./pictures/2.2 rf model confusion matrix.jpg", dpi=200, bbox_inches="tight")


# 实验对比，发现删除loan更好，准确率更高

# 第二组属性准确率为0.77

# 1.8第三组属性选取

# 这次我们采用pca降维

# 可视化

# In[44]:


from sklearn.decomposition import PCA
model = PCA()
model.fit(X1)
#每个主成分能解释的方差
model.explained_variance_
#每个主成分能解释的方差的百分比
model.explained_variance_ratio_
#可视化
plt.plot(model.explained_variance_ratio_, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('PVE')


# In[45]:


plt.plot(model.explained_variance_ratio_.cumsum(), 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.axhline(0.9, color='k', linestyle='--', linewidth=1)
plt.title('Cumulative PVE')


# In[46]:


import plotly.express as px
from sklearn.decomposition import PCA
pca = PCA()     #实例化一个PCA方法
components = pca.fit_transform(X1) 
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"      #输出字符串第i个主成分的贡献率
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)

fig = px.scatter_matrix(components,labels=labels, dimensions=range(22),color=[])
fig.update_traces(diagonal_visible=False)  #设置对角线不显示，有兴趣可删除查看
fig.show()


# 结果分析：12（3）个主成分能解释到90%以上了（之前是25个属性），和第二组属性选取不同，我们这次是筛选主成分，是原始属性值产生的新的变量(线性组合）/第二组是剔除相关性高的

# In[47]:


px.scatter_matrix


# In[48]:


pca = PCA(n_components=0.9)
data = pca.fit_transform(X1)

# 输出数据
print(data)


# In[49]:


data.shape


# 设置第三组属性值

# In[50]:


X3=data
from sklearn.model_selection import train_test_split
X3_train,X3_test,y3_train,y3_test = train_test_split(X3,y,test_size=0.23,random_state=3)


# 训练

# In[52]:


# use randomforest model to train and predict
print("use randomforest model to train and predict")
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(X3_train, y3_train)
rf_y_pred = rf.predict(X3_test)
rf_test_acc = accuracy_score(y3_test, rf_y_pred)
rf_classification_score = classification_report(y3_test, rf_y_pred)
print("Rf model test accuracy:{:.4f}".format(rf_test_acc))
print("rf model classification_score:\n", rf_classification_score)
rf_confusion_score = confusion_matrix(y3_test, rf_y_pred)
# print(rf_confusion_score)
f_rf, ax_rf = plt.subplots(1, 3, figsize=(15, 10))
# 混淆矩阵的y轴为true label,x轴为pred label
# 精确率,如对正类 ,所有预测为正类样本中中真实的正类占所有预测为正类的比例
# 召回率,如对正类,所有真实的正类样本中有多少被预测为正类的比例
# 分别计算预测预测的正样本数和负样本数以及真实的正样本数和负样本数
rf_cm_pred_label_sum = rf_confusion_score.sum(axis=0)
rf_cm_true_label_sum = rf_confusion_score.sum(axis=1)
# 计算正样本和负样本的精确率和召回率
rf_model_precision, rf_model_recall = np.empty([2, 2], dtype=float), np.empty([2, 2], dtype=float)
rf_model_precision[0][0], rf_model_precision[1][0] = rf_confusion_score[0][0] / rf_cm_pred_label_sum[0],                                                      rf_confusion_score[1][0] / rf_cm_pred_label_sum[0]
rf_model_precision[0][1], rf_model_precision[1][1] = rf_confusion_score[0][1] / rf_cm_pred_label_sum[1],                                                      rf_confusion_score[1][1] / rf_cm_pred_label_sum[1]
rf_model_recall[0][0], rf_model_recall[0][1] = rf_confusion_score[0][0] / rf_cm_true_label_sum[0],                                                rf_confusion_score[0][1] / rf_cm_true_label_sum[0]
rf_model_recall[1][0], rf_model_recall[1][1] = rf_confusion_score[1][0] / rf_cm_true_label_sum[1],                                                rf_confusion_score[1][1] / rf_cm_true_label_sum[1]
sns.heatmap(rf_confusion_score, annot=True, fmt="d", cmap="Blues", ax=ax_rf[0], square=True, linewidths=0.5)
sns.heatmap(rf_model_precision, annot=True, fmt=".5f", cmap="Blues", ax=ax_rf[1], square=True, linewidths=0.5)
sns.heatmap(rf_model_recall, annot=True, fmt=".5f", cmap="Blues", ax=ax_rf[2], square=True, linewidths=0.5)
ax_rf[0].set_title("rf confusion matrix", fontsize=16)
ax_rf[1].set_title("rf model precision", fontsize=16)
ax_rf[2].set_title("rf model recall", fontsize=16)
ax_rf[0].set_xlabel("Predicted label", fontsize=16)
ax_rf[0].set_ylabel("True label", fontsize=16)
ax_rf[1].set_xlabel("Predicted label", fontsize=16)
ax_rf[1].set_ylabel("True label", fontsize=16)
ax_rf[2].set_xlabel("Predicted label", fontsize=16)
ax_rf[2].set_ylabel("True label", fontsize=16)
plt.show()
plt.close()
f_rf.savefig("2.3 rf model confusion matrix.jpg", dpi=200, bbox_inches="tight")


# 结果分析：0.7的准确度

# 对比三组结果，1.业务主观判断剔除（0.8）2.相关系数大的剔除（0.77）3.pca降维（0.7）
# 1v2.
# 【缺】相关系数大的剔除虽然降低了准确度，【优】但是我们它可以防止过拟合现象
# 
# 
# #结论：主成分分析法效果弱于相关性的剔除，可能愿意在于 一些属性虽然相关性大 大可能对标签有决定性意义；
# #但都不如初始的属性好，主成分的降维可能导致关键信息的丢失（查主成分分析的缺点）
# 
# 还是第一组属性合适，理由是，2，3组都是依据一些统计学方法进行的，但是统计学的逻辑回归和机器学习的逻辑回归的目标不同，统计学中默认有一个潜在的规律，调模型时有各种限制来满足假设条件（VIF，线性相关就是这样），来找到那个潜在的规律，而机器学习不同，只关心预测值和真实值的偏差，因此，要选择准确度最高的属性更合适。
# 
# 【备选的结果分析】#甚至train和oot上的悬殊差别也能接受，只要oot上AUC越高，就可以以方差衡量信息的无监督学习，不受样本标签限制。
# 1v3.
# pca的优点：1.由于协方差矩阵对称，因此k个特征向量之间两两正交，也就是各主成分之间正交，正交就肯定线性不相关，可消除原始数据成分间的相互影响。2.可减少指标选择的工作量，我们本来选用onehot编码就会无形增加维度，降维可以帮助减少这个问题，而且计算方法简单，易于在计算机上实现。虽然略有下降精确度，但算法复杂度降低。
# pca缺点：主成分解释其含义往往具有一定的模糊性，不如原始样本完整。贡献率小的主成分往往可能含有对样本差异的重要信息，也就是可能对于区分样本的类别（标签）更有用。
# 3v1，2最差原因：
# 主成分分析作为一个非监督学习的降维方法，它只需要特征值分解，就可以对数据进行压缩，去噪。因此在实际场景应用很广泛。通过对原始变量进行综合与简化，可以客观地确定各个指标的权重，避免主观判断的随意性。并且不要求数据呈正态分布，其就是按数据离散程度最大的方向对基组进行旋转，这特性扩展了其应用范围，比如，用于人脸识别。
# 但同时，其适用于变量间有较强相关性的数据【本数据相关性并不强】，若原始数据相关性弱，则起不到很好的降维作用（必须通过KMO和Bartlett的检验），并且降维后，存在少量信息丢失，不可能包含100%原始数据，原始数据经过标准化处理之后，含义会发生变化，且主成分的解释含义较原始数据比较模糊。
