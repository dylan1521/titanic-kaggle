#encoding=utf-8
from __future__ import division
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as pp
from sklearn.ensemble import RandomForestRegressor

Data = pd.read_csv('train.csv')
Data = Data.drop('PassengerId',axis=1) #舍弃PassengerId，这个肯定与预测结果没有关系的数据
# Data['Age'].fillna(Data['Age'].mean(),inplace=True)
Data['Embarked'].fillna('S',inplace=True) 
#逻辑回归填充年龄
def set_missing_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']] # 选取一些已有的数值型特征取出来丢进Random Forest Regressor中
    known_age = age_df[age_df.Age.notnull()].as_matrix()  
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:, 0] # y即目标年龄
    X = known_age[:, 1:] # X即特征属性值
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1) 
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::]) 
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    return df
Data = set_missing_ages(Data)
Data.fillna('NAN',inplace=True)
# print Data.info()
# print Data.head()

#大体的相关度分析
def all_analy(Data):
	f,ax = plt.subplots()
	Data_corr = Data.corr()
	print Data_corr
	sns.heatmap(Data_corr)
	ax.set_xticklabels(Data_corr.index)
	ax.set_yticklabels(Data_corr.columns)
	ax.set_title('Data_corr')
	plt.show()
# all_analy(Data)

#分析年龄
def age_analy(Data):
	f,ax = plt.subplots()
	Data_1 = Data[Data['Survived']==1]['Age']
	Data_2 = Data[Data['Survived']==0]['Age']
	sns.kdeplot(Data_1,label='Survived')
	sns.kdeplot(Data_2,label='Dead')
	plt.title('Age-fenbu')
	plt.ylim((0,0.04))
	plt.show()
# age_analy(Data)

#分析性别
def sex_analy(Data):
	plt.figure()
	Data_0 = Data.groupby('Sex')['Survived'].count()
	rate_0 = '{:.3f}'.format(Data_0['male']/Data_0['female'])	
	Data_1 = Data[Data['Survived']==1].groupby('Sex')['Survived'].count()
	rate_1 = '{:.3f}'.format(Data_1['male']/Data_1['female'])
	Data_2 = Data[Data['Survived']==0].groupby('Sex')['Survived'].count()
	rate_2 = '{:.3f}'.format(Data_2['male']/Data_2['female'])
	x = ['primary','Survived','Dead']
	y = [rate_0,rate_1,rate_2]
	y = map(eval,y)
	plt.bar(x,y,label='male/female')
	plt.ylim((0,8))
	plt.title('Sex-rate')
	for x,y in zip(x,y):
		plt.text(x,y+0.4,y,ha='center',va='top',fontsize=14)
	plt.legend(loc='best')
	plt.show()
# sex_analy(Data)

#分析class
def pclass_analy(Data):
	Data_1 = Data[Data.Survived==1].groupby('Pclass').Survived.count()
	Data_2 = Data[Data.Survived==0].groupby('Pclass').Survived.count()
	rate = (Data_1/(Data_2+Data_1))
	plt.figure()
	x = ['class1','class2','class3']
	plt.bar(x,Data_1,label='Survived',color='green')
	plt.bar(x,Data_2,label='Dead',bottom=Data_1,color='black')
	y = Data_1+Data_2
	print y
	print rate
	for i,j,k in zip(x,y,rate):
		plt.text(i,j+24,'{:.3f}'.format(k),va='top',ha='center',fontsize=14)
	plt.legend(loc='best')
	plt.title('different pclaa-survived rate')
	plt.show()
# pclass_analy(Data)

#分析票价
def fare_analy(Data):
	f,ax = plt.subplots()
	Data_1 = Data[Data.Survived==1].Fare
	Data_2 = Data[Data.Survived==0].Fare
	sns.kdeplot(Data_1,label='Survived')
	sns.kdeplot(Data_2,label='Dead')
	plt.ylim((0,0.06))
	plt.title('fare-fenbu')
	plt.xticks(range(0,500,20))
	plt.show()
# fare_analy(Data)

#分析仓位（过于混乱，缺失值也较多，所以分成两类，有仓位信息的和仓位信息为NAN的）
def cabin_analy(Data):
	plt.figure()
	Data_1 = Data[(Data.Survived==1) & (Data.Cabin=='NAN')].Cabin.count()
	Data_2 = Data[(Data.Survived==1) & (Data.Cabin!='NAN')].Cabin.count()
	Data_3 = Data[(Data.Survived==0) & (Data.Cabin=='NAN')].Cabin.count()
	Data_4 = Data[(Data.Survived==0) & (Data.Cabin!='NAN')].Cabin.count()
	x = ['Cabin==NAN','Cabin!=NAN']
	y = ['{:.3f}'.format(Data_1/(Data_1+Data_3)),'{:.3f}'.format(Data_2/(Data_2+Data_4))]
	y = map(eval,y)
	plt.bar(x,y)
	plt.title('Survived Rate')
	plt.legend('best')
	for i,j in zip(x,y):
		plt.text(i,j+0.03,j,ha='center',va='top',fontsize=12)
	plt.show()
# cabin_analy(Data)

#分析船票（数据混乱，但是观察到有的包含英文，有的不包含英文，所以考虑分为两类看看）
def ticket_analy(Data):
	plt.figure()
	def have_alpha(data):
		for i in data:
			if i.isalpha()==True:
				return 1
			else:
				return 0
	Data_new = Data
	Data_new.Ticket = Data_new.Ticket.apply(lambda x:have_alpha(x))
	Data_new = Data_new.groupby('Ticket').Survived.mean()
	x = ['ticket_have_alpha','ticket_no_alpha']
	plt.bar(x,Data_new)
	plt.title('Survived-Rate')
	plt.show()
# ticket_analy(Data) #没有明显影响

#分析上岸地点
def embark_analy(Data):
	Data_new = Data.groupby('Embarked').Survived.mean()
	x = ['C','Q','S']
	y = Data_new
	plt.bar(x,y)
	for i,j in zip(x,y):
		plt.text(i,j+0.03,'{:.3f}'.format(j),ha='center',va='top')
	plt.title('Survived-Rate')
	plt.show()
# embark_analy(Data)

#分析亲属
def sib_par_analy(Data):
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	Data.groupby('SibSp')['Survived'].mean().plot(kind='bar', ax=ax1)
	ax1.set_title('Sibsp Survived-Rate')
	ax1.set_xlabel('')
	x1_ticks = np.unique(Data.SibSp)
	ax1.set_xticklabels(x1_ticks)

	ax2 = fig.add_subplot(212)
	Data.groupby('Parch')['Survived'].mean().plot(kind='bar', ax=ax2)
	ax2.set_title('Parch Survived-Rate')
	x2_ticks = np.unique(Data.Parch)
	ax2.set_xticklabels(x2_ticks)
	ax2.set_xlabel('')
	plt.show()
# sib_par_analy(Data) #关系混乱，没有对应关系,舍弃

#分析姓名
def name_analy(Data):
	def hava_M(data):
		if 'Ms' in data or 'Mrs.' in data or 'Lady' in data or 'Mlle' in data:
			return 1
		else:
			return 0
	Data.Name = Data.Name.apply(lambda x:hava_M(x))
	f,ax=plt.subplots()
	y = Data.groupby('Name').Survived.mean()
	x = ['have M','no M']
	ax.bar(x,y)
	for i,j in zip(x,y):
		plt.text(i,j+0.03,j,ha='center',va='top')
	plt.show()
# name_analy(Data)


#分析完毕，除了亲属和船票看不出明显相关之外，其他特征均作保留，只是对一些特征进行重造
#合并train和test，并进行数据填充
Train_Data = pd.read_csv('train.csv').drop(['PassengerId','Parch','SibSp','Ticket'],axis=1)
Test_Data = pd.read_csv('test.csv').drop(['PassengerId','Parch','SibSp','Ticket'],axis=1)
Total_Data = pd.concat([Train_Data,Test_Data],axis=0,ignore_index=True)
PassengerId = pd.read_csv('test.csv')['PassengerId'] #保留PassengerId信息，为最后输出结果用
# print Total_Data.Survived
Total_Data['Age'].fillna(Total_Data['Age'].mean(),inplace=True)
Total_Data['Embarked'].fillna('S',inplace=True) 
Total_Data['Fare'].fillna(Total_Data['Fare'].mean(),inplace=True)
Total_Data.fillna('NAN',inplace=True)#按照之前特征模型时的方法填充缺失数据
# print Total_Data.info()


#进行特征工程
#Cabin 根据有没有cabin数据分为两类
def hava_cabin(data):
	if data=='NAN':
		return 0
	else:
		return 1
Total_Data.Cabin = Total_Data.Cabin.apply(lambda x:hava_cabin(x))

# Name 看年龄里面有没有和身份相关的信息，进行分类
def hava_M(data):
	if 'Ms' in data or 'Mrs.' in data or 'Sir' in data or 'Lady' in data or 'Mlle' in data:
		return 1
	else:
		return 0
Total_Data.Name = Total_Data.Name.apply(lambda x:hava_M(x))

def change_embarked(data):
	if data=='S':
		return 0
	elif data=='C':
		return 1
	else:
		return 2
Total_Data.Embarked = Total_Data.Embarked.apply(lambda x:change_embarked(x))

#根据年龄分析是不是小孩或者老人，这样就把年龄为分3类
def is_baby_or_older(data):
	if data<12:
		return 1
	elif data>60:
		return 2
	else:
		return 3
Total_Data.Age = Total_Data.Age.apply(lambda x:is_baby_or_older(x))

def change_fare(data):
	if data<30:
		return 0
	elif data<60:
		return 1 
	elif data<100:
		return 2
	else:
		return 3
Total_Data.Fare = Total_Data.Fare.apply(lambda x:change_fare(x))
#对连续性数据进行scaling处理，加速模型收敛
# scaler = pp.StandardScaler()
# age_scale_param = scaler.fit(Total_Data['Age'].reshape(-1, 1))
# Total_Data['Age'] = scaler.fit_transform(Total_Data['Age'].reshape(-1, 1), age_scale_param)
# fare_scale_param = scaler.fit(Total_Data['Fare'].reshape(-1, 1))
# Total_Data['Fare'] = scaler.fit_transform(Total_Data['Fare'].reshape(-1, 1), fare_scale_param)

#将其他离散的数据转化为onehot数据，这样可以实现升维的效果,以便使用后续的svm等分类
dummies_Cabin = pd.get_dummies(Total_Data['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(Total_Data['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(Total_Data['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(Total_Data['Pclass'], prefix= 'Pclass')
dummies_Name = pd.get_dummies(Total_Data['Name'], prefix= 'Name')
dummies_Age = pd.get_dummies(Total_Data['Age'], prefix= 'Age')
dummies_Fare = pd.get_dummies(Total_Data['Fare'], prefix= 'Fare')
df = pd.concat([Total_Data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,dummies_Name,dummies_Age,dummies_Fare], axis=1)
Total_Data = df.drop(['Pclass', 'Name', 'Sex', 'Cabin', 'Embarked','Age','Fare'], axis=1)
print Total_Data.info()
print Total_Data.head() #这样输出的就是完整的处理过后的数据了
# all_analy(Total_Data[Total_Data.Survived!='NAN']) #分析处理完数据的大致相关性


#分离train和test
features = Total_Data[Total_Data.Survived!='NAN'].drop('Survived',axis=1).columns
Train_X = (Total_Data[Total_Data.Survived!='NAN'].drop('Survived',axis=1)).astype(float)
Train_Y = (Total_Data[Total_Data.Survived!='NAN'].Survived).astype(int)
Test = (Total_Data[Total_Data.Survived=='NAN'].drop('Survived',axis=1)).astype(float)
# print Train_X

#测试特征的重要性，并画出柱状排列图
from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier()
rf.fit(Train_X, Train_Y)
importances = {features[i]:rf.feature_importances_[i] for i in range(len(features))}
importances = sorted(importances.items(),key=lambda x:x[1],reverse=True) #对重要性排序
x = map(lambda i:i[0],importances)
y = map(lambda i:i[1],importances)
f, ax = plt.subplots()
plt.bar(range(len(y)),y)
plt.xticks(range(len(y)),x)
plt.title("Feature Importances")
plt.show()
# x = x[:10]#选取前10个重要的特征

# #再次处理train和test数据
# Train_X = Train_X.loc[:,x]
# Test = Test.loc[:,x]

from sklearn.ensemble import VotingClassifier #模型融合
from sklearn import cross_validation,metrics #先对train使用交叉验证求不同模型的预测准确度
from sklearn.svm import SVC  #svm分类
from sklearn.neighbors import KNeighborsClassifier #k近邻
from sklearn.ensemble import GradientBoostingClassifier #梯度提升决策树
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.naive_bayes import GaussianNB #贝叶斯分类
from sklearn.tree import DecisionTreeClassifier#决策树
from sklearn.grid_search import GridSearchCV #调参

clf1 = SVC(kernel='linear', C=5)
scores = cross_validation.cross_val_score(clf1, Train_X, Train_Y, cv=5)
print 'svm:{}'.format(scores.mean())

clf2 = GradientBoostingClassifier(random_state=10)
scores = cross_validation.cross_val_score(clf2, Train_X, Train_Y, cv=5)
print 'gdbc:{}'.format(scores.mean())

clf3 = KNeighborsClassifier(n_neighbors=10) 
scores = cross_validation.cross_val_score(clf3, Train_X, Train_Y, cv=5)
print 'knn:{}'.format(scores.mean())

clf4 = GaussianNB()
scores = cross_validation.cross_val_score(clf4, Train_X, Train_Y, cv=5)
print 'beyse:{}'.format(scores.mean())

# clf5 = LogisticRegression()
# scores = cross_validation.cross_val_score(clf5, Train_X, Train_Y, cv=5)
# print scores.mean()

# clf6 = DecisionTreeClassifier()
# scores = cross_validation.cross_val_score(clf6, Train_X, Train_Y, cv=5)
# print scores.mean()

eclf = VotingClassifier(estimators=[ ('svc',clf1),('gdbc', clf2), ('knn', clf3),('gb',clf4)], voting='hard')
scores = cross_validation.cross_val_score(eclf, Train_X, Train_Y, cv=5)
print scores.mean()
Train_X = pd.concat([Train_X,Train_X],axis=0,ignore_index=True)
Train_Y = pd.concat([Train_Y,Train_Y],axis=0,ignore_index=True)

clf = clf2
clf.fit(Train_X, Train_Y)
predictions_1 = clf.predict(Test)
result = pd.DataFrame({'PassengerId':PassengerId.as_matrix(), 'Survived':predictions_1.astype(int)})
result.to_csv("predictions_1.csv", index=False)

clf = eclf
clf.fit(Train_X, Train_Y)
predictions_2 = clf.predict(Test)
result = pd.DataFrame({'PassengerId':PassengerId.as_matrix(), 'Survived':predictions_2.astype(int)})
result.to_csv("predictions_2.csv", index=False)























