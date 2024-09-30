#!/usr/bin/env python
# coding: utf-8

# # Importing The Required Libraries

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv("creditcard.csv")


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.columns


# In[9]:


data.hist(figsize=(20,20))
plt.show()


# # Removing the Time coloumn and updating with hours and seconds

# In[10]:


data['hour']=((data['Time']//3600)//2.).astype('int')
data['second']=(data['Time']%3600).astype('int')
data.drop('Time',axis=1,inplace=True)
data.head(5)


# # Finding the duplicating rows and deleting them

# In[11]:


data.duplicated().value_counts()


# In[12]:


data.drop_duplicates(inplace = True)


# In[13]:


data.duplicated().value_counts()
data.reset_index(drop = True , inplace = True)


# In[14]:


print(data['Class'].value_counts())
print('\n')
print(data['Class'].value_counts(normalize=True))


# In[15]:


classes=data['Class'].value_counts()
Valid=round(classes[0]/data['Class'].count(), 4)
Fraud=round(classes[1]/data['Class'].count(), 4)
plt.figure(figsize=(10,5))

plt.subplot(122)
plt.title('Percentage of each Class', fontsize=14)
plt.pie(x=[Valid,Fraud],labels=['Valid','Fraud'],
        autopct='%.2f', colors=['blue','red'], startangle=90, shadow=True)
plt.legend(loc='best')

plt.show()


# # Preprocessing

# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

X = data.drop('Class', axis=1)
y = data['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.20)
rob_scaler = RobustScaler()
X_test = rob_scaler.fit_transform(X_test)
X_test = rob_scaler.fit_transform(X_test)


# # Calculating the information gain and selecting the K best features

# In[17]:


from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif
infogain_classif = SelectKBest(score_func=mutual_info_classif, k=6)
infogain_classif.fit(X_train, y_train)

for i in range(len(infogain_classif.scores_)):
    print(f'Feature {i} : {round(infogain_classif.scores_[i],4)}')


# In[18]:


plt.bar([X_train.columns[i] for i in range(len(infogain_classif.scores_))], infogain_classif.scores_ ,color=['green'])
plt.xticks(rotation=90)
plt.rcParams["figure.figsize"] = (8,7)
ax = plt.gca()
ax.patch.set_facecolor('white')
ax.yaxis.grid(True, color = 'blue')
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('1.5')
plt.xlabel('Features',fontsize=14)
plt.ylabel('Information Gain',fontsize=14)
plt.show()


# In[19]:


df = pd.DataFrame(data, columns=['V10' ,'V11' ,'V12' ,'V14' ,'V16' ,'V17', 'Class'])

X = df.drop('Class', axis=1)
y = df['Class']


# In[20]:


train, test = train_test_split(df, test_size = 0.2, random_state=42)

print("Train shape",train.shape)
print("Test shape",test.shape)


# In[21]:


X_train  = train.drop('Class', axis=1).values
y_train  = train['Class'].values
print("Before using the sampling techniques")
print(X_train.shape)
print(y_train.shape)


# In[22]:


X_test  = test.drop('Class', axis=1)
y_test  = test['Class']
print(X_test.shape)
print(y_test.shape)


# In[23]:


X_test  = test
y_test  = test['Class']

X_test_Fraud = X_test[X_test.Class==1]
X_test_Valid = X_test[X_test.Class==0]
y_test_Fraud = y_test[X_test.Class==1]
y_test_Valid = y_test[X_test.Class==0] 

X_test = X_test.drop('Class', axis=1).values
y_test = y_test.values  

X_test_Valid =X_test_Valid.drop('Class', axis=1).values
y_test_Valid = y_test_Valid.values

X_test_Fraud = X_test_Fraud.drop('Class', axis=1).values
y_test_Fraud = y_test_Fraud.values  


# In[24]:


print("Number of fraudulent transactions in test dataset",y_test_Fraud.shape)
print("Number of valid transactions in test dataset",y_test_Valid.shape)


# # Optimising LightGBM parameters using Bayesian Optimisation

# In[25]:


from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate,KFold,StratifiedKFold

def lgbm_cv(learning_rate, max_depth, num_leaves):
    model = LGBMClassifier(learning_rate = learning_rate,
                                num_leaves = int(round(num_leaves)),
                                max_depth = int(round(max_depth)), 
                                class_weight = 'balanced'
                               )
    
    cv = StratifiedKFold(n_splits=5)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring= 'neg_log_loss')
    return np.mean(scores['test_score'])

# Interval to be explored for input values
params = {'learning_rate': (0.001, 0.2),
           'max_depth': (-1, 8),
           'num_leaves': (2, 250)
          }

from bayes_opt import BayesianOptimization
lgbmBO = BayesianOptimization(lgbm_cv, params)

lgbmBO.maximize(init_points=5, n_iter = 8, acq='ei')

params_lgbm = lgbmBO.max['params']
params_lgbm['max_depth'] = round(params_lgbm['max_depth'])
params_lgbm['num_leaves'] = round(params_lgbm['num_leaves'])
print(params_lgbm)


# # Optimising XGBoost parameters using Bayesian Optimisation

# In[26]:


from collections import Counter
counter = Counter(y)
# estimate scale_pos_weight value
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)


# In[ ]:


from xgboost import XGBClassifier
def xgb_cv(learning_rate, max_depth, n_estimators):
    model = XGBClassifier(learning_rate = learning_rate,
                                max_depth = int(round(max_depth)),
                                n_estimators = int(round(n_estimators)),
                                scale_pos_weight = 592
                          )
    cv = StratifiedKFold(n_splits=5)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring='neg_log_loss')
    return np.mean(scores['test_score'])

# Interval to be explored for input values
params={'learning_rate': (0.001, 0.2),
           'max_depth': (3, 10),
           'n_estimators': (50, 100)
          }

xgbBO = BayesianOptimization(xgb_cv, params)
xgbBO.maximize(init_points=5, n_iter = 8, acq='ei')

params_xgb = xgbBO.max['params']
params_xgb['max_depth'] = round(params_xgb['max_depth'])
params_xgb['n_estimators'] = round(params_xgb['n_estimators'])
params_xgb['learning_rate'] = round((params_xgb['learning_rate']),4)
print(params_xgb)


# In[ ]:


from sklearn.model_selection import cross_validate,StratifiedKFold
X = data.drop('Class', axis = 1)
y = data['Class']
Accuracy = []
Precision = []
Recall = []
F1_score = []
ROC_AUC = []
def cv_results(model):
    sk_fold = StratifiedKFold(n_splits=5)
    scoring = ["accuracy","precision","recall","f1","roc_auc"]
    cv_scores = cross_validate(model, X, y, scoring = scoring, cv = sk_fold)
    print("Accuracy : ",cv_scores["test_accuracy"].mean()*100)
    Accuracy.append(cv_scores["test_accuracy"].mean()*100)
    print("Precision : ",cv_scores["test_precision"].mean()*100)
    Precision.append(cv_scores["test_precision"].mean()*100)
    print("Recall : ",cv_scores["test_recall"].mean()*100)
    Recall.append(cv_scores["test_recall"].mean()*100)
    print("F1 Score : ",cv_scores["test_f1"].mean()*100)
    F1_score.append(cv_scores["test_f1"].mean()*100)
    print("ROC AUC : ",cv_scores["test_roc_auc"].mean()*100)
    ROC_AUC.append(cv_scores["test_roc_auc"].mean()*100)


# # Calculating the Performance metrices based on Bayesian Optimisation results

# In[ ]:


from lightgbm import LGBMClassifier
LGBM= LGBMClassifier(learning_rate='0.14905', max_depth='8', num_leaves='123',class_weight='balanced')
cv_results(LGBM)


# In[ ]:


from xgboost import XGBClassifier
XGB = XGBClassifier(scale_pos_weight=592,learning_rate=0.13,max_depth=9,n_estimators=78)
cv_results(XGB)


# In[ ]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(class_weight='balanced')
cv_results(LR)


# # Smote Analysis

# In[ ]:


from sklearn.utils import shuffle
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
X = data.drop('Class', axis = 1)
y = data['Class']
def cv_smote(model):
    pipeline1 = make_pipeline(SMOTE(sampling_strategy='minority', k_neighbors = 5, n_jobs=-1, random_state = 42),model)
    sk_fold = StratifiedKFold(n_splits=5)
    scoring = ["accuracy","precision","recall","f1","roc_auc"]
    cv_scores = cross_validate(model, X, y, scoring = scoring, cv = sk_fold)
    print("Accuracy : ",cv_scores["test_accuracy"].mean()*100)
    print("Precision : ",cv_scores["test_precision"].mean()*100)
    print("Recall : ",cv_scores["test_recall"].mean()*100)
    print("F1 Score : ",cv_scores["test_f1"].mean()*100)
    print("ROC AUC : ",cv_scores["test_roc_auc"].mean()*100)


# In[ ]:


from lightgbm import LGBMClassifier
lightgbm1 = LGBMClassifier()
cv_smote(lightgbm1)


# In[ ]:


from xgboost import XGBClassifier 
xgboost1 = XGBClassifier()
cv_smote(xgboost1)


# In[ ]:


from catboost import CatBoostClassifier
catboost1 = CatBoostClassifier(verbose=False)
cv_smote(catboost1)


# In[ ]:


LR1 = LogisticRegression(class_weight='balanced')
cv_smote(LR1)


# In[ ]:


from sklearn.ensemble import VotingClassifier

Model1 = [('lightgbm', LGBM), ('xgboost', XGB), ('catboost', catboost1)]
Model2 = [('lightgbm', LGBM), ('xgboost', XGB)]
Model3 = [('catboost', catboost1), ('xgboost', XGB)]
Model4 = [('lightgbm', LGBM), ('catboost', catboost1)]

voting1 = VotingClassifier(estimators=Model1,voting='soft')
voting2 = VotingClassifier(estimators=Model2,voting='soft')
voting3 = VotingClassifier(estimators=Model3,voting='soft')
voting4 = VotingClassifier(estimators=Model4,voting='soft')


# In[ ]:


cv_results(voting1)


# In[ ]:


cv_results(voting2)


# In[ ]:


cv_results(voting3)


# In[ ]:


cv_results(voting4)


# 
# # Undersampling
# 

# In[ ]:


from sklearn.utils import shuffle
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
X = data.drop('Class', axis = 1)
y = data['Class']
def cv_undersampler(model):
    pipeline2 = make_pipeline(RandomUnderSampler(sampling_strategy='majority',random_state = 42),model)
    sk_fold = StratifiedKFold(n_splits=5)
    scoring = ["accuracy","precision","recall","f1","roc_auc"]
    cv_values = cross_validate(model, X, y, scoring = scoring, cv = sk_fold)
    print("Accuracy : ",cv_values["test_accuracy"].mean()*100)
    print("Precision : ",cv_values["test_precision"].mean()*100)
    print("Recall : ",cv_values["test_recall"].mean()*100)
    print("F1 Score : ",cv_values["test_f1"].mean()*100)
    print("ROC AUC : ",cv_values["test_roc_auc"].mean()*100)


# In[ ]:


lightgbm2 = LGBMClassifier()
cv_undersampler(lightgbm2)


# In[ ]:


xgboost2 = XGBClassifier()
cv_undersampler(xgboost2)


# In[ ]:


catboost2 = CatBoostClassifier(verbose=False)
cv_undersampler(catboost2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
LR2 = LogisticRegression(class_weight='balanced')
cv_undersampler(LR2)


# In[ ]:


models =[LR,LGBM, XGB, catboost1, voting1, voting2, voting3, voting4]
names = ["LR","LGBM", "XGB", "CatBoost","Vot_Lg,Xg,Ca", "Vot_Lg,Xg","Vot_Xg,Ca", "Vot_Lg,Ca"] 


# In[ ]:


from sklearn.metrics import matthews_corrcoef
X = data.drop(['Class'] ,axis=1)
y = data['Class']
i=0
j=0 
for model in models:
    cv = StratifiedKFold(n_splits=5)
    MCC = []
    for train, test in cv.split(X, y):
        pred = model.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
        MCC.append(matthews_corrcoef(y.iloc[test], pred)) 
        i+=1
    mean_MCC = np.mean(MCC)
    print("MCC : " ,mean_MCC,"\n") 
    print("***************************************")  
    i=0
    j=j+1


# In[ ]:


MCC = [20.83693423123027,77.19305488939379,77.97430970688791, 71.43942448302167,80.39263914716475, 78.9252885937295,80.62463577641974,78.59387225376024]


# In[ ]:


Accuracy = np.array(Accuracy)
Precision = np.array(Precision)
Recall = np.array(Recall)
F1_score = np.array(F1_score)
MCC = np.array(MCC)
ROC_AUC = np.array(ROC_AUC)

Accuracy = np.round(Accuracy, 2)
Precision = np.round(Precision, 2)
Recall = np.round(Recall, 2)
F1_score = np.round(F1_score, 2)
MCC = np.round(MCC, 2)
ROC_AUC = np.round(ROC_AUC, 2)


# In[ ]:


barWidth = 0.1
fig = plt.subplots(figsize =(10, 6))

# Set position of bar on X axis
br1 = np.arange(len(F1_score))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
br7 = [x + barWidth for x in br6]

# Make the plot
plt.bar(br1, Precision, color ='r', width = barWidth,edgecolor ='grey', label ='Precision')
plt.bar(br2, MCC, color ='g', width = barWidth,edgecolor ='grey', label ='MCC')
plt.bar(br3, F1_score, color ='b', width = barWidth,edgecolor ='grey', label ='F1_score')
plt.bar(br4, Recall, color ='y', width = barWidth,edgecolor ='grey', label ='Recall')


# Adding Xticks
plt.xlabel('Algorithms', fontweight ='bold', fontsize = 15)
plt.ylabel('Performance_metrics_values', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(F1_score))],["LR","LGBM", "XGB", "CatBoost","Vot_Lg,Xg,Ca", "Vot_Lg,Xg", "Vot_Xg,Ca", "Vot_Lg,Ca"])
plt.title("Performance metrices of all the ML Algorithms",fontweight ='bold',fontsize = 22,color ='b')
plt.legend(bbox_to_anchor =(1.1,0.27), loc='lower center',title="Performance metrices")
plt.show()


# In[ ]:


colors = ['#ff1493','#8A2BE2', '#FFFF00', '#7FFFD4', '#8A2BE2','#800000', '#008B8B', '#FFFF00']
     

from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
X = data.drop(['Class'] ,axis=1)
y = data['Class']
f, axes = plt.subplots(ncols=1,nrows=1, figsize=(9,13))
linestyle=['-.','-.','-.','-.','','-.','','']
i=0
j=0
for model in models:
    cv = StratifiedKFold(n_splits=5)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        probas_ = model.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:,1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        i+=1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    axes.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#808A87', alpha=.8)
    axes.plot(mean_fpr, mean_tpr, color=colors[j],
            label= r'%s(AUC = %0.4f)'% (names[j], mean_auc),
            linestyle=linestyle[j], lw=2, alpha=0.9)
    xlim=[-0.05, 1.05]
    ylim=[-0.05, 1.05] 
    axes.legend(loc="lower right", prop={'size': 14})  
    i=0
    j=j+1
plt.xlabel('False Positive Rate',fontsize=14)
plt.title('ROC_AUC', fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14) 
plt.ylim([-0.01,1.01])  
plt.show()           


# In[ ]:


from prettytable import PrettyTable
columns = ["Algorithm_name","Accuracy", "Precision", "Recall", "F1","MCC","ROC_AUC"]
myTable = PrettyTable()
myTable._max_width = {"Algorithm_name":3,"Accuracy":4, "Precision":4, "Recall":4, "F1-Score":4,"MCC":6,"ROC_AUC":4}
# Add Columns
myTable.add_column(columns[0],["LR","LGBM","XGB","CatBoost","Vot_Lg,Xg,Ca", "Vot_Lg,Xg", "Vot_Xg,Ca", "Vot_Lg,Ca"])
myTable.add_column(columns[1], Accuracy)
myTable.add_column(columns[2], Precision)
myTable.add_column(columns[3], Recall)
myTable.add_column(columns[4], F1_score)
myTable.add_column(columns[5], MCC)
myTable.add_column(columns[6], ROC_AUC)
print(myTable)


# In[ ]:


undersampler = RandomUnderSampler(sampling_strategy='majority',random_state = 42)
X_resampled,y_resampled = undersampler.fit_resample(X,y)
X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.15,stratify=y_resampled,random_state=42)
light = LGBMClassifier()
light.fit(X_train,y_train)
y_pred = light.predict(X_test)


# In[ ]:


import pickle
filename='creditcard.sav'
pickle.dump(light,open(filename,'wb'))

loaded_model=pickle.load(open('creditcard.sav','rb'))

