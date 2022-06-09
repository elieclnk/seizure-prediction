#!/usr/bin/env python
# coding: utf-8

# # Seizure classification

# ## Import modules

# In[1]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import scipy
import tensorflow as tf
import keras
import mne
from scipy import signal
import seaborn as sns
from sklearn.preprocessing import RobustScaler, Normalizer, MinMaxScaler
from mne.time_frequency import psd_array_multitaper

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score, precision_score, f1_score

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import xgboost as xgb

import warnings
plt.rcParams["figure.figsize"] = [20,12]


# ## Load the dataset

# In[6]:


train_fname = "train_data.csv"
train_info = "train_info.csv"

df_train = pd.read_csv(train_fname)
df_info = pd.read_csv(train_info)


# In[7]:


df_train.head()


# ## Visualize the dataset

# In[8]:


df_train.describe()


# In[9]:


## We can look at the distribution of the data for each patient
df_train.hist(bins=100)


# We observe different distributions of data across patients. We will need to account for this and address it with different preprocessing techniques, such as scaling methods.
# 
# Something else we can notice is the presence of extreme values for some of the patients (eg. train_01, train_10, train_14 etc.). We will need to understand if these values should be removed or if they belong to a certain pattern we could use for seizure classification.
# 
# We can now check if there are any missing values.  

# In[10]:


df_train.isnull().sum()


# We don't observe any missing or NaN values in the dataset, we can go to the next step.

# In[11]:


df_info


# ### Let's visualize a seizure to understand what it looks like

# In[12]:


seizure_files = ['train_01', 'train_03', 'train_06', 'train_07', 'train_10', 'train_12', 'train_14', 'train_15']

plt.rcParams["figure.figsize"] = [20,12]
fig, axes = plt.subplots(4,2, sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    eeg_recording = df_train[seizure_files[i]]

    start = int(df_info['t_start'][df_info['file_id']==seizure_files[i]])*512
    end = int(df_info['t_end'][df_info['file_id']==seizure_files[i]])*512

    seizure_signal = eeg_recording.copy()
    baseline_signal = eeg_recording.copy()

    baseline_signal[start:end]=np.nan
    ax.plot(eeg_recording, color='tab:orange')
    ax.plot(baseline_signal, color='tab:blue')
    
plt.show()


# The visualization of the seizures allows us to see if we can discriminate the seizures at first glance. This is obviously not possible for all seizures and we can notice the diversity of seizures' shapes.

# ## Create the target data
# 
# I am going to create a vector with the label "0" with there is no seizure and "1" if the seizure is happening.

# In[13]:


def create_target_data(df, df_info):
    fs = 512   
    target_tot = []
    patients = df_info['file_id']
    seizure_lengths=[]
    for patient in patients:
        target = [0]*len(df)
        if df_info.loc[df_info['file_id'] == patient, 'sz_present'].iloc[0] == 1:
            start = int(df_info.loc[df_info['file_id'] == patient, 't_start'].iloc[0])*fs
            end = int(df_info.loc[df_info['file_id'] == patient, 't_end'].iloc[0])*fs
            seizure_lengths.append(end-start)
            target[start:end]=[1]*(end-start)
        target_tot.append(target)
    print(f'minimum seizure time = {min(seizure_lengths)/512} s, maximum seizure time = {max(seizure_lengths)/512} s')
    return target_tot

target_tot = create_target_data(df_train, df_info)


# We print the min and max of the seizure lengths. This will be usefull later on to create appropriate window sizes.

# In[14]:


fig, axes = plt.subplots(3,5, sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    ax.plot(target_tot[i])
    ax.set_title(df_info['file_id'].loc[i])
plt.suptitle('Presence of seizure')
plt.show()


# ## Frequency features

# We can take a look at the power spectral density

# In[15]:


plt.rcParams["figure.figsize"] = [20,12]
from scipy.signal import welch
freqs, psd = welch(df_train, 
                   fs=512, 
                   axis=0)
psd = pd.DataFrame(psd, index = freqs, columns = df_train.columns)
psd.head()
sns.lineplot(data=psd)
plt.xlabel('Hz')
plt.ylabel('Power spectral density')


# #### We can observe the big spike around 60hz for train_13. We can either remove this training data or run a Notch filter to remove the band between 50 and 65Hz, which corresponds to the AC line frequency.

# In[16]:


df_train = df_train.drop('train_13', axis=1)


# In[17]:


df_train.head()


# ## Filtering tools

# In[21]:


from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfiltfilt(sos, data)
    return y

def filter_dataset(df, fs, fhigh):
    df=df.copy()
    lowcut=0.1
    highcut=fhigh
    fs=fs
    columns = df.columns
    for col in columns:
        df[col] = butter_bandpass_filter(df[col], lowcut, highcut, fs, order=1)
    return df

df_train_filt = df_train


# In[22]:


plt.rcParams["figure.figsize"] = [20,12]
from scipy.signal import welch
freqs, psd = welch(df_train_filt, 
                   fs=512, 
                   axis=0)
psd = pd.DataFrame(psd, index = freqs, columns = df_train.columns)
psd.head()
sns.lineplot(data=psd)
plt.xlabel('Hz')
plt.ylabel('Power spectral density')


# In[23]:


df_train_filt.hist(bins=100)


# ## Windowing

# In[25]:


def threshold(x):
    if x>0.5:
        return 1
    elif x==0:
        return 0
    else:
        return None
## Cut windows before concatenating otherwise we will have overlaps between patients within the same window. 
def create_window_dataset_before_concat(df,tar):
    fs=512
    window_size = 18*fs # todo : run optimization on the window selected.
    shift = int(window_size/9)
    data_X = []
    data_Y = []
    for i, col in enumerate(df.columns):
        X=np.array(df[col])
        y=tar[i]
        for i in range(0, int(len(X)-window_size-1), shift):
            thresh = threshold(np.mean(y[i:i+window_size]))
            if thresh is not None:
                data_X.append(X[i:i+window_size])
                data_Y.append(thresh)
            else:
                continue
    return np.array(data_X), np.array(data_Y)

X_wind, y_wind = create_window_dataset_before_concat(df_train_filt,target_tot)

print(X_wind.shape, y_wind.shape)


# In[26]:


plt.rcParams["figure.figsize"] = [8,6]
sns.countplot(y_wind)
print('ratio=',np.sum(y_wind/len(y_wind)))


# We observe an imbalanced dataset here. This can be dealt with using augmentation or reduction techniques as well as using appropriate machine learning algorithms.

# ## Feature Engineering

# ### Delta, theta, alpha, beta, gamma power

# In[27]:


from scipy.signal import welch
def get_mean_power_freq(sign, band):
    sample_rate = 512 # in hz
    low = 0.5
    nperseg = (2 / low)*sample_rate
    freqs, psd = welch(sign, sample_rate,nperseg = nperseg, scaling='density' )
    plt.plot(freqs, psd)
    l=[psd[i] for i, f in enumerate(freqs) if f>band[0] and f<band[1]]
    return np.mean(l)

def create_power_features(sign):
    bandpass_data=[]
    bandpass_name=[]
    bandpasses = [[[0.1,4],'power_delta'],
                  [[4,8],'power_theta'],
                  [[8,12],'power_alpha'],
                  [[12,30],'power_beta'],
                  [[30,70],'power_gamma']
                 ]
    for bandpass, freq_name in bandpasses:
        bandpass_data.append(get_mean_power_freq(sign, bandpass))
        bandpass_name.append(freq_name)
    dico = dict(zip(bandpass_name, bandpass_data))
    
    return pd.DataFrame.from_dict(dico, orient='index').T

def create_df(data):
    tot=pd.DataFrame()
    for d in data:
        df = create_power_features(d)
        
        if tot.empty:
            tot = df

        else:
            tot = pd.concat([tot, df])
    return tot

power_df = create_df(X_wind)


# ### Discrete Wavelets

# In[28]:


from pywt import wavedec

def create_wavelets(data):
    fs=512
    level = 6
    frequencies = np.array([fs]*level)
    j = [1<<exponent for exponent in range(level)]
    frequencies = frequencies/j
    coeffs_list = wavedec(data, wavelet='db4', level=level)
    #############################
    nums = list(range(1,level+1))
    names=[]
    for num in nums:
        names.append('D' + str(num))
    names.append('A' + str(nums[-1]))
    names = names[::-1] 
    ###########################
    wavelets = pd.DataFrame()
    for i, array in enumerate(coeffs_list):
        level_df = pd.DataFrame(array)
        level_df.columns = [names[i]]
        if wavelets.empty:
            wavelets = level_df
        else:
            wavelets = pd.concat([wavelets,level_df], axis=1)
    wavelets = wavelets.drop(f'A{str(level)}', axis=1)

    return wavelets, frequencies

def create_abs_mean_from_wavelets_df(data):
    data = data.copy()
    df = pd.DataFrame(data.abs().mean()).T
    df = df.rename(columns={"D6": "D6_abs_mean", "D5": "D5_abs_mean", 
                       "D4": "D4_abs_mean", "D3": "D3_abs_mean",
                       "D2": "D2_abs_mean", "D1": "D1_abs_mean"})
    return df
def create_mean_from_wavelets_df(data):
    data = data.copy()
    df = pd.DataFrame(data.mean()).T
    df = df.rename(columns={"D6": "D6_mean", "D5": "D5_mean", 
                           "D4": "D4_mean", "D3": "D3_mean",
                           "D2": "D2_mean", "D1": "D1_mean"})
    return df
def create_std_from_wavelets_df(data):
    data = data.copy()
    df = pd.DataFrame(data.std()).T
    df = df.rename(columns={"D6": "D6_std", "D5": "D5_std", 
                           "D4": "D4_std", "D3": "D3_std",
                           "D2": "D2_std", "D1": "D1_std"})
    return df

def create_logsum_from_wavelets_df(data):
    data = data.copy()
    df = pd.DataFrame(data.abs().sum()).T
    df = df.apply(np.log)
    df = df.rename(columns={"D6": "D6_logsum", "D5": "D5_logsum", 
                           "D4": "D4_logsum", "D3": "D3_logsum",
                           "D2": "D2_logsum", "D1": "D1_logsum"})
    return df
            
def create_wavelet_df(data):
    tot=pd.DataFrame()
    for d in data:
        wavelets_df, _ = create_wavelets(d)
        df_std = create_std_from_wavelets_df(wavelets_df)
        df_abs = create_abs_mean_from_wavelets_df(wavelets_df)
        df_mean = create_mean_from_wavelets_df(wavelets_df)
        df_logsum = create_logsum_from_wavelets_df(wavelets_df)
        df = pd.concat([df_abs], axis=1)
        if tot.empty:
            tot = df
        else:
            tot = pd.concat([tot, df])
    return tot
        
wave_df = create_wavelet_df(X_wind)


# In[29]:


wave_df = wave_df.reset_index(drop=True)
power_df = power_df.reset_index(drop=True)
wave_df.head()


# In[30]:


print(power_df.shape)
print(wave_df.shape)
final_df = power_df.join(wave_df)
final_df.head()


# ### Features correlation

# In[31]:


corr = final_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# ## Create train/test data

# ### Dealing with imbalanced dataset

# In[25]:


import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


# In[27]:


over = SMOTE(sampling_strategy=1) #We could overfit if we adopt a strategy of more than 0.6
X_train, y_train = over.fit_resample(X_train, y_train)


# In[28]:


sns.countplot(y_train)


# # Classic Machine Learning

# Here we can test a bunch of classifiers like :
# 
# - K Nearest Neighbors
# - Logistic Regression
# - Ridge Regression
# - Stochastic Gradient Descent 
# - Naive Bayes, Decision Trees
# - Random Forest
# - Extreme Random Forest (ExtraTrees)
# - Gradient Boosting
# - Extreme Gradient Boosting (XGBoost).

# ## Train the classical models

# In[30]:


from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
# Logistic Regression
RANDOM_STATE=42

pipe_reg = Pipeline([('scl', StandardScaler()),
                     ('clf', LogisticRegression(class_weight='balanced',
                                                solver = 'liblinear',
                                                random_state=RANDOM_STATE))])

# Support Vector Machine
pipe_svc = Pipeline([('scl', StandardScaler()),
                    ('clf', SVC(kernel='rbf', 
                                class_weight = 'balanced',
                                probability=True,
                                random_state=RANDOM_STATE))])

# Decision Tree
DT = DecisionTreeClassifier(random_state=RANDOM_STATE)

# K-Nearest Neighbours
pipe_kkn = Pipeline([('scl', StandardScaler()),
                    ('clf', KNeighborsClassifier())])

xgb_clf = xgb.XGBClassifier()

# list of classifier names
classifier_names = ['Logistic Regression', 'Support Vector Machine', 
                    'Decision Tree', 'K-Nearest Neighbors', 'XGBoost']

# list of classifiers
classifiers = [pipe_reg, pipe_svc, DT, pipe_kkn, xgb_clf]

# fit all the classifiers to the training data
for classifier in classifiers:
    classifier.fit(X_train, y_train)


# ## Visualize the results

# In[68]:


from sklearn.metrics import multilabel_confusion_matrix
def get_preds(threshold, probabilities):
    return [1 if prob > threshold else 0 for prob in probabilities]

def plot_roc_auc(model, X_test, y_test):
    
    probas = model.predict_proba(X_test)[:, 1]
    
    roc_values = []
    
    for thresh in np.linspace(0, 1, 100):
        preds = get_preds(thresh, probas)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        roc_values.append([tpr, fpr])
        
    tpr_values, fpr_values = zip(*roc_values)
    
    plt.plot(fpr_values, tpr_values)
    plt.plot(np.linspace(0, 1, 100),
             np.linspace(0, 1, 100),
             label='baseline',
             linestyle='--')
    
    plt.title('Receiver Operating Characteristic Curve', fontsize=18)
    plt.ylabel('TPR', fontsize=16)
    plt.xlabel('FPR', fontsize=16)
    plt.legend(fontsize=12)
    
def plot_conf_mat(y_test, y_pred, title):
    cf_matrix = confusion_matrix(y_test, y_pred)
    cf_matrix_pct = (cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis])*100
    
    auc = round(roc_auc_score(y_test,y_pred),2)
    f1 = round(f1_score(y_test, y_pred), 2)
    acc = round(accuracy_score(y_test, y_pred), 2)
    recall = round(recall_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred), 2)

    group_names = ["Baseline state \n correctly detected \n","False detection \n","Seizure not detected\n","Seizure detected\n"]
    group_counts = [f"{round(pct,2)}%\n{round(value,2)}" for value,pct in
                    zip(cf_matrix.flatten(), cf_matrix_pct.flatten())]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_names,group_counts)]

    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix_pct, annot=labels, fmt="", cmap="Blues")

    plt.figtext(0.45, -0.05, f"auc = {auc}\n f1 score = {f1}\n accuracy score = {acc} \n recall score = {recall} \n precision score = {precision}", ha="center", fontsize=10)
    
    plt.title(title)
    plt.show()


# In[69]:


for classifier, name in zip(classifiers, classifier_names):  
    y_pred = classifier.predict(X_test)
    plot_conf_mat(y_test, y_pred, title=name)


# In[44]:


from sklearn.metrics import roc_curve, roc_auc_score, auc
import re

def ROC(classifiers, classifier_names, X_train, 
        X_val, y_train, y_val):
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, classifer in enumerate(classifiers):
        probas = classifer.fit(X_train,
                                    y_train).predict_proba(X_val)
        fpr, tpr, thresholds = roc_curve(y_val, 
                                         probas[:, 1], 
                                         pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, 
                 tpr, 
                 lw=1, 
                 label=f'{classifier_names[i]} (area = {round(roc_auc,2)})')

    plt.plot([0, 1], 
           [0, 1], 
           linestyle='--', 
           label='random guessing')
    
    plt.plot([0, 0, 1], 
           [0, 1, 1], 
           lw=2, 
           linestyle=':', 
           label='perfect performance')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic on Validation Set')
    plt.legend(loc="lower right", fontsize ='medium')

    plt.tight_layout()
    plt.show()    


ROC(classifiers, classifier_names, X_train, X_test, y_train, y_test)


# ### Selected classifier : Xgboost

# #### For xgboost we are going to run a cross validation

# In[47]:


from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import plot_importance

kf = KFold(n_splits=4)
scores = []
kf.get_n_splits(X, y)

for train_index, test_index in kf.split(X):
    
    X_train_chunk, X_test_chunk = X[train_index], X[test_index]
    
    y_train_chunk, y_test_chunk = y[train_index], y[test_index]
    
    print(f'y train ratio = {np.sum(y_train_chunk)/len(y_train_chunk)*100}%')
    print(f'y test ratio = {np.sum(y_test_chunk)/len(y_test_chunk)*100}%')
    model = xgb.XGBClassifier(n_estimators=100,
                              min_child_weight = 1,
                              max_depth = 3,
                              learning_rate= 0.2)
    
    model.fit(X_train_chunk, y_train_chunk)
    y_pred=model.predict(X_test_chunk)
    
    auc = round(roc_auc_score(y_test_chunk,y_pred),3)
    
    scores.append(auc)
    
    print(f'AUC score = {auc}')
    
    
    plt.figure(figsize=(8, 16))
    
    plt.subplot(3,1,1)
    plot_roc_auc(model, X_test_chunk, y_test_chunk)

    plt.subplot(3,1,2)
    plt.barh(final_df.columns, model.feature_importances_)

    plt.subplot(3,1,3)
    cf_matrix = confusion_matrix(y_test_chunk, y_pred)
    cf_matrix_pct = (cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis])*100
    
    f1 = round(f1_score(y_test_chunk, y_pred), 2)
    acc = round(accuracy_score(y_test_chunk, y_pred), 2)
    recall = round(recall_score(y_test_chunk, y_pred), 2)
    precision = round(precision_score(y_test_chunk, y_pred), 2)
    

    group_names = ["Baseline state \n correctly detected \n","False alarm \n","Seizure not detected\n","Seizure detected\n"]
    group_counts = [f"{round(pct,2)}%\n{round(value,2)}" for value,pct in
                    zip(cf_matrix.flatten(), cf_matrix_pct.flatten())]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_names,group_counts)]

    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix_pct, annot=labels, fmt="", cmap="Blues")
    plt.figtext(0.45, 0.05, f"auc score = {auc}\n\n f1 score = {f1}\n accuracy score = {acc} \n recall score = {recall} \n precision score = {precision}", ha="center", fontsize=11)
    plt.show()


# In[35]:


print('AUC scores : ', scores)
print('The mean AUC score accross the 5 fold validation is', round(np.mean(scores), 3))


# In[54]:


model_final = xgb.XGBClassifier(n_estimators=100,
                              min_child_weight = 1,
                              max_depth = 3,
                              learning_rate= 0.2)
model_final.fit(X,y)


# In[61]:


### Final pipeline
## input : 10 min EEG data

def create_window_pipeline(X_val):
    fs=512
    window_size = 18*fs # todo : run optimization on the window selected.
    shift = int(window_size/9)
    data_X = []
    X=X_val
    for i in range(0, int(len(X)-window_size-1), shift):
            data_X.append(X[i:i+window_size])
    return np.array(data_X)
        
def create_df(data):
    tot=pd.DataFrame()
    for d in data:
        df_power = create_power_features(d)
        wavelets_df, _ = create_wavelets(d)
        df_wave_abs = create_abs_mean_from_wavelets_df(wavelets_df)
        df = df_power.join(df_wave_abs)
        if tot.empty:
            tot = df
        else:
            tot = pd.concat([tot, df])
    return tot

X_val = df_train['train_14']

# def predict_pipeline(X_val):
X_wind = create_window_pipeline(X_val)
df_val=create_df(X_wind)
df_val.shape


# In[66]:


y_pred_final = model.predict_proba(df_val)[:,1]


# In[67]:


plt.plot(y_pred_final)


# ## Hyperparameter tuning

# The results for the Xgboost classifiers are very promising as we come up with a result of 0.881 AUC.
# 
# Let's run a step of hyperparameter tuning and see if we can improve the accuracy of the classifier.

# In[36]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

params = {
    'n_estimators': [100, 400, 800],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.20],
    'min_child_weight': [1, 10, 100]
    }

xgbr = xgb.XGBClassifier()

clf = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params,
                         scoring='roc_auc',
                         n_iter=30,
                         verbose=5)

clf.fit(X, y)


# In[40]:


print("Best parameters:", clf.best_params_)
print(clf.best_score_)


# ## MLP

# In[38]:


from sklearn.neural_network import MLPClassifier

kf = KFold(n_splits=5)
kf.get_n_splits(X)
scores = []

for train_index, test_index in kf.split(X):
    
    print(len(train_index), len(test_index))
    
    X_train_chunk, X_test_chunk = X[train_index], X[test_index]
    
    y_train_chunk, y_test_chunk = y[train_index], y[test_index]
    
    model = MLPClassifier(hidden_layer_sizes=(100,100), activation='relu',
                          solver='adam', alpha=0.001, batch_size='auto', learning_rate='adaptive')
    
    model.fit(X_train_chunk, y_train_chunk)
    y_pred=model.predict(X_test_chunk)
    
    auc = round(roc_auc_score(y_test_chunk,y_pred),3)
    
    scores.append(auc)
    
    print(f'AUC score = {auc}')
    
    
    plt.figure(figsize=(10, 10))
    
    plt.subplot(2,1,1)
    plot_roc_auc(model, X_test_chunk, y_test_chunk)
    
    plt.subplot(2,1,2)
    cf_matrix = confusion_matrix(y_test_chunk, y_pred)
    cf_matrix_pct = (cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis])*100

    
    f1 = round(f1_score(y_test_chunk, y_pred), 2)
    acc = round(accuracy_score(y_test_chunk, y_pred), 2)
    recall = round(recall_score(y_test_chunk, y_pred), 2)
    precision = round(precision_score(y_test_chunk, y_pred), 2)
    

    group_names = ["Baseline State \n correctly detected \n","False alarm \n","Seizure not detected\n","Seizure detected\n"]
    group_counts = [f"{round(pct,2)}%\n{round(value,2)}" for value,pct in
                    zip(cf_matrix.flatten(), cf_matrix_pct.flatten())]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_names,group_counts)]

    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix_pct, annot=labels, fmt="", cmap="Blues")
    plt.figtext(0.45, -0.05, f"auc score = {auc}\n\n f1 score = {f1}\n accuracy score = {acc} \n recall score = {recall} \n precision score = {precision}", ha="center", fontsize=11)
    plt.show()


# In[39]:


print('AUC scores : ', scores)
print('The mean AUC score accross the 5 fold validation is', round(np.mean(scores), 3))


# #### CONCLUSION: We obtain an AUC of 0.76 for the XGboost classifier and 0.74 the multi layer perceptron classifier, our 2 best classifiers so far.
