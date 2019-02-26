from __future__ import division
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
print(tf.__version__)

import plot

import uproot
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

from keras import backend as k

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
k.tensorflow_backend.set_session(tf.Session(config=config))

### Add training and target variables here
inputVariables = ['gen_e','eta','pf_totalRaw','pf_ecalRaw','pf_hcalRaw']#,'pf_hoRaw']
targetVariable = 'gen_e'
###
### Get data from inputTree
inputTree = uproot.open("singlePi_histos_trees_valid.root")["t1"]
#print inputTree
dataset = inputTree.pandas.df(inputVariables)
dataset = dataset.dropna()
#print dataset[dataset['gen_e']>500]
### to directly compare to Raw
my_cut = (abs(dataset['eta'])<1.5) & (dataset['pf_totalRaw'] >0) & (dataset['gen_e']>0) & (dataset['gen_e']<500)
train_cut     = (dataset['pf_totalRaw']-dataset['gen_e'])/dataset['gen_e'] > -0.90 ## dont train on bad data with response of -1 
dataset = dataset[(my_cut) & (train_cut)]
dataset = dataset.reset_index()



##Prepare Data for training
#dataset['pf_ecalRaw'] = dataset['pf_ecalRaw']/dataset['pf_totalRaw']
#dataset['pf_hcalRaw'] = dataset['pf_hcalRaw']/dataset['pf_totalRaw']
#dataset['pf_hoRaw'] = dataset['pf_hoRaw']/dataset['pf_totalRaw']

## Mirror low E data across zero to help with performance
#mirror_Data = dataset[dataset['gen_e']<=50].copy()
#mirror_Data['gen_e'] = -mirror_Data['gen_e']
#mirror_Data['pf_totalRaw'] = - mirror_Data['pf_totalRaw']
#dataset = pd.concat([dataset,mirror_Data], ignore_index=True)
###
### define Test and Train Data as well as the target
temp_data = dataset[dataset['gen_e'] <= 200]
a = len(dataset[(dataset['gen_e']>200) & (dataset['gen_e']<=400)] )
b = len(dataset[dataset['gen_e']<=200])
temp_data = temp_data.sample(frac=int(a)/int(b),random_state=1)

#print 'tempData', temp_data

dataset = pd.concat([temp_data, dataset[dataset['gen_e']>200]],ignore_index=False)
#dataset = dataset.drop('index', axis=1)
compareData = dataset.copy()
#plot.plot_hist_compare(dataset['gen_e'],25,'test','test','test')

train_dataset = dataset.sample(frac=0.8,random_state=1)
#train_cut     = (train_dataset['pf_totalRaw']-train_dataset['gen_e'])/train_dataset['gen_e'] > -0.9 ## dont train on bad data with response of -1 
#train_dataset = train_dataset[train_cut]
#train_dataset = dataset.sample(n=1000, random_state=0)
test_dataset  = dataset.drop(train_dataset.index)
test_dataset  = test_dataset[test_dataset['gen_e']>=0]
#print len(train_dataset[train_dataset['gen_e']>=0]), len(train_dataset)

train_labels = train_dataset.pop(targetVariable)
#train_labels = np.log(train_labels/train_dataset['pf_totalRaw'])###########
train_labels = (train_dataset['pf_totalRaw']/train_labels)-1####################
#train_labels = np.log(train_labels)

test_labels  = test_dataset.pop(targetVariable)
#test_labels = np.log(test_labels/test_dataset['pf_totalRaw'])#############
test_labels = (test_dataset['pf_totalRaw']/test_labels)-1######################
#test_labels = np.log(test_labels)


###
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
### need to normalize the data due to scaling/range differences in variable set
def norm(data):
  return (data - train_stats['mean']) / train_stats['std']
def scale(data):
  x = data.values
  scaler = preprocessing.StandardScaler().fit(x)
  x_scaled = scaler.transform(x)
  scaledData = pd.DataFrame(x_scaled, columns=data.columns)
  return scaledData.copy()
def dropIndex(data):
  data = data.drop(columns='index')
  return data.copy()

normed_train_data = dropIndex(norm(train_dataset))
normed_test_data = dropIndex(norm(test_dataset))

scaled_train_data = dropIndex(scale(train_dataset))
scaled_test_data = dropIndex(scale(test_dataset))


#train_labels = train_labels/np.amax(train_labels)
#test_max = np.amax(test_labels)
#test_labels  = test_labels/test_max

############################################
### Build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(4, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001), input_shape=[len(scaled_train_data.keys())]),# l1 0000
    #keras.layers.Dropout(0.3), # 0
    layers.Dense(8, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1(0.0001)),#15, l1 =.001
    keras.layers.Dropout(0.5),#0 
    layers.Dense(4, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1(0.0001)), #10, l1 = .01
    #keras.layers.Dropout(0.5), # .49
    layers.Dense(1)
  ])
  #optimizer = tf.train.RMSPropOptimizer(0.001)
  optimizer  = 'adam'
  
  def RMSLE(y_true,y_pred):
    first_log = math_ops.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = math_ops.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(math_ops.square(first_log - second_log), axis=-1))
  
  model.compile(loss='mse', # mae
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

###########################################
### Train the model
EPOCHS = 10 # 30

history = model.fit(
  scaled_train_data, train_labels,
  epochs=EPOCHS, batch_size=256, # 10k
  validation_data=(scaled_test_data,test_labels),
  verbose=1)
###########################################
####Analyze the training history  
#keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plot.plot_history(history,hist)

###################################
loss, mae, mse = model.evaluate(scaled_test_data, test_labels, verbose=1)

print("Testing set Mean Abs Error: {:5.2f} E".format(mae))
###################################
### Predict using test Dataset
### (reverse feature prep here) 
test_predictions = model.predict(scaled_test_data).flatten()

###Recover meaningful predictions
#test_predictions = np.exp(test_predictions)*test_dataset['pf_totalRaw']###############
test_predictions = 1/((test_predictions+1)/test_dataset['pf_totalRaw'])#######################
#test_predictions = np.exp(test_predictions)
#test_predictions = test_predictions*test_max

#test_labels = np.exp(test_labels)*test_dataset['pf_totalRaw']#########################
test_labels = 1/((test_labels+1)/test_dataset['pf_totalRaw'])#################################
#test_labels = np.exp(test_labels)
#test_labels = test_labels*test_max

###################################
results = pd.DataFrame({'gen_e':test_labels,'DNN':test_predictions})

### analysis of the performance ###

fig = plt.figure(figsize=(16, 8))
#plot_perf(results,None,"All")
plot.plot_perf(results, None, "Pred vs True")
#[plot_perf(results,
#           (results['gen_e']>(i*50)) & (results['gen_e']<((i+1)*50)),
#           "label[pt]>"+str(i*50)+" & label[pt]<"+str((i+1)*50)) for i in range(0,11)]
def lin_regressor(x,y):
  linear_regressor = LinearRegression()
  linear_regressor.fit(x.values.reshape(-1, 1),y.values.reshape(-1, 1))
  fit = linear_regressor.predict(x.values.reshape(-1, 1))
  mpl.rcParams['agg.path.chunksize'] = 10000
  plt.plot(x.values.reshape(-1, 1),fit,'--k')
lin_regressor(results['gen_e'][results['gen_e']<=200],results['DNN'][results['gen_e']<=200])
lin_regressor(results['gen_e'][(results['gen_e']<=500) & (results['gen_e']>200)],results['DNN'][(results['gen_e']<=500) & (results['gen_e']>200)])
fig.savefig("model_acc.png")
plt.show()
plt.clf()
plt.close()

###compare performance of the networks

### compare resonpse ###
res_DNN   = (compareData['pf_totalRaw']-compareData['gen_e'])/compareData['gen_e']
res_Keras = (test_predictions-test_labels)/test_labels
plot.plot_hist_compare([res_DNN,res_Keras],100,['Raw','Keras'],"(Pred-True)/True [E]","perf_comparison.pdf")
### compare pt distribution ###
plot.plot_hist_compare([compareData['pf_totalRaw'],test_predictions],25,['Raw','Keras'],"E","pt_comparison.pdf")

##### testing profile plot in python

### Pred/True vs True ###
plot.profile_plot_compare(compareData['gen_e'], compareData['pf_totalRaw']/compareData['gen_e'], 'Raw',
                     test_labels, test_predictions/test_labels, 'Keras',
                     100, 0, 500,
                     "True [E]", "Pred/True [E]", "scale_comparison.pdf")
### Response vs True ###
plot.profile_plot_compare(compareData['gen_e'], (compareData['pf_totalRaw']-compareData['gen_e'])/compareData['gen_e'], 'Raw',
                     test_labels, (test_predictions-test_labels)/test_labels, 'Keras',
                     100, 0, 500,
                     "True [E]", "(Pred-True)/True [E]", "response_comparison.pdf")

##### Debug low pt #####
if False:
  for i, label in enumerate(test_labels):
    if label < 10:
      print (label, test_predictions[i])
