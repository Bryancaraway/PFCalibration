from __future__ import division
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow import set_random_seed
set_random_seed(2)
print(tf.__version__)

import plot

import pandas as pd
import numpy as np
np.random.seed(0)

from tensorflow.python.keras import backend as K
from keras import backend as k
from tensorflow.python.ops import math_ops
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def AllocateVRam():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.5
    k.tensorflow_backend.set_session(tf.Session(config=cfg))


def Build_Model(N_inputs, N_outputs, N_epochs, train_data):
    main_input = keras.layers.Input(shape=[N_inputs], name='main_input')
    layer = keras.layers.Lambda(lambda x: (x - K.constant(train_data.mean().values)) / K.constant(train_data.std().values), name='normalizeData')(main_input)
    layer = keras.layers.Dropout(0.0)(layer)
    layer = keras.layers.Dense(2*N_inputs, activation='relu', kernel_regularizer=keras.regularizers.l2(0.000))(layer)
    layer = keras.layers.Dropout(0.0)(layer)
    layer = keras.layers.Dense(2*N_inputs, activation='relu', kernel_regularizer=keras.regularizers.l1(0.000))(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    first_output = keras.layers.Dense(1, activation='linear', name='first_output')(layer)
    second_output = keras.layers.Dense(3, activation='softmax', name='second_output')(layer)

    model = keras.models.Model(inputs=main_input, outputs=[first_output, second_output], name='model')
    
    optimizer  = keras.optimizers.Adam(lr=.001, decay= .001 / N_epochs) # lr = .0001, decay = .0001 / 200
  
    def RMSLE(y_true,y_pred):
        first_log = math_ops.log(K.clip(y_pred, K.epsilon(), None) + 1.)
        second_log = math_ops.log(K.clip(y_true, K.epsilon(), None) + 1.)
        return K.sqrt(K.mean(math_ops.square(first_log - second_log), axis=-1))
  
    def MSER(y_true,y_pred):
        return K.mean(K.square(y_pred-y_true)+K.abs(y_pred-y_true),axis=-1)

    def WMSER(y_true,y_pred):
        return K.mean(K.abs(K.pow(y_pred-y_true,2)/y_pred), axis=-1)

    model.compile(loss=['mse','sparse_categorical_crossentropy'], # mse
                  optimizer=optimizer,
                  metrics=['mae', 'mse', 'acc'])
    return model

def doMachineLearn(train_data, train_labels, test_data, test_labels):

    
    ### Normalie Data

    def norm(data):
        data_stats = data.describe()
        data_stats = data_stats.transpose()
        return (data - data_stats['mean']) / data_stats['std']
    def dropIndex(data):
        data = data.drop(columns='index')
        return data.copy()
    def dataStats(data):
        data_stats = data.describe()
        data_stats = data_stats.transpose()
        return data_stats['mean'], data_stats['std']

    train_data = dropIndex(train_data) 
    test_data  = dropIndex(test_data)
    print "Training Variables: " + str(train_data.keys())
    print "Target Variables: " + str(train_labels.keys())
    #### Train Model

    EPOCHS = 200;
    model = Build_Model(len(train_data.keys()), len(train_labels.keys()), EPOCHS, train_data)

    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20) # 10
    history = model.fit(
        train_data, [train_labels['gen_e'],train_labels['type']], # target is the response
        epochs=EPOCHS, batch_size=512, # 5128
        validation_data=(test_data,[test_labels['gen_e'],test_labels['type']]),
        verbose=1) #callbacks=[early_stop])
    
    # Save trainig model as a protocol buffers file
    inputName = model.input.op.name.split(':')[0]
    outputName = model.output[0].op.name.split(':')[0]
    print "Input name:", inputName
    print "Output name:", outputName
    saver = tf.train.Saver()
    outputDir = "TrainOutput" ############### fix this
    saver.save(keras.backend.get_session(), outputDir+"/keras_model.ckpt")
    export_path="./"+outputDir+"/"
    freeze_graph_binary = "python freeze_graph.py"
    graph_file=export_path+"keras_model.ckpt.meta"
    ckpt_file=export_path+"keras_model.ckpt"
    output_file=export_path+"keras_frozen.pb"
    command = freeze_graph_binary+" --input_meta_graph="+graph_file+" --input_checkpoint="+ckpt_file+" --output_graph="+output_file+" --output_node_names="+outputName+" --input_binary=true"
    os.system(command)

    ####Analyze the training history  
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plot.plot_history(history,hist) ######## FIX ME ########
        
    #loss, mae, mse, acc = model.evaluate(test_data, [test_labels['gen_e'],test_labels['EH_Hadron'],test_labels['H_Hadron']], verbose=1)
    #print("Testing set Mean Abs Error: {:5.3f} E".format(mae))

    (test_predictions1, test_predictions2) = model.predict(test_data)
    test_predictions = test_predictions1.ravel()
    
    return test_predictions
