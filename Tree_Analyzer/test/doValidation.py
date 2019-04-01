from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
np.random.seed(0)
import tensorflow as tf
print(tf.__version__)
import os
import pandas as pd
import math

import plot
import process_data

inputVariables = ['eta', #'charge']#,'pf_hoRaw'], "p", 'pt'
                  'phi', 
                  'pf_totalRaw','pf_ecalRaw','pf_hcalRaw']

targetVariables = ['gen_e','type'] ##### type corresponds to: 1 == E Hadron, 2 == EH Hadron, 3 == H Hadron
inputFiles = ["singlePi_histos_trees_corr_samples.root"]
#inputFiles = ["singlePi_histos_trees_new_samples.root","singlePi_histos_trees_valid.root"]

### Get data from inputTree
dataset, compareData = process_data.Get_tree_data(inputFiles,
                                                  inputVariables, targetVariables,               
                                                  withTracks = True, withDepth = True,
                                                  endcapOnly = False, barrelOnly = False,
                                                  withCorr = True, isTrainProbe = False)
train_data, test_data, train_labels, test_labels = process_data.PreProcess(dataset, targetVariables)

def makePrediction(test_data):
    graph_def = tf.GraphDef()

    # These are set to the default names from exported models, update as needed.
    filename = 'TrainOutput/keras_frozen.pb'

    # Import the TF graph
    with tf.gfile.GFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    input_node = 'main_input:0'
    output_layer = 'first_output/BiasAdd:0'
    def dropIndex(data):
        data = data.drop(columns='index')
        return data.copy()
    with tf.Session() as sess:
    
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        test_predictions = sess.run(prob_tensor, {input_node: dropIndex(test_data).values })

    test_predictions = test_predictions.ravel()
    return test_predictions

test_predictions = makePrediction(test_data)
##############################################################
##########Recover meaningful predictions
test_predictions, test_labels = process_data.PostProcess(test_predictions, test_data, test_labels)

##############################################################
########## TRAINING ANALYSIS 
results = test_data.copy()
for variable in targetVariables:
    results[variable] = test_labels[variable]

results['DNN'] = test_predictions



################# START OF ------- TEST FOR KEN **** REMOVE LATER~~~~~~~~~~~!!!!!
#compareData = test_data.copy()
##add_access = 100
#compareData['pf_ecalRaw'] = compareData['pf_ecalRaw']*1.5
#compareData['pf_hcalRaw'] = compareData['pf_hcalRaw']*1.5
#compareData['pf_totalRaw'] = compareData['pf_ecalRaw'] + compareData['pf_hcalRaw']
#compareData['Response'] = makePrediction(compareData)
#for variable in targetVariables:
#    compareData[variable] = test_labels[variable]
#plot.plot_hist_compare([compareData['Response']*compareData['gen_e'],(results['DNN']-results['gen_e'])],200,-200,200,['test','Keras'],"Pred-True [E]", "pdf/test_comparison.pdf")
################# END OF ------- TEST FOR KEN **** REMOVE LATER~~~~~~~~~~~!!!!! 

compareData['Response'] = (compareData['pf_totalRaw'] - compareData['gen_e'])/compareData['gen_e']
results['Response'] = (results['DNN']-results['gen_e'])/results['gen_e']

##############################################
########## PLOT MAKING #######################


plot.plot_perf(results, None, "Pred vs True")


plot.plot_hist_compare([compareData['Response'],results['Response']],100, -1.2,1.2,['PF_Corr','Keras'],"(Pred-True)/True ","pdf/perf_comparison.pdf")
plot.plot_hist_compare([compareData['Response'],(results['DNN']-results['p'])/results['p']],100, -1.2,1.2,['PF_Corr','Keras'],"(Pred-p)/p","pdf/perf_comparison_p.pdf")
### compare pt distribution ###
plot.plot_hist_compare([compareData['pf_totalRaw'],results['DNN']],25,0,550,['PF_Corr','Keras'],"E","pdf/pt_comparison.pdf")

### Pred/True vs True ###
#plot.profile_plot_compare(compareData['gen_e'], compareData['pf_totalRaw']/compareData['gen_e'], 'Raw',
#                     test_labels, test_predictions/test_labels, 'Keras',
#                     100, 0, 500,
#                     "True [E]", "Pred/True [E]", "scale_comparison.pdf")
### Response vs True ###
plot.profile_plot_compare(compareData['gen_e'], compareData['Response'], 'PF_Corr',
                          results['gen_e'], results['Response'], 'Keras',
                          100, 0, 500,
                          "True [E]", "(Pred-True)/True [E]", "pdf/response_comparison.pdf")


### Handle EH and H seperately ###
plot.profile_plot_compare(compareData['gen_e'][compareData['type'] == 1], compareData['Response'][compareData['type'] == 1], 'PF_Corr EH Had',
                          results['gen_e'][results['type'] == 1], results['Response'][results['type']==1], 'PF_Corr EH Had',
                          100, 0, 500,
                          "True [E]", "(Pred-True)/True [E]", "pdf/eh_response_comparison.pdf")

plot.profile_plot_compare(compareData['gen_e'][compareData['type'] == 2], compareData['Response'][compareData['type'] == 2], 'PF_Corr H Had',
                          results['gen_e'][results['type'] == 2], results['Response'][results['type']==2], 'PF_Corr H Had',
                          100, 0, 500,
                          "True [E]", "(Pred-True)/True [E]", "pdf/h_response_comparison.pdf")

### Response vs Eta ###

plot.profile_plot_compare(abs(compareData['eta']), compareData['Response'], 'PF_Corr',
                          abs(results['eta']), results['Response'], 'Keras',
                          24, 0, 2.4,
                          "Eta", "(Pred-True)/True [E]", "pdf/response_vs_eta.pdf")


plot.EH_vs_E_plot(results['pf_ecalRaw']/results['gen_e'],results['pf_hcalRaw']/results['gen_e'],
                  results['pf_ecalRaw']/results['DNN'], results['pf_hcalRaw']/results['DNN'],
                  50, 'PF_Corr', 'Keras_Corr')

plot.E_bin_response(compareData,results,20, 500,['PF_Corr','Keras'],-1.2,1.2,"(Pred-True)/True (GeV)","pdf/1DResponse.pdf")    
plot.E_bin_response(compareData,results,4, 20,['PF_Corr','Keras'],-1.2,1.2,"(Pred-True)/True (GeV)","pdf/1DResponse.pdf")    


