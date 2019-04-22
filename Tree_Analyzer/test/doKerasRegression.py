from __future__ import division

import plot
import process_data
import train


import pandas as pd
import numpy as np
np.random.seed(0)
import math

### Allocate the GPU usage on lpc
train.AllocateVRam()

###################################################################
########### DEFINE inputs and target for training and the ROOT file
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
                                                  withCorr = True, isTrainProbe = True)

### prepare test and training data
train_data, test_data, train_labels, test_labels = process_data.PreProcess(dataset, targetVariables)


##############################################################
########## Build and Train the model
test_predictions = train.doMachineLearn(train_data, train_labels, test_data, test_labels)
print "Saving Model"
quit()
##############################################################
##########Recover meaningful predictions
test_predictions, test_labels = process_data.PostProcess(test_predictions, test_data, test_labels)

##############################################################
########## TRAINING ANALYSIS 
results = test_data.copy()
for variable in targetVariables:
    results[variable] = test_labels[variable]

results['DNN'] = test_predictions


compareData['Response'] = (compareData['pf_totalRaw'] - compareData['gen_e'])/compareData['gen_e']
results['Response'] = (results['DNN']-results['gen_e'])/results['gen_e']

##############################################
########## PLOT MAKING #######################


plot.plot_perf(results, None, "Pred vs True")

plot.plot_hist_compare([compareData['Response'],results['Response']],100, -1.2,1.2,['PF_Corr','Keras'],"(Pred-True)/True [E]","pdf/perf_comparison.pdf")
plot.plot_hist_compare([compareData['Response'],(results['DNN']-results['p'])/results['p']],100, -1.2,1.2,['PF_Corr','Keras'],"(Pred-p)/p","pdf/perf_comparison_p.pdf")
### compare pt distribution ###
plot.plot_hist_compare([compareData['pf_totalRaw'],results['DNN']],25,0,550,['PF_Corr','Keras'],"E","pdf/pt_comparison.pdf")

plot.E_reso_plot(compareData['pf_totalRaw']/compareData['gen_e'],results['DNN']/results['gen_e'],100,500,['PF_Corr','Keras'],0,500,'Ecorr/ETrue','width/mean','pdf/resulution.pdf')
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

