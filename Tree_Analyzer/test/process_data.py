import uproot
import pandas as pd
import numpy as np 
np.random.seed(0)
import math 

def Get_tree_data(inputFiles,inputVariables, targetVariables,
                  withTracks,withDepth,endcapOnly,barrelOnly,
                  withCorr, isTrainProbe):

#'charge']#,'pf_hoRaw']
    if(withTracks):
        inputVariables += ["p"]#,"pt"]
    if(withDepth):
        inputVariables += ['pf_hcalFrac1', 'pf_hcalFrac2', 'pf_hcalFrac3', 
                  'pf_hcalFrac4', 'pf_hcalFrac5', 'pf_hcalFrac6', 'pf_hcalFrac7']
    if(withCorr):
        inputVariables += ['pf_total','pf_ecal','pf_hcal']

    def TChain(inputFiles):
        data = pd.DataFrame()
        for inputFile in inputFiles:
            print "Opening file: "+str(inputFile)
            inputTree = uproot.open("root/"+inputFile)["t1"]
            variables_ = []
            variables_ = list(variable for variable in (inputVariables+targetVariables) if variable in inputTree.keys())
            data = pd.concat([data,inputTree.pandas.df(variables_)], ignore_index=True) 
        return data

    dataset = TChain(inputFiles)
    dataset = dataset.dropna()
    
    my_cut = (abs(dataset['eta'])< 2.4) & (dataset['pf_totalRaw'] >0) & (dataset['gen_e']>0) & (dataset['gen_e']<500)
    #train_cut     = (dataset['pf_totalRaw']-dataset['gen_e'])/dataset['gen_e'] > -0.90 ## dont train on bad data with response of -1 
    dataset = dataset[(my_cut)]# & (train_cut)]
    if (endcapOnly):
        endcap_train = (abs(dataset['eta'])< 2.4) & (abs(dataset['eta'])>1.6)
        dataset = dataset[endcap_train]
    if (barrelOnly):
        barrel_train = (abs(dataset['eta'])<1.5)
        dataset = dataset[barrel_train]
    if (withTracks):
        p_cut = (dataset['p']>0.0) #& (dataset['p']<500.0)# & (dataset['pt']<500.0)
        dataset = dataset[p_cut]
    dataset = dataset.reset_index()
    
    dataset['type'] = np.nan
    dataset['type'][(dataset['pf_ecalRaw'] > 0) & (dataset['pf_hcalRaw'] == 0)] = 0 ## E  Hadron
    dataset['type'][(dataset['pf_ecalRaw'] > 0) & (dataset['pf_hcalRaw'] > 0)] = 1  ## EH Hadron
    dataset['type'][(dataset['pf_ecalRaw'] == 0) & (dataset['pf_hcalRaw'] > 0)] = 2 ## H  Hadron

    if (isTrainProbe):
        dataset = dataset.sample(frac=.25, random_state=1)
    
    compareData = dataset.copy()
    
    if(withCorr):
        compareData['pf_totalRaw'] = dataset['pf_total']
        compareData['pf_hcalRaw'] = dataset['pf_hcal']
        compareData['pf_ecalRaw'] = dataset['pf_ecal']
        del dataset['pf_total']
        del dataset['pf_hcal']
        del dataset['pf_ecal']
    
    return dataset, compareData

def PreProcess(dataset, targetVariables):

    train_dataset = dataset.sample(frac=0.75,random_state=1)
    test_dataset  = dataset.drop(train_dataset.index)
    test_dataset  = test_dataset[test_dataset['gen_e']>=0]
    
    train_labels = pd.DataFrame()
    test_labels = pd.DataFrame()
    for key in dataset.keys():
        if str(key) in targetVariables: 
            train_labels[key] = train_dataset[key].copy()
            del train_dataset[key]
            test_labels[key] = test_dataset[key].copy()
            del test_dataset[key]
            
    train_labels['gen_e'] = (train_dataset['pf_totalRaw']/train_labels['gen_e'])-1
#test_labels = np.log(test_dataset['pf_totalRaw']/test_labels)#############
    test_labels['gen_e'] = (test_dataset['pf_totalRaw']/test_labels['gen_e'])-1######################

    return train_dataset, test_dataset, train_labels, test_labels

def PostProcess(test_predictions, test_data, test_labels):
    #test_predictions = 1/(np.exp(test_predictions))*test_dataset['pf_totalRaw']###############
    test_predictions = 1/((test_predictions+1)/test_data['pf_totalRaw'])#######################
#test_predictions = np.exp(test_predictions)
#test_predictions = test_predictions*test_max
    
#test_labels = 1/(np.exp(test_labels))*test_dataset['pf_totalRaw']#########################
    test_labels['gen_e'] = 1/((test_labels['gen_e']+1)/test_data['pf_totalRaw'])#################################
#test_labels = np.exp(test_labels)
#test_labels = test_labels*test_max

    return test_predictions, test_labels
    
