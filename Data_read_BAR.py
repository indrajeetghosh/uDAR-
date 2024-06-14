import pandas as pd
import numpy as np

def extract_data(persons, sensors):
    
    count1 = 0
            
    for person in persons:
        count = 0
        min_size = np.Inf
        data = None
        label = None
        for sensor in sensors:
            individual_data = np.load("../Badminton/dataset/Preprocessed Data/"+person+"_"+sensor+"_testing_X_data.npy")
            #print(individual_data.shape)
            if individual_data.shape[0] < min_size:
                min_size = individual_data.shape[0]
        
        for sensor in sensors:
            #print("dataset/processed/"+person+"_"+sensor+"_testing_X_data.npy")
            individual_data = np.load("../Badminton/dataset/Preprocessed Data/"+person+"_"+sensor+"_testing_X_data.npy")
            individual_data = individual_data[(individual_data.shape[0]-min_size):individual_data.shape[0],:,:]
            #print(individual_data.shape)
            individual_label = np.load("../Badminton/dataset/Preprocessed Data/"+person+"_"+sensor+"_testing_X_data.npy")
            individual_label = individual_label[(individual_label.shape[0]-min_size):individual_label.shape[0]]
            if count==0:
                data = individual_data
                label = individual_label
                label = np.reshape(label,(label.shape[0],1))
            else:
                data = np.append(data,individual_data,axis=1)
                #label = np.append(label,individual_label,axis=0)
            count+=1
            print(person)
            print(data.shape)
            
        if count1==0:
            data_full = data
            label_full = label
        else:
            data_full = np.append(data_full,data,axis=0)
            label_full = np.append(label_full,label)
        count1+=1
        label_full = np.reshape(label_full,(label_full.shape[0],1))
    return np.asarray(data_full), np.asarray(label_full)