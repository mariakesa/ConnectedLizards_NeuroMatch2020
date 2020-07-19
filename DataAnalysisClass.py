
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import elephant.conversion as conv
import neo 
import quantities as pq
from elephant.statistics import instantaneous_rate
import neo
from quantities import Hz, s, ms

class DataAnalysis():
    def __init__(self,all_data_path, selected_recordings):
	'''
	Example class usage:
	all_data_path='/media/maria/DATA1/Documents/NeuroMatchAcademy2020_dat/unzipped_files'
	selected_recordings=['Richards_2017-10-31.tar']
	dat_an=DataAnalysis(all_data_path,selected_recordings)
	dat_an.plot_one_trial_one_neuron(0,2,611)
	'''
        '''
        Initialize class with path to all data and a list containing the name of the recordings
        to be analyzed. 
        '''
        self.all_data_path=all_data_path
        self.selected_recordings=selected_recordings
    
    def plot_one_trial_one_neuron(self,recordings_index,trial_index,neuron_index):
        '''
        
        
        '''
        path=self.all_data_path+'/'+self.selected_recordings[recordings_index]
        #Neural data
        neuron_inds=np.load(path+'/'+'spikes.clusters.npy')
        spk_tms=np.load(path+'/'+'spikes.times.npy')
        trials=np.load(path+'/'+'trials.intervals.npy')
        #Behavioral data
        mot_timestamps=np.load(path+'/'+'face.timestamps.npy')
        mot_energy=np.load(path+'/'+'face.motionEnergy.npy')
        
        spk_ids=np.where(neuron_inds==neuron_index)
        spk_tms_one_neuron=spk_tms[spk_ids]
        
        #Select spikes in the trial for the neuron that we care about
        spks_range = np.bitwise_and(spk_tms_one_neuron>=trials[trial_index][0],spk_tms_one_neuron<=trials[trial_index][1])
        subset=spk_tms_one_neuron[spks_range]
        
        #Create elephant SpikeTrain object
        spk_tr=neo.SpikeTrain(subset*pq.s,t_start=trials[trial_index][0],t_stop=trials[trial_index][1])
        #print(spk_tr)
        
        #Plot spike train
        plt.eventplot(spk_tr)
        plt.title('Spike train for one trial. trial '+str(trial_index)+' '+', neuron: '+str(neuron_index))
        plt.show()
        
        #Plot instantaneous firing rate
        r=instantaneous_rate(spk_tr,t_start=trials[trial_index][0]*pq.s,t_stop=trials[trial_index][1]*pq.s, sampling_period=3.3333333333333e-5*pq.s, kernel='auto', cutoff=5.0)
        plt.plot(r)
        plt.title('Instantaneous rate for one trial')
        plt.show()
        
        #Plot behavior motion energy
        beh_range= np.bitwise_and(mot_timestamps[:,1]>=trials[trial_index][0],mot_timestamps[:,1]<=trials[trial_index][1])
        #print(np.where(beh_range==True))
        #print(mot_timestamps[beh_range])
        beh_subset=mot_energy[beh_range]
        plt.plot(mot_timestamps[beh_range][:,1].flatten(),beh_subset)
        plt.title('Motion energy in trial')
        plt.show()
        
        
