
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import elephant.conversion as conv
import neo 
import quantities as pq
from elephant.statistics import instantaneous_rate,time_histogram
import neo
from elephant import kernels
from quantities import Hz, s, ms

def unzip_files(path_to_zip_files,path_to_unzip_files):
    '''
    Function for unzip all data files into a single directory.
    
    '''
    extension = ".tar"

    os.chdir(path_to_zip_files) # change directory from working dir to dir with files

    for item in os.listdir(path_to_zip_files): # loop through items in dir
        if item.endswith(extension): # check for ".zip" extension
            file_name = os.path.abspath(item) # get full path of files
            tar = tarfile.open(file_name, "r:")
            tar.extractall(path_to_unzip_files+'/'+str(item))
            tar.close()

class DataAnalysis():
    def __init__(self,all_data_path, selected_recordings):
        '''
        Initialize class with path to all data and a list containing the name of the recordings
        to be analyzed. 

        Example usage:

        all_data_path='/media/maria/DATA1/Documents/NeuroMatchAcademy2020_dat/unzipped_files'
        selected_recordings=['Richards_2017-10-31.tar']
        dat_an=DataAnalysis(all_data_path,selected_recordings)
        dat_an.plot_one_trial_one_neuron(0,0,611)
        '''
        self.all_data_path=all_data_path
        self.selected_recordings=selected_recordings
        self.mid_brain_circuits=['SCs','SCm','MRN','APN','PAG','ZI']
        self.frontal_circuits=['MOs','PL','ILA','ORB','MOp','SSp']
    
    def plot_one_trial_one_neuron(self,recordings_index,trial_index,neuron_index):
        '''
        Plots spikes, rates and behavior over a specified trial and neuron.
        
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
        spk_tr=neo.SpikeTrain(subset*pq.s,t_start=trials[trial_index][0]*pq.s,t_stop=trials[trial_index][1]*pq.s)
        #print(spk_tr)
        print((trials[trial_index][1]-trials[trial_index][0]))
        
        #Plot spike train
        plt.eventplot(spk_tr)
        plt.title('Spike train for one trial. trial '+str(trial_index)+' '+', neuron: '+str(neuron_index))
        plt.show()
        
        #Plot instantaneous firing rate
        kernel = kernels.GaussianKernel(sigma=0.1*pq.s, invert=True)
        #sampling_rate the same as behavior
        r=instantaneous_rate(spk_tr,t_start=trials[trial_index][0]*pq.s,t_stop=trials[trial_index][1]*pq.s, sampling_period=0.02524578*pq.s, kernel=kernel) #cutoff=5.0)
        plt.plot(r)
        plt.title('Instantaneous rate for one trial')
        plt.show()
        print('r shape',r.shape)
        
        #Plot behavior motion energy
        beh_range= np.bitwise_and(mot_timestamps[:,1]>=trials[trial_index][0],mot_timestamps[:,1]<=trials[trial_index][1])
        #print(np.where(beh_range==True))
        #print(mot_timestamps[beh_range])
        beh_subset=mot_energy[beh_range]
        plt.plot(mot_timestamps[beh_range][:,1].flatten(),beh_subset)
        plt.title('Motion energy in trial')
        plt.show()
        print('beh shp',beh_subset.shape)
        
        rate=np.array(r).flatten()
        beh_subset_aligned=self.align_rate_and_behavior(beh_subset,rate).flatten()
        print('Correlation coefficient between rate and behavior: '+str(np.corrcoef(beh_subset_aligned,rate)[0,1]))
        
    def align_rate_and_behavior(self,beh_subset,rate):
        rate_shp=rate.shape[0]
        beh_subset_aligned=beh_subset[:rate_shp]
        return beh_subset_aligned
    
    def extract_brain_region_neuron_indices(self,recordings_index,brain_area):
        neurons_df=pd.read_csv(self.all_data_path+'/'+self.selected_recordings[recordings_index]+'/'+'channels.brainLocation.tsv', sep='\t')
        subset=neurons_df[neurons_df['allen_ontology']==brain_area]
        dat=np.array(subset.index).flatten()
        return dat

    def get_spikes_of_one_population(self,recordings_index,brain_area):
        path=self.all_data_path+'/'+self.selected_recordings[recordings_index]
        neurons=self.extract_brain_region_neuron_indices(recordings_index,brain_area)
        neuron_inds=np.load(path+'/'+'spikes.clusters.npy')
        spk_tms=np.load(path+'/'+'spikes.times.npy')
        
        spike_times_lst=[]
        
        for neuron in neurons:
            spk_ids=np.where(neuron_inds==neuron)
            spk_tms_one_neuron=spk_tms[spk_ids]
            spike_times_lst.append(spk_tms_one_neuron)
            
            
        #print(spike_times_lst)
        
        return spike_times_lst
            
    def convert_one_population_to_rates(self,recordings_index,trial_index,brain_area):
        path=self.all_data_path+'/'+self.selected_recordings[recordings_index]
        trials=np.load(path+'/'+'trials.intervals.npy')
        spike_times_lst=self.get_spikes_of_one_population(recordings_index,brain_area)
        
        rates_lst=[]
        for spk_tms_one_neuron in spike_times_lst:
            spks_range = np.bitwise_and(spk_tms_one_neuron>=trials[trial_index][0],spk_tms_one_neuron<=trials[trial_index][1])
            subset=spk_tms_one_neuron[spks_range]

            #Create elephant SpikeTrain object
            spk_tr=neo.SpikeTrain(subset*pq.s,t_start=trials[trial_index][0]*pq.s,t_stop=trials[trial_index][1]*pq.s)
            #plt.eventplot(spk_tr)
            #plt.show()
            kernel = kernels.GaussianKernel(sigma=0.1*pq.s, invert=True)
            #sampling_rate the same as behavior
            r=instantaneous_rate(spk_tr,t_start=trials[trial_index][0]*pq.s,t_stop=trials[trial_index][1]*pq.s, sampling_period=0.02524578*pq.s, kernel=kernel) #cutoff=5.0)
            rates_lst.append(r.flatten())
    
        rates_lst=np.array(rates_lst)
        print(rates_lst.shape)
        return rates_lst
        
        
        
        
