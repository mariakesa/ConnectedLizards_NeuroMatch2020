
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


    def CCA_analysis(self,recordings_index,trial_index,brain_area):
        path=self.all_data_path+'/'+self.selected_recordings[recordings_index]

        #Prepare rates
        rates=self.convert_one_population_to_rates(recordings_index,trial_index,brain_area).T

        #Prepare behavior
        trials=np.load(path+'/'+'trials.intervals.npy')
        #Behavioral data
        mot_timestamps=np.load(path+'/'+'face.timestamps.npy')
        mot_energy=np.load(path+'/'+'face.motionEnergy.npy')

        beh_range= np.bitwise_and(mot_timestamps[:,1]>=trials[trial_index][0],mot_timestamps[:,1]<=trials[trial_index][1])
        #print(np.where(beh_range==True))
        #print(mot_timestamps[beh_range])
        beh_subset=mot_energy[beh_range]

        beh_subset_aligned=self.align_rate_and_behavior(beh_subset,rates[:,0]).reshape(-1,1)

        from sklearn.cross_decomposition import CCA

        cca = CCA(n_components=2)
        cca.fit(rates, beh_subset_aligned)
        X_train_r, Y_train_r = cca.transform(rates, beh_subset_aligned)
        print(X_train_r.shape)
        print(Y_train_r.shape)
        plt.scatter(X_train_r[:, 0], Y_train_r[:], label="train",
            marker="*", c="b", s=50)

        plt.show()

        plt.scatter(X_train_r[:, 1], Y_train_r[:], label="train",
            marker="*", c="b", s=50)

        #Can't do analysis on other trials because the number of timepoints is different!
        #E.g. the analysis below fails
        #rates_test=self.convert_one_population_to_rates(recordings_index,2,brain_area).T

        #X_test_r, Y_test_r = cca.transform(rates_test, beh_subset_aligned)
        #plt.scatter(X_test_r[:, 0], Y_test_r[:], label="test",
            #marker="^", c="b", s=50)

        #plt.show()

        print(beh_subset_aligned.shape)
        print(rates.shape)

    def twpca_model(self,recordings_index,trial_index,brain_area):
        from twpca import TWPCA
        path=self.all_data_path+'/'+self.selected_recordings[recordings_index]
        trials=np.load(path+'/'+'trials.intervals.npy')
        rates=self.convert_one_population_to_rates(recordings_index,trial_index,brain_area)
        rates=rates.T.reshape(1,rates.shape[1],rates.shape[0])#
        print(rates.shape)#.reshape(1,260,74)

        print(rates.shape)
        #print(rates)
        #print(rates.shape)
        from twpca.regularizers import curvature

        warp_penalty_strength = 0.05
        time_penalty_strength = 0.5

        # Add an L1 penalty on the second order finite difference of the warping functions
        # This encourages the warping functions to be piecewise linear.
        warp_regularizer = curvature(scale=warp_penalty_strength, power=1)
        # Adds an L2 penatly on the second order finite difference of the temporal factors.
        # Encourages the temporal factors to be smooth in time.
        time_regularizer = curvature(scale=time_penalty_strength, power=2, axis=0)
        n_components=10
        model = TWPCA(n_components,
              warp_regularizer=warp_regularizer,
              time_regularizer=time_regularizer,
              fit_trial_factors=False,
              #nonneg=True,
              warpinit='shift')
        # Fit model with gradient descent, starting with a learning rate of 1e-1 for 250 iterations,
        # and then a learning rate of 1e-2 for 500 iterations
        X_pred=model.fit_transform(rates, lr=(1e-1, 1e-2), niter=(250, 500))
        #print(help(TWPCA))
        #X_pred = model.predict(rates)
        print(X_pred.shape)

        plt.imshow(X_pred.reshape(rates.shape[2],rates.shape[1]))
        plt.show()
        plt.imshow(rates[0,:,:].T)

all_data_path='/media/maria/DATA1/Documents/NeuroMatchAcademy2020_dat/unzipped_files'
selected_recordings=['Richards_2017-10-31.tar']
dat_an=DataAnalysis(all_data_path,selected_recordings)
