#To install in terminal type:
#pip install brainrender
#pip install streamlit
#pip install pandas
#pip install affinewarp

#To run the code:
#Navigate to the folder of the a in the terminal
#

import streamlit as st
import numpy as np
from streamlit import caching
import pandas as pd

# Import variables
from brainrender import * # <- these can be changed to personalize the look of your renders

# Import scene class
from brainrender.scene import Scene


from affinewarp import PiecewiseWarping, SpikeData
import numpy as np
#from affinewarp.visualization import rasters
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
import os

# Set up VTKPLOTTER to work in Jupyter notebooks
from vtkplotter import *

all_data_path='/media/maria/DATA1/Documents/NeuroMatchAcademy2020_dat/unzipped_files'

caching.clear_cache()

@st.cache(persist=True)
def get_trial_data(all_data_path,sl_neuron):
    selected_recordings=['Richards_2017-10-31.tar']
    path=all_data_path+'/'+selected_recordings[0]
    neuron_inds=np.load(path+'/'+'spikes.clusters.npy')
    spk_tms=np.load(path+'/'+'spikes.times.npy')
    trials_orig=np.load(path+'/'+'trials.intervals.npy')
    print(trials_orig.shape)
    #Behavioral data
    #mot_timestamps=np.load(path+'/'+'face.timestamps.npy')
    #mot_energy=np.load(path+'/'+'face.motionEnergy.npy')

    #spk_ids=np.where(neuron_inds==618)
    #spk_tms_one_neuron=spk_tms[spk_ids]

    spikes=[]
    neurons=[]
    trials=[]
    neurons_orig=[sl_neuron]
    for neuron in neurons_orig:
        spk_ids=np.where(neuron_inds==neuron)
        spk_tms_one_neuron=spk_tms[spk_ids]
        for trial in range(0,260):
            trial_range= np.bitwise_and(spk_tms_one_neuron>=trials_orig[trial][0],spk_tms_one_neuron<=trials_orig[trial][1])
            if trial==0:
                #trial_range= np.bitwise_and(spk_tms_one_neuron>=trials[trial][0],spk_tms_one_neuron<=trials[trial][1])
                subset=spk_tms_one_neuron[trial_range]
            else:
                subset=spk_tms_one_neuron[trial_range]-trials_orig[trial-1][1]
            for spike in subset:
                trials.append(trial)
                spikes.append(spike)
                neurons.append(neuron)
        #Select spikes in the trial for the neuron that we care about
        #subset=spk_tms_one_neuron[trial_range]

        #print(subset)
        #print(trial_range)
    #print(neurons)
    data = SpikeData(trials, spikes, neurons, tmin=0, tmax=5.0)
    return data

sl_neuron=st.text_area('Choose neuron here:')
if sl_neuron:
    sl_neuron=int(sl_neuron)
else:
    sl_neuron=0
data= get_trial_data(all_data_path,sl_neuron)
scatter_kw = dict(s=2, c='k', lw=0, alpha=.8)
idx=data.neurons==int(sl_neuron)
y, x = data.trials[idx], data.spiketimes[idx]
        #print(x)
        #print(y,x)
plt.scatter(x, y,**scatter_kw)
plt.title('Spike trains from neuron '+str(int(sl_neuron)))
plt.xlabel('Time s')
plt.ylabel('Trials')
plt.xlim(0,5)
plt.show()

st.pyplot()

def get_ix_name(all_data_path,sl_neuron):
    selected_recordings=['Richards_2017-10-31.tar']
    path=all_data_path+'/'+selected_recordings[0]
    brain_df=pd.read_csv(path+'/'+'channels.brainLocation.tsv', sep='\t')
    brain_areas_=brain_df[brain_df.index == sl_neuron]['allen_ontology'].tolist()
    #return brain_df
    return brain_areas_,brain_df

#st.write(get_ix_name(all_data_path,sl_neuron))


@st.cache(persist=True)

def get_brain_areas(all_data_path):
    selected_recordings=['Richards_2017-10-31.tar']
    path=all_data_path+'/'+selected_recordings[0]
    brain_df=pd.read_csv(path+'/'+'channels.brainLocation.tsv', sep='\t')
    brain_areas_=np.unique(brain_df['allen_ontology'])
    brain_areas=[]
    for brain_ar in range(0,brain_areas_.shape[0]):

        brain_areas.append(str(brain_areas_[brain_ar]))
    return brain_areas

brain_areas=get_brain_areas(all_data_path)

#brain_option = st.selectbox('Which circuit to visualize?', brain_areas)
brain_option=get_ix_name(all_data_path,sl_neuron)[0][0]

st.write('You selected:', brain_option)

import brainrender
brainrender.SHADER_STYLE = 'cartoon'
from brainrender.scene import Scene

# Create a scene
scene = Scene()

scene.add_brain_regions([brain_option], alpha=.15)

#scene.render()
vp = Plotter(axes=0)
vp.show(scene.get_actors(), viewup=(10, 0.7, 0))

scene.close()

brain_df=get_ix_name(all_data_path,sl_neuron)[1]

st.write(brain_df)

#vp = Plotter(axes=0)
#vp.show(tutorial_scene.get_actors(), viewup=(10, 0.7, 0))
