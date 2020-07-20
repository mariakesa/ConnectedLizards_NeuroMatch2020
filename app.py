import streamlit as st
import numpy as np
from streamlit import caching
import pandas as pd

# Import variables
from brainrender import * # <- these can be changed to personalize the look of your renders

# Import scene class
from brainrender.scene import Scene

import sys
sys.path.append('../')
import os

# Set up VTKPLOTTER to work in Jupyter notebooks
from vtkplotter import *

all_data_path='/media/maria/DATA1/Documents/NeuroMatchAcademy2020_dat/unzipped_files'

caching.clear_cache()

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

brain_option = st.selectbox('Which circuit to visualize?', brain_areas)

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

#vp = Plotter(axes=0)
#vp.show(tutorial_scene.get_actors(), viewup=(10, 0.7, 0))
