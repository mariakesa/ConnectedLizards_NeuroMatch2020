import numpy as np
from sklearn.linear_model import Ridge
import numpy as np
import os , csv
from os import listdir
import matplotlib.pyplot as plt
import pandas as pd


#Sklearn imports
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import seaborn as sns

from scipy.linalg import null_space

from sklearn.model_selection import train_test_split

def ridge_regression(train_inputs,train_outputs,test_inputs,test_outputs,regularization_parameter=0.01):

    from_front_to_mid_reg=Ridge(regularization_parameter)

    from_front_to_mid_reg.fit(train_inputs,train_outputs)
    prediction=from_front_to_mid_reg.predict(test_inputs)

    pcs_corr=[]
    for j in range(0,test_outputs.shape[1]):
        #print(np.corrcoef(prediction[:,j],test_outputs[:,j])[0,1])
        pcs_corr.append(np.corrcoef(prediction[:,j],test_outputs[:,j])[0,1])

    plt.hist(pcs_corr)
    plt.show()

    ind_to_plot=np.argsort(pcs_corr)[-1]
    #ind_to_plot=2
    plt.plot(prediction[:,ind_to_plot],label='prediction')
    plt.plot(test_outputs[:,ind_to_plot],label='data')
    plt.title('correlation coef: '+str(pcs_corr[ind_to_plot]))
    plt.legend()

    return from_front_to_mid_reg

def ridge_behavior(from_front_to_mid_reg,pca_frontal,pcs_front,front_test,pcs_front_test,pcs_beh,pcs_beh_test,pcs_mid_test,dimensions,alpha=0.1):
    ridge_beh=Ridge(alpha)
    ridge_beh.fit((from_front_to_mid_reg.coef_[dimensions[0],:]@(pcs_front.T)).T,pcs_beh)
    pred_beh=ridge_beh.coef_@from_front_to_mid_reg.coef_[dimensions[0],:]@(pcs_front_test.T)
    actual=pcs_beh_test.T
    for j in range(0,dimensions[0].shape[0]):
        print(j)
        print('corr coef, dim=' + str(dimensions[0][j])+':', np.corrcoef(pred_beh[j,:],actual[j,:])[0,1])
        plt.plot(pred_beh[j,:],label='prediction')
        plt.plot(actual[j,:],label='original')
        plt.legend()
        plt.show()

def variance_explained(dat,prediction):
    residuals=(dat-prediction)**2
    natural_variance=(dat-np.mean(dat))**2
    return 1-residuals.sum()/natural_variance.sum()

def ridge_nullspace(projection_onto_nullspace_train,projection_onto_nullspace_test,pcs_beh,pcs_beh_test,alpha=0.01):
    ridge_null=Ridge(alpha)
    ridge_null.fit(projection_onto_nullspace_train.T,pcs_beh)
    pred_beh=ridge_null.coef_@projection_onto_nullspace_test
    actual=pcs_beh_test.T
    for j in range(0,actual.shape[0]):
        print('corr coef, dim=' + str(j)+':', np.corrcoef(pred_beh[j,:],actual[j,:])[0,1])
        plt.plot(pred_beh[j,:],label='prediction')
        plt.plot(actual[j,:],label='original')
        plt.legend()
        plt.show()
