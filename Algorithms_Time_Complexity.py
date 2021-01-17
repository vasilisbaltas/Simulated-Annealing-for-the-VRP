# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 19:37:47 2021

@author: Vasileios Baltas
"""


import numpy as np
import pandas as pd
from timeit import default_timer as timer
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



RandomSearch_results = pd.read_csv('RandomSearch_results.csv')
SimulatedAnnealing_results = pd.read_csv('SimulatedAnnealing_results.csv')
SimulatedAnnealing_2_results = pd.read_csv('SimulatedAnnealing_2_results.csv')



###  Now I am going to approximate the total number of solution evaluations
###  that the algorithms can generate in an hour for the initial dataset of
###  100k destinations 




def make_prediction(results):
    
    time1 = results['gifts1_time']
    time2 = results['gifts2_time']
    time3 = results['gifts3_time']
    
    times = np.append(time1,time2)      ### we create a column with the computational times required for every
    times = np.append(times,time3)      ### seed and for every dataset
    
    x=[]
    for i in range(30):
        x.append(10)
        
    for i in range(30):
        x.append(100)                   ### we create a column with 90 elements corresponding to the 
                                        ### problem sizes
    for i in range(30):
        x.append(1000)
        
    x = np.array(x).reshape(-1,1)
    
    
    z=x**2
    
    model=LinearRegression().fit(z, times)         ### we use the LinearRegression object from scikitlearn
    
    r_sq=model.score(z, times)                     ### the R squared of our model
    
    prediction=model.intercept_+model.coef_[0]*100000**2      ### the prediction of our equation for the demanded time
                                                              ### in order to run 100.000 * 1000 solution evaluations
    f=np.linspace(0, 100000,10)
    a=model.intercept_+model.coef_[0]*f**2                    
    
    fig,ax= plt.subplots()                           
    plt.plot(f,a)                                    
    plt.plot(100000,prediction,'ro')                          ### highlighting the prediction point
    plt.title("Regression Line",fontsize=15)
    plt.xlabel('Problem Size',fontsize=15)
    plt.ylabel('Time (sec) x 10^7',fontsize=15)
                                                              ### layout settings
    ax.tick_params(axis='both',which='major',labelsize=15)    
    plt.grid(b=True,which='minor',color='black', linestyle='-', linewidth=1,alpha=0.1)
    ax.ticklabel_format(axis='y',style='sci', scilimits=(3,4))

    ax.set_xscale('linear')
    plt.axvline(x=100000,linestyle='dotted',linewidth=1,ymin=0,ymax=0.95,color='red')

    plt.axhline(y=prediction,xmin=0,xmax=0.95,linestyle='dotted',color='red')



    plt.show()
    
    ### our function returns the predicted number of solution evaluations per hour, the R squared of the model, the slope and the intercept
    
    return print("Prediction:", 3600/(prediction/(100000*1000)),"\nR-squared:",r_sq,"\nCoefficient:",model.coef_[0],"\nIntercept:",model.intercept_)



make_prediction(RandomSearch_results)   
make_prediction(SimulatedAnnealing_results)      ### making predictions for our algorithms
make_prediction(SimulatedAnnealing_2_results)

