# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from timeit import default_timer as timer
import random
import math
from Fundamental_Calculation_Functions import weighted_reindeer_weariness


### import the sampled datasets

gifts1= pd.read_csv('gifts1.csv')
gifts2= pd.read_csv('gifts2.csv')
gifts3= pd.read_csv('gifts3.csv')



### implmentation of the Random Search algorithm

                                    ### the output of this function is the best weariness(objective function) found 
def Random_Search(gifts):           ### for 1000xN solution evaluations where N is the size of the problem
    
    ### at first we generate and evaluate one random solution
        
    y = weighted_reindeer_weariness(gifts)
    
    opt_total_weariness = y
    
    s_best =  opt_total_weariness
    
    maximum_eval = 1000*gifts.shape[0]   ### define the maximum number of evaluations
    for i in range(maximum_eval):        ### as 1000xN 
         x1 =  weighted_reindeer_weariness(gifts)
         if (x1 < s_best):
             s_best = x1
            
            
    return int(s_best)             
             



          
def RandomSeed_RandomSearch(gifts):    ### with this function we can run Random_Search
                                       ### for 30 different seeds
    weariness_set = []
    time = []
    
    
    for i in range(30,60):
       np.random.seed(i)
       start_time = timer()
       x  = Random_Search(gifts)
       stop_time = timer()
       weariness_set.append(x)
       time.append(round(stop_time - start_time,2))
     
    time = np.array(time)    
    weariness_set = np.array(weariness_set)
    

    return weariness_set,time


### we will save the results in a dataframe
RandomSearch_results = pd.DataFrame()

x1,x2 = RandomSeed_RandomSearch(gifts1)
RandomSearch_results.loc[:,'gifts1'] = x1
RandomSearch_results.loc[:,'gifts1_time'] = x2

y1,y2 = RandomSeed_RandomSearch(gifts2)
RandomSearch_results.loc[:,'gifts2'] = y1
RandomSearch_results.loc[:,'gifts2_time'] = y2
 
z1,z2 = RandomSeed_RandomSearch(gifts3)
RandomSearch_results.loc[:,'gifts3'] = z1
RandomSearch_results.loc[:,'gifts3_time'] = z2
 

RandomSearch_results.iloc[:,::2].min()   
RandomSearch_results.iloc[:,::2].max()   
RandomSearch_results.iloc[:,::2].mean()  
RandomSearch_results.iloc[:,::2].std()   
RandomSearch_results.iloc[:,1::2].mean()




RandomSearch_results.to_csv('RandomSearch_results.csv', encoding='utf-8')
