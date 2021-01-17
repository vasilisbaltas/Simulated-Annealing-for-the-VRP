# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
from timeit import default_timer as timer
import random
import math
from Fundamental_Calculation_Functions import weighted_reindeer_weariness_2
from Fundamental_Calculation_Functions import weighted_reindeer_weariness_3
from Fundamental_Calculation_Functions import calculate_distance


### import the sampled datasets

gifts1= pd.read_csv('gifts1.csv')
gifts2= pd.read_csv('gifts2.csv')
gifts3= pd.read_csv('gifts3.csv')


### calculate the tabular matrices with the pairwise distances

distances_problem_1 = calculate_distance(gifts1)
distances_problem_2 = calculate_distance(gifts2)
distances_problem_3 = calculate_distance(gifts3)



### now we are going to implement simulated annealing along with a neighborhood move
### that swaps  gifts in a trip



                                    ### this function performs our movement : swap two gifts in a trip
def swap_gifts(gifts):              ### the input argument of this function is our gifts dataset as a 2D np array
    
    flag = True
    while (flag == True):    
      swapped_trip = np.random.choice(np.unique(gifts[:,4]))    ### we chose the trip in which we swap gifts
      gifts_new = np.array(gifts[gifts[:,4]== swapped_trip])    ### the subset of the dataset corresponding to the chosen trip
      if (len(gifts_new[:,0])>1):
           flag = False      
    
    swapped_gifts =  np.random.choice(len(gifts_new[:,0]),size=2,replace = False)  ### choosing the gifts to be swapped
    
    ### swapping the sequene of delivery
    gifts_new[swapped_gifts[0].astype(int)-1,0],gifts_new[swapped_gifts[1].astype(int)-1,0] = gifts_new[swapped_gifts[1].astype(int)-1,0],gifts_new[swapped_gifts[0].astype(int)-1,0]
    ### swapping the corresponding weights
    gifts_new[swapped_gifts[0].astype(int)-1,3],gifts_new[swapped_gifts[1].astype(int)-1,3] = gifts_new[swapped_gifts[1].astype(int)-1,3],gifts_new[swapped_gifts[0].astype(int)-1,3]
    
    
    gifts[gifts[:,4]==swapped_trip,0] = gifts_new[:,0]       ### change the sequence of delivery in our initial dataset
    gifts[gifts[:,4]==swapped_trip,3] = gifts_new[:,3]       ### change the corresponding weights
    
    return gifts






def SA_swap_gifts(gifts):                     ### the output of this function is the best weariness
                                              ### found for 1000xN solution evaluations
    
    if (gifts.shape == gifts1.shape):        
        x = distances_problem_1              
    elif (gifts.shape == gifts2.shape):      
        x = distances_problem_2
    else:
        x = distances_problem_3
        
    gifts_alt = gifts.reindex(np.random.permutation(gifts.index))          ### firstly we produce a random sequence 
                                                                           ### of deliveries
    wrw,gifts_perm = weighted_reindeer_weariness_2(gifts_alt)
    
    store_best = wrw                            ### store the trully best weariness in case of uphill move when reaching 1000xN solutions
    wrw_best = wrw                              ### defining the initial solution as best
    gifts_best = gifts_perm                     ### and we also obtain a 2D numpy array
    
    T = 1
    a = 0.95     ### the chosen decay
    
    maximum_eval = 1000*gifts.shape[0]   
    for i in range(maximum_eval):   
    
       gifts_new = swap_gifts(gifts_best)                             ### we swap 2 gifts in a trip of this dataset
       wrw_to_check =  weighted_reindeer_weariness_3(gifts_new,x)     ### estimate the new weariness after the swap
    
       if (wrw_to_check < store_best):
          store_best = wrw_to_check             ### the trully optimal weariness at the moment
          wrw_best = wrw_to_check
          gifts_best = gifts_new
       if ( (wrw_to_check < wrw_best) or  (math.exp((wrw_best - wrw_to_check)/( T * wrw_best))> random.random()) ):
          wrw_best = wrw_to_check
          gifts_best = gifts_new
       
       if (T > 0.05) :
          T = a*T
          
     
    return store_best   



def SA_Random_Seed(gifts):           ### with this function we run Simulated Annealing
                                     ### for 30 different seeds
    weariness_set = []
    time = []
    
    for i in range(30,60):
       np.random.seed(i)
       start_time = timer()
       x  = SA_swap_gifts(gifts)
       stop_time = timer()
       weariness_set.append(x)
       time.append(round(stop_time - start_time,2))
    
    time = np.array(time)
    weariness_set = np.array(weariness_set)
   
    
    return weariness_set,time




### save results in a dataframe

SimulatedAnnealing_results = pd.DataFrame()

x1,x2 = SA_Random_Seed(gifts1)
SimulatedAnnealing_results.loc[:,'gifts1'] = x1
SimulatedAnnealing_results.loc[:,'gifts1_time'] = x2

y1,y2 = SA_Random_Seed(gifts2)
SimulatedAnnealing_results.loc[:,'gifts2'] = y1
SimulatedAnnealing_results.loc[:,'gifts2_time'] = y2
 
z1,z2 = SA_Random_Seed(gifts3)
SimulatedAnnealing_results.loc[:,'gifts3'] = z1
SimulatedAnnealing_results.loc[:,'gifts3_time'] = z2


SimulatedAnnealing_results.iloc[:,::2].min()   
SimulatedAnnealing_results.iloc[:,::2].max()   
SimulatedAnnealing_results.iloc[:,::2].mean()  
SimulatedAnnealing_results.iloc[:,::2].std()   
SimulatedAnnealing_results.iloc[:,1::2].mean()



SimulatedAnnealing_results.to_csv('SimulatedAnnealing_results.csv', encoding='utf-8')
