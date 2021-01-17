# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
from timeit import default_timer as timer
import random
import math
from Fundamental_Calculation_Functions import weighted_reindeer_weariness_2
from Fundamental_Calculation_Functions import weighted_reindeer_weariness_3
from Fundamental_Calculation_Functions import calculate_distance
from Simulated_Annealing_1 import swap_gifts



### import the sampled datasets

gifts1= pd.read_csv('gifts1.csv')
gifts2= pd.read_csv('gifts2.csv')
gifts3= pd.read_csv('gifts3.csv')


### calculate the tabular matrices with the pairwise distances

distances_problem_1 = calculate_distance(gifts1)
distances_problem_2 = calculate_distance(gifts2)
distances_problem_3 = calculate_distance(gifts3)



### In this task I am going to combine 2 neighbourhood moves; swapping 2 gifts
### in a trip and the 3way suffix swap. In each iteration I will accept the 
### movement that results in lowest weariness and compare this weariness with the
### best weariness found so far



def Three_way_suff(gifts):           ### with this function I perform a 3way suffix swap
    
  overweight = True 
  while(overweight == True):
    check = True                     ### check trips' length
    while(check == True):     
      
       trips_to_cut = np.random.choice(np.unique(gifts[:,4]),size=3,replace = False)    ### random pick of the trips to be changed
       cutted_trip_1 = np.array(gifts[gifts[:,4]== trips_to_cut[0]])                    
       cutted_trip_2 = np.array(gifts[gifts[:,4]== trips_to_cut[1]])                    ### subsets of the initial dataset corresponding  
       cutted_trip_3 = np.array(gifts[gifts[:,4]== trips_to_cut[2]])                    ### to the trips that were randomly chosen 
       if ( len(cutted_trip_1[:,0])>1 and len(cutted_trip_2[:,0])>1 and len(cutted_trip_3[:,0])>1  ):
           check = False
 
    cut_1 = np.random.choice(range(1,len(cutted_trip_1[:,0])))                       ### random choice of the position where the cut 
    cut_2 = np.random.choice(range(1,len(cutted_trip_2[:,0])))                       ### will be performed - the first gift of the trip
    cut_3 = np.random.choice(range(1,len(cutted_trip_3[:,0])))                       ### is excluded
    
    flag1 = cutted_trip_1[0,0]                  
    flag2 = cutted_trip_2[0,0]                 ### store the ID of the first gift in the trip, so as not to lose track
    flag3 = cutted_trip_3[0,0]

    string1 = len(cutted_trip_1[:,0]) - cut_1      
    string2 = len(cutted_trip_2[:,0]) - cut_2 
    string3 = len(cutted_trip_3[:,0]) - cut_3

    new_trip_1 = np.delete(cutted_trip_1,np.s_[cut_1:cut_1+string1],axis=0)
    new_trip_2 = np.delete(cutted_trip_2,np.s_[cut_2:cut_2+string2],axis=0)    ### delete the gifts that were extracted
    new_trip_3 = np.delete(cutted_trip_3,np.s_[cut_3:cut_3+string3],axis=0)

    new_trip_1 = np.insert(new_trip_1,cut_1,cutted_trip_2[cut_2:,:],axis=0)
    new_trip_2 = np.insert(new_trip_2,cut_2,cutted_trip_3[cut_3:,:],axis=0)    ### insert the incoming gifts
    new_trip_3 = np.insert(new_trip_3,cut_3,cutted_trip_1[cut_1:,:],axis=0)

    new_trip_1[:,4]= trips_to_cut[0]                                           ### align the gifts' TripId in every trip
    new_trip_2[:,4]= trips_to_cut[1]
    new_trip_3[:,4]= trips_to_cut[2]
 
 
    if (np.sum( new_trip_1[:,3])<=1000.0 and np.sum( new_trip_2[:,3])<= 1000.0 and np.sum( new_trip_2[:,3])<=1000.0):
      overweight = False                       ### satisfying the weight limit constraint
     


  s1 = np.where(gifts[:,0]==flag1)           ### with the help of our 'flag' we track the position
  s1 = np.array(s1)                          ### of our trips inside the initial dataset    
  s1 = s1.item()     

  gifts = np.delete(gifts,np.s_[s1:s1+len(cutted_trip_1)],axis=0)       ### insert the new trip 
  gifts = np.insert(gifts,s1,new_trip_1,axis=0)                         ### after the string exchange



  s2 = np.where(gifts[:,0]==flag2)
  s2 = np.array(s2)
  s2 = s2.item()
  gifts = np.delete(gifts,np.s_[s2:s2+len(cutted_trip_2)],axis=0)    
  gifts = np.insert(gifts,s2,new_trip_2,axis=0)    


  s3 = np.where(gifts[:,0]==flag3)
  s3 = np.array(s3)
  s3 = s3.item()          
  gifts = np.delete(gifts,np.s_[s3:s3+len(cutted_trip_3)],axis=0)    
  gifts = np.insert(gifts,s3,new_trip_3,axis=0)     

    
  return gifts







                                             ### the output of this function is the best
def SimulatedAnnealing_2(gifts):             ### weariness found for 1000xN solution evaluations
    
    
    if (gifts.shape == gifts1.shape):        
        x = distances_problem_1              
    elif (gifts.shape == gifts2.shape):      
        x = distances_problem_2
    else:
        x = distances_problem_3
        
        
    gifts_alt = gifts.reindex(np.random.permutation(gifts.index))     

    wrw,gifts_perm = weighted_reindeer_weariness_2(gifts_alt)     


    store_best = wrw                            ### store the trully best weariness in case of uphill move when reaching 1000xN solutions
    wrw_best = wrw                              ### defining the initial solution as best
    gifts_best = gifts_perm     
    
    
    if (np.max(gifts_best[:,4])<3):                        ### make sure that there are at least 3 trips
       gifts_best[:len(gifts_best[:,4])//3,4] =1.0         ### so that we can apply 3way suffix swap
       gifts_best[len(gifts_best[:,4])//3:2*len(gifts_best[:,4])//3,4]=2.0
       gifts_best[2*len(gifts_best[:,4])//3:,4]=3.0    

    T = 1
    a = 0.95
    
    maximum_eval = 1000*gifts.shape[0]
    for i in range(maximum_eval):
        
        swapped_gifts = swap_gifts(gifts_best)                   ### perform movement 1
        wrw_1 =  weighted_reindeer_weariness_3(swapped_gifts,x)  ### functions defined in TASK 3
        
        threeway_gifts = Three_way_suff(gifts_best)              ### perform movement 2
        wrw_2 = weighted_reindeer_weariness_3(threeway_gifts,x) 

        if (wrw_2 <= wrw_1):
            wrw_check = wrw_2
            gifts_check = threeway_gifts
        else:                                                    ### we choose the best movement
            wrw_check = wrw_1
            gifts_check = swapped_gifts
            
        if (wrw_check < store_best):
            store_best = wrw_check
            wrw_best = wrw_check
            gifts_best = gifts_check
        
        if ( (wrw_check < wrw_best) or  (math.exp((wrw_best - wrw_check)/( T * wrw_best))> random.random()) ):
            wrw_best = wrw_check
            gifts_best = gifts_check
            
        if (T > 0.05) :
          T = a*T
    
    
    return store_best          




                                     ### with the help of this function we run
def SA_2_Random_Seed(gifts):         ### Simulated_Annealing_2 for 30 seeds 
    
    weariness_set = []
    time = []
    
    for i in range(30,33):
       np.random.seed(i)
       start_time = timer()
       x  = SimulatedAnnealing_2(gifts)
       stop_time = timer()
       weariness_set.append(x)
       time.append(round(stop_time - start_time,2))
    
    time = np.array(time)
    weariness_set = np.array(weariness_set)
   
    
    return weariness_set,time



### store results in a dataframe
SimulatedAnnealing_2_results = pd.DataFrame()

x1,x2 = SA_2_Random_Seed(gifts1)
SimulatedAnnealing_2_results.loc[:,'gifts1'] = x1
SimulatedAnnealing_2_results.loc[:,'gifts1_time'] = x2

y1,y2 = SA_2_Random_Seed(gifts2)
SimulatedAnnealing_2_results.loc[:,'gifts2'] = y1
SimulatedAnnealing_2_results.loc[:,'gifts2_time'] = y2
 
z1,z2 = SA_2_Random_Seed(gifts3)
SimulatedAnnealing_2_results.loc[:,'gifts3'] = z1
SimulatedAnnealing_2_results.loc[:,'gifts3_time'] = z2


SimulatedAnnealing_2_results.iloc[:,::2].min()   
SimulatedAnnealing_2_results.iloc[:,::2].max()   
SimulatedAnnealing_2_results.iloc[:,::2].mean()  
SimulatedAnnealing_2_results.iloc[:,::2].std()   
SimulatedAnnealing_2_results.iloc[:,1::2].mean()


SimulatedAnnealing_2_results.to_csv('SimulatedAnnealing_2_results.csv', encoding='utf-8')





