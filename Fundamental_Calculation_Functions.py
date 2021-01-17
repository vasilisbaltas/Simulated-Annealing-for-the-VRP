# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from timeit import default_timer as timer
from math import radians, cos, sin, asin, sqrt
import random
import math




### gifts.csv is the original dataset containing 100k destinations and I am going
### to random sample from it in order to create smaller ones

gifts = pd.read_csv('gifts.csv')

gifts1 = gifts.sample (n=10,random_state=1)      ### 10 - gifts dataset
gifts2 = gifts.sample (n=100,random_state=2)     ### 100 - gifts dataset
gifts3 = gifts.sample (n=1000,random_state=3)    ### 1000 - gifts dataset

### set the gifts' Ids from 1 to N (size of the problem) for every dataset

gifts1.index = pd.Series(np.arange(1,gifts1.shape[0]+1))
gifts2.index = pd.Series(np.arange(1,gifts2.shape[0]+1))
gifts3.index = pd.Series(np.arange(1,gifts3.shape[0]+1))

gifts1.iloc[:,0] = pd.Series(np.arange(0,gifts1.shape[0]+1))
gifts2.iloc[:,0] = pd.Series(np.arange(0,gifts2.shape[0]+1))
gifts3.iloc[:,0] = pd.Series(np.arange(0,gifts3.shape[0]+1))



gifts1.to_csv('gifts1.csv',encoding='utf-8')
gifts2.to_csv('gifts2.csv',encoding='utf-8')
gifts3.to_csv('gifts3.csv',encoding='utf-8')




### implement the haversine distance function that calculates the distance between
### two destinations given their longitude and latitude

def haversine(lat1, lon1, lat2, lon2):
    """ Calculate the great-circle distance between two points on the Earth surface.
    Takes 4 numbers, containing the latitude and longitude of each point in decimal degrees.

    The default returned unit is kilometers.
    """
    # mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
    avg_earth_radius = 6371.0 # 6371.0088

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))

    # calculate haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    d = sin(dlat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(dlon * 0.5) ** 2
    c = 2.0 * avg_earth_radius
    return c  * asin(sqrt(d))






####  create a function that gives us a tabular matrix, containing all the pairwise 
####  distances between our datasets' deliveries as well as the distances from
####  and to North Pole, meaning Matrix[0,1] is the distance from North Pole to
####  the first gift and Matrix[1,0] is again the distance between  the first 
####  gift and North Pole

def calculate_distance(gifts):
    
    
      lats = np.array(gifts.iloc[:,1])   ### column 1 contains the latitudes of our gifts
      longs = np.array(gifts.iloc[:,2])  ### column 2 contains the longitudes of our gifts
      
      lats = np.insert(lats,0,90)        ### we include the North Pole's longitude 
      longs = np.insert(longs,0,0)       ### and latitude in the distances matrix
      
                 
      trip_distances = np.zeros(shape=(len(lats),len(longs)))
      
  ### we calculate every pairwise distance using the haversine function
      
      for i in range(len(lats)):
           for j in range(len(longs)):
               trip_distances[i,j]=haversine(lats[i],longs[i],lats[j],longs[j])
        
      return trip_distances


### create the distance matrices for each of our problem instances

distances_problem_1 = calculate_distance(gifts1)
distances_problem_2 = calculate_distance(gifts2)
distances_problem_3 = calculate_distance(gifts3)




### objective function : this function computes the total weariness of all needed trips

def weighted_reindeer_weariness(gifts):      
    
  
    if (gifts.shape == gifts1.shape):        ### we define the distances matrix that
        x = distances_problem_1              ### we are going to use according to
    elif (gifts.shape == gifts2.shape):      ### the problem size
        x = distances_problem_2
    else:
        x = distances_problem_3
        
    
    gifts = gifts.reindex(np.random.permutation(gifts.index))     ### we perform a random permutation of the gifts
    gifts = gifts.values
   
    dist_total = 0.0
    weight = np.sum(gifts[:,3])             ### the total weight of gifts that need to
                                            ### be delivered
    
    while  (weight > 0.001):                ### by checking the total weight we ensure that all gifts are delivered
        extracted_gifts = None              ### python cannot recognise absolut 0 because of the many decimal points                     
        extracted_gifts = np.compress(np.cumsum(gifts[:,3])<=330.0,gifts,axis=0)
                                            ### we define the maximum weight limit for each trip as 330kg
                                            ### as well as a subset of the initial dataset that is going to be delivered
        sleigh_weight = 10                     
        dist = 0.0
        
        
        prev_weight = np.sum(extracted_gifts[:,3]) + sleigh_weight
    
        
        dist += x[0,extracted_gifts[0,0].astype(int)]*prev_weight 
        prev_weight -= extracted_gifts[0,3]
        
        for delivery in range(len(extracted_gifts[:,0])-1):
            
        
        
           dist +=  x[extracted_gifts[delivery,0].astype(int),extracted_gifts[delivery+1,0].astype(int)]* prev_weight
           prev_weight -= extracted_gifts[delivery+1,3]
        
    
        
        dist +=  x[extracted_gifts[-1,0].astype(int),0]* sleigh_weight

        gifts = np.delete(gifts,np.s_[:len(extracted_gifts)],0)   ### we delete the subset of the
        weight -= np.sum(extracted_gifts[:,3])                    ### catalogue that has been delivered
        
        dist_total += dist
     
        
    
    return dist_total







### this is another variant of the objective function according to the metaheuristic 
### that is going to be used

def weighted_reindeer_weariness_2(gifts):    ### we use a different function in order
                                             ### to estimate total weariness because now
                                             ### we need to obtain the initial sequence of deliveries
                                             ### as well as the total number of trips. We are going to
    if (gifts.shape == gifts1.shape):        ### use this function at the beggining in order to 
        x = distances_problem_1              ### to obtain our initial solution
    elif (gifts.shape == gifts2.shape):      
        x = distances_problem_2
    else:
        x = distances_problem_3
        
    counter_trips = np.array([1])                    ### counter of trips
    wght = np.array([0],dtype=float)                 ### sleigh's weight per trip
    trip = np.empty(shape=[0],dtype=int)             ### an array containing every gift's TripId !!!!!
   
    
    gifts = gifts.values                             
    for i in range(1,len(gifts)+1):
        
        if (wght + gifts[i-1,3]) <= 330.0:           ### we ensure that our weight limit is not exceeded
            trip = np.append(trip,counter_trips)     ### the fourth column is the weight column
            wght += gifts[i-1,3]                     ### the gift is loaded in the sleigh
            
        else:                                        ### the sleigh is full and we need to define a new trip
            
            wght = np.array([0],dtype=float)
            counter_trips += 1
            trip = np.append(trip,counter_trips)
            wght += gifts[i-1,3]   
    
    gifts = np.insert(gifts,4,trip,axis=1)           ### insert column with trip IDs !!!!!!
  
    
  
    distol = 0.0
    for t in range(1,np.max(gifts[:,4].astype(int))+1):              ### for the total of trips
        gifts_new = np.empty(shape=[np.count_nonzero(trip == t),5])  ### how many gifts in the t trip
        gifts_new = np.array(gifts[gifts[:,4]== t])                  ### gifts_new is a submatrix for every trip
        sleigh_weight = 10
        dist = 0.0
    
        prev_weight = np.sum(gifts_new[:,3]) + sleigh_weight
    
    
        dist += x[0,gifts_new[0,0].astype(int)]*prev_weight 
        prev_weight -= gifts_new[0,3]
    
        for delivery in range(len(gifts_new[:,0])-1):
    
        
           dist +=  x[gifts_new[delivery,0].astype(int),gifts_new[delivery+1,0].astype(int)]* prev_weight
           prev_weight -= gifts_new[delivery+1,3]
        
    
    
        dist +=  x[gifts_new[-1,0].astype(int),0]* sleigh_weight
        
        distol += dist
        
    return int(distol),gifts               ### returns total weariness as well as
                                           ### the dataset as 2D numpy array







                                                     ### we will use this function in order to obtain
                                                     ### our 1000xN solution evaluations
def  weighted_reindeer_weariness_3(gifts,x):         ### the input arguments of this function is the gifts dataset - 
                                                     ### after the swaps as a 2D np array as well as the distances matrix
                                                         
    trip = gifts[:,4].astype(int)                    ### the fifth column contains the TripIds     
    distol = 0.0
    
    for t in range(1,np.max(trip)+1):                                 
        gifts_new = np.empty(shape=[np.count_nonzero(trip == t),5])   
        gifts_new = np.array(gifts[gifts[:,4]== t])                   
        sleigh_weight = 10
        dist = 0.0
    
        prev_weight = np.sum(gifts_new[:,3]) + sleigh_weight
    
    
        dist += x[0,gifts_new[0,0].astype(int)]*prev_weight 
        prev_weight -= gifts_new[0,3]
    
        for delivery in range(len(gifts_new[:,0])-1):
    
        
           dist +=  x[gifts_new[delivery,0].astype(int),gifts_new[delivery+1,0].astype(int)]* prev_weight
           prev_weight -= gifts_new[delivery+1,3]
        
    
    
        dist +=  x[gifts_new[-1,0].astype(int),0]* sleigh_weight
        
        distol += dist
        
    return int(distol)

