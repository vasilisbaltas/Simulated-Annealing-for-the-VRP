# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt



RandomSearch_results = pd.read_csv('RandomSearch_results.csv')
SimulatedAnnealing_results = pd.read_csv('SimulatedAnnealing_results.csv')
SimulatedAnnealing_2_results = pd.read_csv('SimulatedAnnealing_2_results.csv')




###  this function creates boxplots that compare the performance 
###  of each algorithm according to the problem size

def visualise_benchmarks(dataRS,dataSA,dataSA2):
    
    box_round=1              ### counting the number of boxes we want
    while box_round<4: 
     
        fig, ax = plt.subplots(figsize=(2, 3),dpi=80,facecolor='white')          ### setting the figure size
        plt.yscale("log")                                                        ### changing the scale to log to make it more compact
        ax.set_facecolor('white') 
        ax.set_xticklabels(['Random Search','Simulated Annealing','Simulated Annealing 2'])
        plt.ylabel('Weariness',size=20,color='black')
        plt.xlabel("",size=20,color='black') 
        mpl.pyplot.grid(b=True,which='major',color='black', linestyle='dotted', linewidth=2,alpha=0.3) 
        ax.spines['bottom'].set_color('black') 
        
        ### defining the font and size parameters
        
        mpl.rcParams['font.family'] = "serif"
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.labelsize'] = 10
        mpl.rcParams['axes.labelweight'] = 'bold'
        mpl.rcParams['xtick.labelsize'] = 20
        mpl.rcParams['ytick.labelsize'] = 20
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['figure.titlesize'] = 12
        colors = ['#c27e79',"#77a1b5","#6ba353"]    ### choosing colors

        if box_round==1:                            ###  creating box1
               
               ### setting boxplots layout
               
               box1=plt.boxplot([dataRS['gifts1'],dataSA['gifts1'],dataSA2['gifts1']],notch=False,   
               patch_artist=True,showmeans=True,vert =1,
               flierprops=dict(markerfacecolor='r', marker='D'),                   
               medianprops = dict(linestyle='-.', linewidth=2.5,color='yellow'))
               plt.title('Algorithm Benchmarking in 10 gifts',size=25) 
       
               for box, patch, color in zip(box1['boxes'],box1['boxes'], colors):
                   patch.set_facecolor(color)
                   box.set(hatch='/')
                   box.set(linewidth=2)
               box1=plt.figure(1)
       
        elif box_round==2:                         ### creating box 2
            
        
               box2=plt.boxplot([dataRS['gifts2'],dataSA['gifts2'],dataSA2['gifts2']],notch=False,  
               patch_artist=True,showmeans=True,vert =1,
               flierprops=dict(markerfacecolor='r', marker='D'),
               medianprops = dict(linestyle='-.', linewidth=2.5,color='yellow'))
               plt.title('Algorithm Benchmarking in 100 gifts',size=25)
        
               for box, patch, color in zip(box2['boxes'],box2['boxes'], colors):
                   patch.set_facecolor(color)
                   box.set(hatch='/')
                   box.set(linewidth=2)
               box2=plt.figure(2)
        else:
            
       
                box3=plt.boxplot([dataRS['gifts3'],dataSA['gifts3'],dataSA2['gifts3']],notch=False,   
                patch_artist=True,showmeans=True,vert =1,
                flierprops=dict(markerfacecolor='r', marker='D'),
                medianprops = dict(linestyle='-.', linewidth=2.5,color='yellow'))
                plt.title('Algorithm Benchmarking in 1000 gifts',size=25)  
            
                for box, patch, color in zip(box3['boxes'],box3['boxes'], colors):
                    patch.set_facecolor(color)
                    box.set(hatch='/')
                    box.set(linewidth=2)
                box3=plt.figure(3)
       
        box_round+=1       
      
    return   box1.show(),box2.show(),box3.show() 


#Now we create the boxplots with weariness values for our three algorithms
    
visualise_benchmarks(RandomSearch_results,SimulatedAnnealing_results,SimulatedAnnealing_2_results)

