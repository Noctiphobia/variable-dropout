import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_variable_dropout(*args : pd.Series, maxvars = 10, figsize = (8, 10)):
           
    
    indexes = args[0]['variable']   
    

    index_dictionary = {}

    for ind in indexes:
        tmp = [(x[x['variable'] == ind]['dropout_loss']).values[0] for x in args]
        index_dictionary[ind] = (np.mean(tmp), np.max(tmp))
        

        
    sorted_indexes = sorted(index_dictionary.items(), key = lambda x: x[1], reverse = True) 
    
    if(maxvars != None):
        sorted_indexes = sorted_indexes[0:maxvars]
        
    maxx = np.max([x[1][1] for x in sorted_indexes])
    
    maxx *= 1.05
    
    selected_indexes = [x[0] for x in sorted_indexes]
    
    plots_number = len(selected_indexes)
    
    counter = 0  
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle("drop-out loss", y=1.02, fontsize=12)
    
    for arg in args:
            
        counter +=1       
            
        tmp = plt.subplot(plots_number, 1, counter)

        tmp.set_xlim([0, maxx])
        
        if(not '_full_model_' in selected_indexes):
            selected_indexes += ['_full_model_']
        
        labels = selected_indexes 
        
        y_pos = np.arange(len(labels)) 
        
        values = arg[arg['variable'].isin(labels)]['dropout_loss']
        
        minx = np.min(values)
        
        n = len(labels)
        
        xerr = []
        
        for v in values:
            xerr.append((0, v - minx))
            
        xerr = np.array(xerr).T  
        
        plt.barh(y_pos, [minx] * n, color = 'white', capsize = 10, xerr = xerr, ecolor='black')
        
        tmp.set_yticks(y_pos)
        tmp.set_yticklabels(labels)
        
        title = arg['label'][0]        
        
        tmp.set_title(title)

        plt.gca().invert_yaxis()   

            
    plt.tight_layout()

    plt.show()
