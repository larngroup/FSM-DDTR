# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:14:06 2022

@author: Ana Catarina
"""

import pandas as pd 
import copy
import numpy as np
 


def verify_domination(pts,total_array):
    #TODO adapt to several elements
    flag_non_dominant=True
    #Simple iterationn
   
    for el in total_array:
        
        if pts[0]<el[0] and pts[1]<el[1]:
            flag_non_dominant=False
            break

    return flag_non_dominant




def nsga(pd_data, maxi_mol):
    print("NSGA")
   
    
   

    df_original=pd_data #pd.read_csv(path_data,index_col=None)
    df_v=df_original.copy()
    df_v['index_col_org'] = df_v.index
    new_columns=df_original.columns
    df_new=pd.DataFrame([],columns=df_original.columns)

    #best_molecules=pd.DataFrame([])
    #Define the first and second parameters
    pn1="pIC50"
    pn2="SAS"

    


    max_mol=maxi_mol
    rank_list=[]
    non_dominated=[]
    rank_n=1
 
    while df_new.shape[0]<max_mol:
        #Each cycle will add indexes to the list    

        data=df_v[[pn1,pn2]].to_numpy()

        for idx in range(data.shape[0]):
            if idx in non_dominated:
                continue
            data_point=data[idx]
            new_data = copy.deepcopy(data)
            
            new_data=np.delete(new_data,idx,axis=0)
            
            my_flag=verify_domination(data_point,new_data)
            if my_flag:
                non_dominated.append(idx)

        selected=df_v.iloc[non_dominated]
        df_v=df_v.drop(non_dominated)
        df_v=df_v.reset_index(drop=True)
     
        some_list=np.ones(len(non_dominated))*rank_n
        some_list=some_list.tolist()
        rank_list+=some_list
        rank_n+=1
        non_dominated=[]
        df_new=pd.concat([df_new,selected],ignore_index=True)

    #Plot the molecules
    df_new["Rank"]=rank_list
    data_p1=df_v[[pn1,pn2]]

    data_p2=df_new[[pn1,pn2,"Rank",'index_col_org']]
 
    poly_data = data_p2.sort_values(by=['Rank'], ascending=False)
    print("END")
    return poly_data




def verify_domination_(pts,total_array):
    #TODO adapt to several elements
    flag_non_dominant=True
    #Simple iterationn
   
    for el in total_array:
        
        if pts[0]<el[0] and pts[1]<el[1] and pts[2]<el[2]:
            flag_non_dominant=False
            break

    return flag_non_dominant




def nsga3(pd_data, maxi_mol):
    print("NSGA")
   
    
   

    df_original=pd_data #pd.read_csv(path_data,index_col=None)
    df_v=df_original.copy()
    df_v['index_col_org'] = df_v.index
    new_columns=df_original.columns
    df_new=pd.DataFrame([],columns=df_original.columns)

    #best_molecules=pd.DataFrame([])
    #Define the first and second parameters
    pn1="pIC50"
    pn2="SAS"
    pn3 = "Mw" #"Mw" , "TPSA" , "LogP

    


    max_mol=maxi_mol
    rank_list=[]
    non_dominated=[]
    rank_n=1
 
    while df_new.shape[0]<max_mol:
        #Each cycle will add indexes to the list    

        data=df_v[[pn1,pn2,pn3]].to_numpy()

        for idx in range(data.shape[0]):
            if idx in non_dominated:
                continue
            data_point=data[idx]
            new_data = copy.deepcopy(data)
            
            new_data=np.delete(new_data,idx,axis=0)
            
            my_flag=verify_domination_(data_point,new_data)
            if my_flag:
                non_dominated.append(idx)

        selected=df_v.iloc[non_dominated]
        df_v=df_v.drop(non_dominated)
        df_v=df_v.reset_index(drop=True)
     
        some_list=np.ones(len(non_dominated))*rank_n
        some_list=some_list.tolist()
        rank_list+=some_list
        rank_n+=1
        non_dominated=[]
        df_new=pd.concat([df_new,selected],ignore_index=True)

    #Plot the molecules
    df_new["Rank"]=rank_list
    data_p1=df_v[[pn1,pn2,pn3]]

    data_p2=df_new[[pn1,pn2,pn3,"Rank",'index_col_org']]
 
    poly_data = data_p2.sort_values(by=['Rank'], ascending=False)
    print("END...")
    return poly_data










def verify_domination_5(pts,total_array):
    #TODO adapt to several elements
    flag_non_dominant=True
    #Simple iterationn
   
    for el in total_array:
        
        if pts[0]<el[0] and pts[1]<el[1] and pts[2]<el[2] and pts[3]<el[3] and pts[4]<el[4]:
            flag_non_dominant=False
            break

    return flag_non_dominant




def nsga5(pd_data, maxi_mol):
    print("NSGA")
   
    
   

    df_original=pd_data #pd.read_csv(path_data,index_col=None)
    df_v=df_original.copy()
    df_v['index_col_org'] = df_v.index
    new_columns=df_original.columns
    df_new=pd.DataFrame([],columns=df_original.columns)

    #best_molecules=pd.DataFrame([])
    #Define the first and second parameters
    pn1="pIC50"
    pn2="SAS"
    pn3 = "Mw" #"Mw" , "TPSA" , "LogP
    pn4 = "TSPA"
    pn5 = "LogP"
   

    max_mol=maxi_mol
    rank_list=[]
    non_dominated=[]
    rank_n=1
 
    while df_new.shape[0]<max_mol:
        #Each cycle will add indexes to the list    

        data=df_v[[pn1,pn2,pn3,pn4,pn5]].to_numpy()

        for idx in range(data.shape[0]):
            if idx in non_dominated:
                continue
            data_point=data[idx]
            new_data = copy.deepcopy(data)
            
            new_data=np.delete(new_data,idx,axis=0)
            
            my_flag=verify_domination_5(data_point,new_data)
            if my_flag:
                non_dominated.append(idx)

        selected=df_v.iloc[non_dominated]
        df_v=df_v.drop(non_dominated)
        df_v=df_v.reset_index(drop=True)
     
        some_list=np.ones(len(non_dominated))*rank_n
        some_list=some_list.tolist()
        rank_list+=some_list
        rank_n+=1
        non_dominated=[]
        df_new=pd.concat([df_new,selected],ignore_index=True)

    #Plot the molecules
    df_new["Rank"]=rank_list
    data_p1=df_v[[pn1,pn2,pn3,pn4,pn5]]

    data_p2=df_new[[pn1,pn2,pn3,pn4,pn5,"Rank",'index_col_org']]
 
    poly_data = data_p2.sort_values(by=['Rank'], ascending=False)
    print("END...")
    return poly_data







