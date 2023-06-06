# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 14:06:38 2022

@author: Ana Catarina
"""


import pandas as pd
import json
from transformers import Masked_Smiles_Model
import tensorflow as tf
import tensorflow_addons as tfa
from prepare_data_classification import *
import numpy as np
import itertools
import glob

def save_func(file_path,values):
    file=[i.rstrip().split(',') for i in open(file_path).readlines()]
    file.append(values)
    file=pd.DataFrame(file)
    file.to_csv(file_path,header=None,index=None)

def find_clusters(data_path):
    clusters = []
    for i in data_path:
        if 'test' in i:
            clusters.append(('test', pd.read_csv(i,header=None)))
        else:
            clusters.append(('train', pd.read_csv(i,header=None)))
    return clusters
    
def q2(y_true,y_pred):
    ss_res = tf.reduce_sum((y_true-y_pred)**2)
    y_mean = tf.reduce_mean(y_true)
    ss_tot = tf.reduce_sum((y_true-y_mean)**2)
    r2 = 1.0-ss_res/ss_tot
    #print(r2)
    return r2
    

def ccc(y_true,y_pred):
    ''' Concordance Correlation Coefficient'''

    #print(f'y {y_true} ,{y_pred}')
    x = tf.cast(y_true, dtype=tf.float32)
    y = tf.cast(y_pred, dtype=tf.float32)
    
    #print(f'y {x} ,{y}')
    x_mean = tf.reduce_mean(x)
    y_mean = tf.reduce_mean(y)
    #print(f'mean {x_mean} ,{y_mean}')
    x_var =  tf.math.reduce_variance(x)
    y_var =  tf.math.reduce_variance(y)
    #print(f'var {x_var} ,{y_var}')
    error_x = x-x_mean
    error_y = y-y_mean
    
    #print(f'Error {error_x} ,{error_y}')
    covar = tf.reduce_sum(error_x*error_y)/tf.cast(len(x), dtype=tf.float32)
    #print(covar)
    rhoc = 2*covar/(x_var + y_var + (x_mean - y_mean)**2 )
    #print(rhoc)
    #sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    #rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

def inference_metrics(model,data):
    pred_values = model.predict([data[0]])
    mse = tf.keras.losses.MeanSquaredError()
    metrics = {'MSE': mse(data[1],pred_values),'R2': q2(data[1],pred_values),
               'CCC':ccc(data[1],pred_values)}
    
    return metrics

def listStr_to_listInt(df_column):
    list_int = []
    for row in range(len(df_column)):           
        x = df_column[row]
        
        x = x.replace(",","")
        x = x.replace("[","")
        x = x.replace("]","")
        x = x.split()
        list_int.append([int(x[i]) for i in range(len(x))])
    return list_int


    
class Classification_SMILES_Model(tf.keras.Model):
    def __init__(self, smiles_len, smiles_vocab_size, dict_mol, rate, d_model, dff, num_heads, num_encoders, optimizer_fn, loss_fn, metrics_list, checkpoint_path, cls_layers_units, cls_act_func, **kwargs):
        super().__init__()
        
        self.smiles_input = tf.keras.Input(
            shape=(smiles_len,), dtype=tf.int32, name='smiles_input') 
        

        #self.model = Masked_Smiles_Model(d_model, dff, num_heads, num_encoders, smiles_len, smiles_vocab_size, dict_mol, rate)
        #self.model.load_weights('../model/pretrain_BERT/model')#('../checkpoints/weights/model_cls')
        #self.smiles_bert = self.model.layers[0]
        """self.model_trained = tf.keras.models.load_model('../model/pretrain_BERT/model_pretrained_8heads_4encoders.h5py')#('../model/pretrain_BERT/model_pretrained.h5py')
        self.model_trained.summary()
        self.smiles_bert = self.model_trained.layers[0] 
        """
        self.model_trained = tf.keras.models.load_model('../model/pretrain_BERT/model_4heads_4encoders_batch_size256_RAdam.h5py')#('../model/pretrain_BERT/model_pretrained.h5py')
        self.model_trained.summary()
	
        #self.global_pool = tf.keras.layers.GlobalMaxPool1D(name ='global_max')
        self.ffnn_layer = [tf.keras.layers.Dense(units=cls_layers_units[i],activation=cls_act_func[i], name=f'Classification_Layer_{i}') for i in range(len(cls_layers_units))]
        self.dropout_layer = [tf.keras.layers.Dropout(0.1, name=f'Dropout_Layer_{i}') for i in range(len(cls_layers_units))]
        
        self.final_layer = tf.keras.layers.Dense(units=1,activation='linear', name='Final_Classification_Layer')
                                         
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        
        self.checkpoint_path = checkpoint_path
        #self.ckpt = tf.train.Checkpoint(model=self.smiles_bert,
         #                  optimizer=self.optimizer_fn)

        #self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=1)


        #if self.ckpt_manager.latest_checkpoint:
        #    self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
         #   print('Latest checkpoint restored!!')
        
        self.metrics_list = metrics_list
        self.model = self.build_model()
              
        
    def build_model(self):
        
        # ckpt = tf.train.Checkpoint(model=self.smiles_bert,
        #                    optimizer=self.optimizer_fn)

        # ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=1)


        # if ckpt_manager.latest_checkpoint:
        #     ckpt.restore(ckpt_manager.latest_checkpoint)
        #     print('Latest checkpoint restored!!')
            
           
        smiles_in = self.smiles_input 
        smiles_out,_,_,all_layers = self.model_trained(smiles_in)  
        
        #all_layers = tf.convert_to_tensor(all_layers)
        #cls_layers = tf.gather(all_layers, 0, axis=2)
        #cls_layers = tf.transpose(cls_layers, perm=[1,0,2]) 
        #smiles_cls_out = self.global_pool(cls_layers)
        #print(cls_layers.shape)
        ##cls_layers = tf.transpose(cls_layers, perm=[1,2,0]) #
        #print(cls_layers.shape)
        #print(tf.shape(smiles_out_)[0])
        ##smiles_cls_out = tf.reshape(cls_layers, [tf.shape(smiles_out)[0],smiles_out.shape[2]*4])
        #print(smiles_cls_out.shape)   
        #####	smiles_out,_ = self.smiles_bert(smiles_in)
        #smiles_out = self.m.layers[0](smiles_in)#self.smiles_bert(smiles_in)
        #print(smiles_out)
        #self.model.summary()
        smiles_cls_out = tf.gather(smiles_out, 0, axis=1)
        #print(inp.shape)
        for i in range(len(self.ffnn_layer)):
           smiles_cls_out = self.dropout_layer[i](self.ffnn_layer[i](smiles_cls_out))
          
        out = self.final_layer(smiles_cls_out)
        
        losses = self.loss_fn
        list_metrics= self.metrics_list
        
    
        
        #smiles_bert_fcnn_model = CustomModel([smiles_in],[out]) 
        
        
        #cnn_fcnn_model = tf.keras.Model(inputs=[prot_in,smiles_in,fg_input],outputs=[out,fg], name='CNN_FCNN_Model')
        smiles_bert_fcnn_model = tf.keras.Model(inputs=[smiles_in],outputs=[out], name='SMILES_BERT_Model')
         
        #smiles_model.compile_(self.optimizer_fn,losses)#(optimizer=self.optimizer_fn,loss=losses, metrics=list_metrics, run_eagerly=True)
        smiles_bert_fcnn_model.compile(optimizer=self.optimizer_fn,loss=losses, metrics=list_metrics)
        smiles_bert_fcnn_model.summary()
        #smiles_model = model
        return smiles_bert_fcnn_model
    
    
    def fit_model(self,dataset, batch, epochs, callback_list = None, val_option = False, val_dataset = None):
        #model = self.build_model()
        smiles_inp, target = dataset
        #print(f'protein: {protein_data.shape}, smiles {smiles_data.shape}') protein: (19456, 1400), smiles (19456, 72)
        callback_list = callback_list
        #print(f'target: {target.shape}')
    
        if not val_option:
            #self.model.fit(x=[protein_data,smiles_data,num_fg], y=[kd_values, num_fg],
            self.model.fit(x=smiles_inp, y=target,
                      batch_size = batch, epochs = epochs,
                      verbose = 2, callbacks = callback_list)
            
        else:
            smiles_val_inp, target_val_y = val_dataset
            #self.model.fit(x=[protein_data,smiles_data,num_fg], y=[kd_values, num_fg],         
            self.model.fit(x=smiles_inp, y=target,         
                      batch_size = batch, epochs = epochs,
                      verbose = 2, callbacks = callback_list, 
                      validation_data=([smiles_val_inp], [target_val_y])) 
       
        return self.model

def grid_search(parameters,data,k_folds,folder,save_file):
    metrics_results=[[],[],[]]
    
    for num_run in range(len(k_folds)):
        print("-------------------//--------------------")
        print("Run: "+str(num_run))
        print("-------------------//--------------------")
        index_train=list(itertools.chain.from_iterable([k_folds[i] for i in range(len(k_folds)) if i!=num_run]))

        index_val=k_folds[num_run]

        
        data_train = [tf.gather(i,index_train) for i in data]
  
        data_val = [tf.gather(i,index_val) for i in data]
        
        es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                         min_delta = 0.0001, patience = 20, mode = 'min', 
                                         restore_best_weights=True)
        
        cls_units= parameters['Dim_cls_layers'].copy()
        cls_units.append(1)
        
        file_name = ['model_'+ str(parameters['D_model']) +'_' + str(parameters['Dff']) +'_'+
                     str(parameters['Num_heads'])+'_'+str(parameters['Num_encoder'])+ '_' +
                     str(parameters['Dropout'])+'_'+str(parameters['cls_act_func'])+ '_'+'linear'+ '_'+str(cls_units)+'_'+str(num_run)]


        mc = tf.keras.callbacks.ModelCheckpoint(filepath = folder+file_name[0], monitor = 'loss',
                                           save_best_only=True, save_weights_only = False, mode = 'min')
        
        callbacks = [es,mc]  
        
        optimizer = tfa.optimizers.RectifiedAdam(lr=5e-5)
        pred_model = Classification_SMILES_Model(74,len(smiles_dictionary), smiles_dictionary, parameters['Dropout'], 
                                              parameters['D_model'], parameters['Dff'], parameters['Num_heads'], parameters['Num_encoder'],#'adam',#RAdamOptimizer(),
                                              optimizer, 'mse', parameters['list_metrics'], parameters['checkpoint_path'],parameters['Dim_cls_layers'], parameters['cls_act_func']).fit_model(data_train,
                                        parameters['batch_size'],200,callbacks,True,data_val)
                                                                                    
        #mse,mse1, rmse,ci=cnn_fcnn_model.evaluate([data_val[0],data_val[1]],data_val[2],data_val[3])
        #cnn_fcnn_model = tf.keras.models.load_model('../gs_results/model_3_[64, 64, 128]_128_512_4_1_3_[1024, 512, 1024]_[0.5, 0.1]_0.1_0', #'../model/cnn_model_pad_same',
                                            #custom_objects={'c_index':c_index})                                  
        mse, q2, ccc = pred_model.evaluate([data_val[0]],[data_val[1]])
        
        metrics_results[0].append(mse)
        metrics_results[1].append(q2)
        metrics_results[2].append(ccc)
        
        
        result_values = list(np.hstack([parameters['D_model'],parameters['Dff'],
                     parameters['Num_heads'],parameters['Num_encoder'],parameters['Dropout'],cls_units, mse,q2,ccc,num_run]))
        

        
        save_func(save_file,result_values)
        tf.keras.backend.clear_session()
        
    result_values = list(np.hstack([parameters['D_model'],parameters['Dff'],
                     parameters['Num_heads'],parameters['Num_encoder'],parameters['Dropout'],cls_units,
                     np.mean(metrics_results[0]),np.mean(metrics_results[1]),
                     np.mean(metrics_results[2]),'Mean']))
        
        
    save_func(save_file,result_values)   
    
if __name__ == '__main__':
    # Data
    data_path={'data':'../dataset/data_kop_train.csv', #'../dataset/data_classification_a2a_train.csv',
             'data_test': '../dataset/data_kop_test.csv', #'../dataset/data_classification_a2a_test.csv'
            'smiles_dic':'../dictionary/dictionary_smiles.txt',
            'clusters':glob.glob('../Train_Test_Clusters_Indices/*'),
            'save_model': '../model/classification/BERT/',
            'trained_model': '../model/smiles_bert_train_4_4_RAdam_num_smiles_1M_batch_256_2'}
    
    smiles_dictionary = json.load(open(data_path['smiles_dic']))

    #df_smiles = pd.read_csv(data_path['data'])  
    
    validation = False 
    train = True
    evaluate = True
   
    
    if validation:
        
        df = pd.read_csv(data_path['data'], index_col = False)
        #df = df.iloc[0:10]
        tokens_mol = [token(x) for x in df['smiles'].tolist()]
        smiles,_= embedding_token_id_and_padding(tokens_mol, 72, smiles_dictionary)    
        pic50, q1,q3 = normalization(df['pIC50'].tolist())

        inp = tf.convert_to_tensor(smiles)
        target = tf.convert_to_tensor(pic50)[:,tf.newaxis]
        
        clusters = find_clusters(data_path['clusters'])
        
        
        for cls_units, cls_act in zip([[512,128], [512],[512,128]], [['gelu', 'relu'], ['linear'], ['linear','gelu']]):
        
            parameters = {'D_model': 512, 'Dff': 1024, 'Num_heads': 4, 'Num_encoder': 4, 'Dropout': 0.1, 'list_metrics':[q2, ccc],
                          'cls_act_func': cls_act, 'Dim_cls_layers':cls_units, 'batch_size': 32, 'checkpoint_path': data_path['trained_model']}
            
            grid_search(parameters,[inp, target],
                        [list(clusters[i][1].iloc[:,0]) for i in range(len(clusters)) if clusters[i][0]!='test'],
                        '../results/',
                        '../results/grid_results.csv')
            tf.keras.backend.clear_session()
            
           
        
        
    if train:                  
        
        
        parameters = {'D_model': 512, 'Dff': 1024, 'Num_heads': 4, 'Num_encoder': 4, 'Rate': 0.1, 'batch_size': 32, 'epochs':200, 'optimizer':'RAdam'}
        
        
        # Callbacks
        es = tf.keras.callbacks.EarlyStopping(monitor ='loss',
                                          min_delta = 0.0001, patience = 30, mode = 'min', 
                                          restore_best_weights=True)
        
    
        mc = tf.keras.callbacks.ModelCheckpoint(filepath = f'{data_path["save_model"]}/model_kop_{parameters["Num_heads"]}_{parameters["Num_encoder"]}_{parameters["optimizer"]}',
                                                monitor = 'loss',
                                            save_best_only=True, save_weights_only = False, mode = 'min')
       
        callbacks = [es,mc]#,tensorboard_callback]
        
        
        checkpoint_path = data_path['trained_model']
        #df = df_smiles.iloc[0:parameters['num_smiles']]
        
        
        
        
        ######### train ###########
        df = pd.read_csv(data_path['data'], index_col = False)
        #df = df.iloc[0:10]
        tokens_mol = [token(x) for x in df['smiles'].tolist()]
        smiles, _ = embedding_token_id_and_padding(tokens_mol, 72, smiles_dictionary)    
        pic50, q1, q3 = normalization(df['pIC50'].tolist())

        inp = tf.convert_to_tensor(smiles)
        target = tf.convert_to_tensor(pic50)[:,tf.newaxis]
        print(f'inp {inp.shape}, target {target.shape}')
	    ######### test ###########
        df_val = pd.read_csv(data_path['data_test'], index_col = False) 
        #df = df.iloc[0:10]
        tokens_mol_val = [token(x) for x in df_val['smiles'].tolist()]
        smiles_val, _ = embedding_token_id_and_padding(tokens_mol_val, 72, smiles_dictionary)    
        pic50_val, _,_ = normalization(df_val['pIC50'].tolist(), q1, q3)

        x_val = tf.convert_to_tensor(smiles_val)
        y_val = tf.convert_to_tensor(pic50_val)[:,tf.newaxis]
        print(f'x_val {x_val.shape}, y_val {y_val.shape}')
        mse_loss = tf.keras.losses.MeanSquaredError(
        name='mean_squared_error'
        )
        
        #q2 =  tfa.metrics.r_square.RSquare()
        #ccc = 
        mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
        Q2 = tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))
        list_metrics = [q2, ccc] #q2,ccc]
        
        optimizer = tfa.optimizers.RectifiedAdam(
             lr=5e-5,
             beta_1 = 0.9,
             beta_2 = 0.999,
             weight_decay = 0.1)
             
        
        smiles_bert_model = Classification_SMILES_Model(74,len(smiles_dictionary), smiles_dictionary, parameters['Rate'], 
                                              parameters['D_model'], parameters['Dff'], parameters['Num_heads'], parameters['Num_encoder'],#'adam',#RAdamOptimizer(),
                                              optimizer, 'mse', list_metrics, checkpoint_path, [], []).fit_model((inp, target),
                                             parameters['batch_size'], parameters['epochs'],callbacks,False, None)
        
                                                                                                               
                                                                                                                      
    if evaluate:
                
        parameters = {'D_model': 512, 'Dff': 1024, 'Num_heads': 4, 'Num_encoder': 4, 'Rate': 0.1, 'num_smiles': 10, 'batch_size': 32, 'epochs':200, 'optimizer':'RAdam'}
                                                                                                    
        #smiles_bert_model.model.fit_model((inp, target), parameters['batch_size'], parameters['epochs'],
                                                  #   callbacks, False, None)
        #df_val = pd.read_csv(data_path['data_val'], index_col = False) 
        df = pd.read_csv(data_path['data'], index_col = False)
        _, q1, q3 = normalization(df['pIC50'].tolist())
        
        
        df_val = pd.read_csv(data_path['data_test'], index_col = False) 
        #clusters = find_clusters(data_path['clusters'])
        #df_val = [list(clusters[i][1].iloc[:,0]) for i in range(len(clusters)) if clusters[i][0]=='test']
        #df_val = df_val[0]
        #df = df.iloc[0:10]
        tokens_mol_val = [token(x) for x in df_val['smiles'].tolist()]
        smiles_val,_ = embedding_token_id_and_padding(tokens_mol_val, 72, smiles_dictionary)    
        pic50_val,_,_ = normalization(df_val['pIC50'].tolist(), q1, q3)

        x_val = tf.convert_to_tensor(smiles_val)
        y_val = tf.convert_to_tensor(pic50_val)[:,tf.newaxis]
        
        filepath = f'{data_path["save_model"]}/model_kop_{parameters["Num_heads"]}_{parameters["Num_encoder"]}_{parameters["optimizer"]}'#_3'
        trained_model = tf.keras.models.load_model(filepath,custom_objects={'q2':q2, 'ccc':ccc}) #('../model/classification/smiles_bert_4_4_RAdam_ffn_512linear_128_gelu_1linear',custom_objects={'q2':q2, 'ccc':ccc}) 
            
        pred_metrics = inference_metrics(trained_model,[x_val,y_val])
        
        pred_values = trained_model.predict([x_val])  
        print(f' metrics {pred_metrics}')
        """
        fig, ax = plt.subplots()
        ax.scatter(y_val, pred_values, c='red',
                       alpha=0.6, edgecolors='black')
            # ax.plot([real_values.min(),real_values.max()],[real_values.min(),real_values.max()],'k--',lw = 4)
        ax.plot(y_val, y_val, 'k--', lw=2)
        ax.set_xlabel('True', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        plt.title("Adenosine A2A")
        plt.savefig('pic50.png', dpi=300)

           """                           
#from sklearn.metrics import r2_score
#r2_score(y_val.numpy(), p)
#r2(y_val.numpy(), p)
