import re
import timeit
import pdb
import pandas as pd

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from tensorflow.python.keras.layers import BatchNormalization, Dense, Dropout, Input, Lambda
from tensorflow.python.keras.models import Model

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--cvset",      default=0,          type=int,    help="Random seed for train/validation split")
parser.add_argument("--n_features", default=500,        type=int,    help="Number of features")
parser.add_argument("--batch_size", default=1000,       type=int,    help="Batch size")
parser.add_argument("--p_drop",     default=0.5,        type=float,  help="Dropout rate")
parser.add_argument("--latent_dim", default=2,          type=int,    help="Latent dims")

parser.add_argument("--n_epoch",    default=5000,       type=int,    help="Number of epochs to train")
parser.add_argument("--run_iter",   default=0,          type=int,    help="Run-specific id")
parser.add_argument("--exp_name",   default='synphys',  type=str,    help="Experiment name")
parser.add_argument("--model_id",   default='v1',       type=str,    help="Model name")

def main(cvset=0,n_features=5000,
         batch_size=1000, p_drop=0.5, latent_dim=2,
         n_epoch=5000, 
         run_iter=0 , exp_name='nagent', model_id='nagent_model'):
    train_dict, val_dict, full_dict, dir_pth = dataIO(cvset=0,n_features = n_features,exp_name=exp_name, train_size=25000)
    
    #Architecture parameters ------------------------------
    input_dim = train_dict['X'].shape[1]
    print(input_dim)
    fc_dim = 50
    
    fileid = model_id + \
        '_cv_' + str(cvset) + \
        '_ng_' + str(n_features) + \
        '_pd_' + str(p_drop) + \
        '_bs_' + str(batch_size) + \
        '_ld_' + str(latent_dim) + \
        '_ne_' + str(n_epoch) + \
        '_ri_' + str(run_iter)
    fileid = fileid.replace('.', '-')
    print(fileid)
    
    n_agents = 1 
    #Model definition -----------------------------------------------
    M = {}
    M['in_ae']  = Input(shape=(input_dim,), name='in_ae')
    M['mask_ae'] = Input(shape=(input_dim,), name='mask_ae')
    for i in range(n_agents):
        
        M['dr_ae_'+str(i)]   = Dropout(p_drop, name='dr_ae_'+str(i))(M['in_ae'])
        M['fc01_ae_'+str(i)] = Dense(fc_dim, activation='elu', name='fc01_ae_'+str(i))(M['dr_ae_'+str(i)])
        M['fc02_ae_'+str(i)] = Dense(fc_dim, activation='elu', name='fc02_ae_'+str(i))(M['fc01_ae_'+str(i)])
        M['fc03_ae_'+str(i)] = Dense(fc_dim, activation='elu', name='fc03_ae_'+str(i))(M['fc02_ae_'+str(i)])
        M['fc04_ae_'+str(i)] = Dense(fc_dim, activation='elu', name='fc04_ae_'+str(i))(M['fc03_ae_'+str(i)])
        M['fc05_ae_'+str(i)] = Dense(latent_dim, activation='linear',name='fc05_ae_'+str(i))(M['fc04_ae_'+str(i)])
        M['ld_ae_'+str(i)]   = BatchNormalization(scale = False, center = False ,epsilon = 1e-10, momentum = 0. ,name='ld_ae_'+str(i))(M['fc05_ae_'+str(i)])
        
        M['fc06_ae_'+str(i)] = Dense(fc_dim, activation='elu', name='fc06_ae_'+str(i))(M['ld_ae_'+str(i)])
        M['fc07_ae_'+str(i)] = Dense(fc_dim, activation='elu', name='fc07_ae_'+str(i))(M['fc06_ae_'+str(i)])
        M['fc08_ae_'+str(i)] = Dense(fc_dim, activation='elu', name='fc08_ae_c'+str(i))(M['fc07_ae_'+str(i)])
        M['fc09_ae_'+str(i)] = Dense(fc_dim, activation='elu', name='fc09_ae_'+str(i))(M['fc08_ae_'+str(i)])
        M['ou_ae_'+str(i)]   = Dense(input_dim, activation='linear', name='ou_ae_'+str(i))(M['fc09_ae_'+str(i)])

    AE = Model(inputs=[M['in_ae'],M['mask_ae']],
               outputs=[M['ou_ae_' + str(i)] for i in range(n_agents)])

    def masked_mse(X, Y, mask):
        loss_val = tf.reduce_mean(tf.multiply(tf.math.squared_difference(X, Y), mask))
        def masked_loss(y_true, y_pred):
            return loss_val
        return masked_loss
       
    #Create loss dictionary
    loss_dict = {'ou_ae_'+str(i): masked_mse(M['in_ae'],M['ou_ae_0'],M['mask_ae']) for i in range(n_agents)}
    
    #Loss weights dictionary
    loss_wt_dict = {'ou_ae_'+str(i): 1.0 for i in range(n_agents)}

    #Add loss definitions to the model
    AE.compile(optimizer='adam', loss=loss_dict, loss_weights=loss_wt_dict)

    #Custom logging
    cb_obj = CSVLogger(filename=dir_pth['logs']+fileid+'.csv')

    train_input_dict  = {'in_ae': train_dict['X'],
                         'mask_ae': train_dict['mask']}
    train_output_dict = {'ou_ae_'+str(i): train_dict['X'] for i in range(n_agents)}

    val_input_dict  = {'in_ae': val_dict['X'],
                         'mask_ae': val_dict['mask']}
    val_output_dict = {'ou_ae_'+str(i): val_dict['X'] for i in range(n_agents)}


    #Model training
    start_time = timeit.default_timer()
    AE.fit(train_input_dict, train_output_dict,
           batch_size=batch_size, initial_epoch=0, epochs=n_epoch,
           validation_data=(val_input_dict,val_output_dict),
           verbose=2, callbacks=[cb_obj])

    elapsed = timeit.default_timer() - start_time

    print('-------------------------------')
    print('Training time:',elapsed)
    print('-------------------------------')

    #Save weights
    AE.save_weights(dir_pth['result']+fileid+'-modelweights'+'.h5')

    #Generate summaries
    summary = {}
    for i in range(n_agents):
        encoder = Model(inputs=M['in_ae'], outputs=M['ld_ae_'+str(i)])
        summary['z']   = encoder.predict(full_dict['X'])
        
    sio.savemat(dir_pth['result']+fileid+'-summary.mat', summary)
    return


def dataIO(cvset=0,n_features = 500,train_size = 25000,exp_name='',standardize=True):
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    
    curr_path = str(Path().absolute())
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellTypes/'
    elif '/home/rohan' in curr_path:
        base_path = '/home/rohan/Dropbox/AllenInstitute/CellTypes/'
    elif '/allen' in curr_path:
        base_path = '/allen/programs/celltypes/workgroups/mousecelltypes/Rohan/'
    else: #beaker relative paths
        base_path = '/'

    dir_pth = {}
    dir_pth['data']       = base_path + 'dat/raw/synphys/'
    dir_pth['result']     = base_path + 'dat/result/' + exp_name + '/'
    dir_pth['logs']       = dir_pth['result'] + 'logs/'
    Path(dir_pth['result']).mkdir(parents=True, exist_ok=True)
    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True)

    df=pd.read_csv(dir_pth['data']+'autoencoder_data_09_06_2019.csv', sep='#',low_memory=False)
    df.astype({'expt': str}) # to see experiment id as str
    feature_list = df.keys().values[7:]
    data = df[feature_list].values

    ind = np.argsort(np.sum(np.isnan(data)/np.shape(data)[0],axis = 0))
    keep = ind[:n_features]

    data = data[:,keep]
    feature_list[keep]
    X = (data-np.nanmean(data,axis=0))/np.nanstd(data,axis=0)

    full_dict={}
    full_dict['X'] = X
    full_dict['feature'] = feature_list

    train_dict={}
    val_dict={}

    train_dict['X'], val_dict['X'] = train_test_split(full_dict['X'][:, :n_features],
                                                      train_size=train_size,
                                                      test_size=full_dict['X'].shape[0] - train_size,
                                                      random_state=cvset)
    train_dict['mask'] = (~np.isnan(train_dict['X'])).astype('float')
    train_dict['X'][train_dict['mask']==0]=0

    val_dict['mask'] = (~np.isnan(val_dict['X'])).astype('float')
    val_dict['X'][val_dict['mask']==0]=0

    full_dict['mask'] = (~np.isnan(full_dict['X'])).astype('float')
    full_dict['X'][full_dict['mask']==0]=0

    return train_dict, val_dict, full_dict, dir_pth

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
