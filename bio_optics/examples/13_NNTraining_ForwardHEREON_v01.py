import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import lmfit

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Convolution2D, Flatten, Dense, Concatenate
from tensorflow.keras import layers
# from tensorflow.keras.models import load_model

from bio_optics.water import absorption, attenuation, backscattering, scattering, lee
from bio_optics.atmosphere import downwelling_irradiance
from bio_optics.models import hereon, model
from bio_optics.helper import resampling, utils, owt, indices, plotting

from bio_optics.water import fluorescence

def setup_model_name(arch, Ninput, Noutput):
    nodes_layer = arch
    model_name = 'I' + str(Ninput) + 'x' + str(nodes_layer[0])
    if len(nodes_layer) > 1:
        for i in range(len(nodes_layer) - 1):
            # Add one hidden layer
            model_name = model_name + 'x' + str(nodes_layer[i + 1])
    model_name = model_name + 'xO' + str(Noutput)
    return model_name


def setup_folder_NNtraining(path, arch, file_ending, Ninput, Noutput):
    folder_name = setup_model_name(arch, Ninput, Noutput) + '_' + file_ending

    newpath = path + folder_name
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    if not os.path.exists(newpath + '/temp/'):
        os.makedirs(newpath + '/temp/')

    return folder_name

def setup_input_output_data_byLabel(data, output_labels=[], input_labels=[]):
    # Specify the data
    print(len(input_labels))
    if len(input_labels) > 0:
        X = np.array(data[input_labels])
    else:
        X = data
        for label in output_labels:
            X = X.drop(label, axis=1)
            # Specify the target labels and flatten the array
    if len(output_labels) > 1:
        Y = np.array(data[output_labels])
    else:
        Y = np.ravel(data[output_labels])

    for label in output_labels:
        print(label, np.sum(data[label].values))

    return X, Y

def train_test_split(X, Y, abs_size=2000):
    ## randomly split:
    ID = np.random.choice(np.arange(0, X.shape[0], 1, dtype='int'), size=abs_size, replace=False)
    X_test = X[ID, :]
    if len(Y.shape)==1:
        Y_test = Y[ID,]
    else:
        Y_test = Y[ID, :]

    IDbool = np.ones(X.shape[0], dtype='int')
    IDbool[ID] = 0
    X_train = X[IDbool == 1, :]

    if len(Y.shape) == 1:
        Y_train = Y[IDbool==1,]
    else:
        Y_train = Y[IDbool == 1, :]
    X_train = np.array(X_train, dtype='float32')
    Y_train = np.array(Y_train, dtype='float32')
    X_test = np.array(X_test, dtype='float32')
    Y_test = np.array(Y_test, dtype='float32')
    return X_train, X_test, Y_train, Y_test


def setup_model_layers_tensorflow(model_dict):
    kernel_regularizer = model_dict['kernel_regularizer']
    nodes_layer = model_dict['Nodes_layer']
    activation_layer = model_dict['activation_function_layer']
    if len(nodes_layer)>1:
        if len(activation_layer) == 1:
            activation_layer = np.repeat(activation_layer, len(nodes_layer))

    # Add an input layer
    # inputs = keras.Input(shape=(784,), name='digits')
    # x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    # x = layers.Dense(64, activation='relu', name='dense_2')(x)
    # outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    # model = keras.Model(inputs=inputs, outputs=outputs)

    inputLayer = keras.Input(shape=(model_dict['Ninput'],), name='input')

    model_name = 'I' + str(model_dict['Ninput'])

    x = None
    for i in range(len(nodes_layer)):
        if i == 0:
            if activation_layer[i] == 'leakyRelu':
                x = layers.Dense(nodes_layer[i], kernel_regularizer=kernel_regularizer)(inputLayer)
                x = layers.LeakyReLU()(x)
            else:
                x = layers.Dense(nodes_layer[i], activation=activation_layer[i], kernel_regularizer=kernel_regularizer)(inputLayer)
        else:
            if activation_layer[i] == 'leakyRelu':
                x = layers.Dense(nodes_layer[i], kernel_regularizer=kernel_regularizer)(x)
                x = layers.LeakyReLU()(x)
            else:
                x = layers.Dense(nodes_layer[i], activation=activation_layer[i], kernel_regularizer=kernel_regularizer)(x)
        # model_list.append(layers.Dense(nodes_layer[i], activation=activation_layer[i])(last_layer))
        # last_layer = model_list[-1]
        model_name += 'x'+str(nodes_layer[i])


    # Add an output layer
    if model_dict['activation_function_output_layer'] == '':
        outputs = layers.Dense(model_dict['Noutput'], kernel_regularizer=kernel_regularizer)(x)
        # model_list.append(Dense(model_dict['Noutput'])(last_layer))
    else:
        print(model_dict['activation_function_output_layer'])
        outputs = layers.Dense(model_dict['Noutput'], activation=model_dict['activation_function_output_layer'], kernel_regularizer=kernel_regularizer)(x)
        # model_list.append(Dense(model_dict['Noutput'], activation=model_dict['activation_function_output_layer'])(last_layer))

    model_name = model_name + 'xO' + str(model_dict['Noutput'])
    return inputLayer, outputs, model_name


def plot_Hist_categories(ytest, ypred, output_label, plotPath, model_fn, epoch=None, hist_range=(-0.2,1.2)):

    minY = np.min(ytest)
    maxY = np.max(ytest)
    delta = (maxY-minY)/5.
    hist_range = (minY-delta, maxY + delta)

    if len(ytest.shape) > 1:
        for i in range(ytest.shape[1]):
            levels = np.unique(ytest[:, i])
            for lev in levels:
                ID = np.array(ytest[:, i] == lev)
                plt.hist(ypred[ID, i], 50, range=hist_range)
                fig_path = plotPath + 'NN_test_' + output_label[i] + '_' + str(lev) + '_' + model_fn + '_' + str(epoch)+ '.png'
                plt.savefig(fig_path, dpi=200)
                plt.close()
    elif len(ytest.shape) == 1:
        levels = np.unique(ytest)
        for lev in levels:
            ID = np.array(ytest == lev)
            plt.hist(ypred[ID], 50, range=hist_range)
            fig_path = plotPath + 'NN_test_' + output_label[0] + '_' + str(lev) + '_' + model_fn +'_' + str(epoch)+ '.png'
            plt.savefig(fig_path, dpi=200)
            plt.close()


def transform_NorthSea_IOPs(iopArr, reverse = False):
    # set fixed IOP values, for variable: 'C_0', 'C_2', 'C_5', 'C_6', 'C_7', 'C_Y', 'C_ism', 'L_fl_lambda0', 'b_ratio_md', 'b_ratio_bd', 'S_cdom'
    varList = ['C_0', 'C_2', 'C_5', 'C_6', 'C_7', 'C_Y', 'C_ism', 'L_fl_lambda0', 'b_ratio_md', 'b_ratio_bd', 'S_cdom']
    rangeDict = {
        'C_0': [0., 1000.],
        'C_2': [0., 1000.],
        'C_5': [0., 1000.],
        'C_6': [0., 1000.],
        'C_7': [0., 1.],
        'C_Y': [0., 2.],
        'C_ism': [0., 100.],
        'L_fl_lambda0': [0., 0.2],
        'b_ratio_md': [0.021, 0.3756],
        'b_ratio_bd': [0.021, 0.3756],
        'S_cdom': [0.005, 0.032]
    }

    iopArrT = np.zeros(iopArr.shape)
    for i, v in enumerate(varList):
        delta = rangeDict[v][1] - rangeDict[v][0]
        if reverse:
            iopArrT[:, i] = iopArr[:, i]*delta + delta / 2.
        else:
            iopArrT[:, i] = (iopArr[:, i] - delta / 2.) / delta

    return iopArrT

def transform_NorthSea_IOPs_single(x, v, reverse = False):
    # set fixed IOP values, for variable: 'C_0', 'C_2', 'C_5', 'C_6', 'C_7', 'C_Y', 'C_ism', 'L_fl_lambda0', 'b_ratio_md', 'b_ratio_bd', 'S_cdom'
    varList = ['C_0', 'C_2', 'C_5', 'C_6', 'C_7', 'C_Y', 'C_ism', 'L_fl_lambda0', 'b_ratio_md', 'b_ratio_bd', 'S_cdom']
    rangeDict = {
        'C_0': [0., 1000.],
        'C_2': [0., 1000.],
        'C_5': [0., 1000.],
        'C_6': [0., 1000.],
        'C_7': [0., 1.],
        'C_Y': [0., 2.],
        'C_ism': [0., 100.],
        'L_fl_lambda0': [0., 0.2],
        'b_ratio_md': [0.021, 0.3756],
        'b_ratio_bd': [0.021, 0.3756],
        'S_cdom': [0.005, 0.032]
    }

    delta = rangeDict[v][1] - rangeDict[v][0]
    if reverse:
        x = x*delta + delta / 2.
    else:
        x = (x - delta / 2.) / delta

    return x



def NN_tensorflow_training_IOP2Model(d, NNdict, NNtraining_metadata, outpath = '', epochs=1000, batch_size=500,
                                    resume=False, ndigits=5, intermediate_results=True, image_outpath=''):
    print(tf.__version__)
    print(tf.keras.__version__)

    ###
    # split into trainig + validation.
    X, Y = setup_input_output_data_byLabel(d, NNtraining_metadata['output_label'], NNtraining_metadata['input_names'])

    print('training data', X.shape)

    ###
    # Transformation of input data
    if NNtraining_metadata['transformation_method'] != '' :
        if NNtraining_metadata['transformation_method'] == 'log10':
            X = np.log10(X)
        if NNtraining_metadata['transformation_method'] == 'sqrt':
            X = np.sqrt(X)
        if NNtraining_metadata['transformation_method'] == 'IOP_NorthSea_ranges':
            ###
            # transform/scale IOP data !
            X = transform_NorthSea_IOPs(X)

    ###
    # check: remove nan!
    ID = np.ones(X.shape[0]) * True
    for i in range(X.shape[1]):
        ID = np.logical_and(ID, np.isfinite(X[:, i]))

    X = X[ID, :]

    if len(NNtraining_metadata['output_label']) == 1:
        Y = Y[ID,]
    else:
        Y = Y[ID, :]
    print('training data', X.shape)
    print('test size N: ', int(X.shape[0]*NNdict['test_size']))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, abs_size=int(X.shape[0]*NNdict['test_size'])) #2000

    if NNdict.get('Ninput') != X.shape[1]:
        NNdict['Ninput'] = X.shape[1]

    if len(Y.shape)==1:
        NNdict['Noutput'] = 1
    else:
        if NNdict.get('Noutput') != Y.shape[1]:
            NNdict['Noutput'] = Y.shape[1]

    nn_input, nn_output, NNname = setup_model_layers_tensorflow(NNdict)
    epoch_start = 0
    bestLoss = None
    if not resume:
        model_ = keras.Model(inputs=nn_input, outputs=nn_output)
    else:
        NNfnames = os.listdir(outpath)
        NNfnames = [fn for fn in NNfnames if fn.endswith('.h5')]
        epochs_ = [int(a.split('epoch')[1].split('_')[0]) for a in NNfnames]
        epochs_ = np.asarray(epochs_)
        ID = np.array(epochs_ == np.max(epochs_))
        bestLoss = [float(a.split('loss')[1].split('.h5')[0]) for a in NNfnames]
        bestLoss = np.min(np.asarray(bestLoss))
        if bestLoss == 0.:
            bestLoss = 0.001
        NNfnames = np.asarray(NNfnames)[ID][0]
        fwpath = outpath + NNfnames
        epoch_start = np.max(epochs_)
        model_ = tf.keras.models.load_model(fwpath)

    ##
    model_.summary()

    ## todo: best activation function, best optimizer, best loss function??
    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.Nadam()
    # Instantiate a loss function.
    if NNdict['activation_function_output_layer'] == 'softmax':
        print('softmax categorical crossentropy')
        loss_fn = keras.losses.CategoricalCrossentropy()
        model_.compile(optimizer=optimizer, loss="categorical_crossentropy")
    elif NNdict['activation_function_output_layer'] == 'sigmoid':
        loss_fn = keras.losses.BinaryCrossentropy()
        model_.compile(loss='binary_crossentropy', optimizer=optimizer) # optimizer = 'sgd'
    else:
        loss_fn = keras.losses.MeanSquaredError()
        model_.compile(optimizer=optimizer, loss="mean_squared_error")


    # Prepare the training dataset.
    train_dataset_fw = tf.data.Dataset.from_tensor_slices((Y_train, X_train))
    train_dataset_fw = train_dataset_fw.shuffle(buffer_size=1024).batch(batch_size)

    loss_ = np.zeros(epochs)
    if resume:
        loss_ += bestLoss

    writeNextBestNN = False
    outfname = ''
    loss_value_ = 0.

    for epoch in range(epochs):
        if epoch >= epoch_start:
            # print('Start of epoch %d' % (epoch,))
            for step, (Y_batch, X_batch) in enumerate(train_dataset_fw.take(1), 1):
                # define two GradientTapes:
                with tf.GradientTape() as tape_fw:
                    logits_ = model_(X_batch)
                    loss_value_ = loss_fn(Y_batch, logits_)

                grads = tape_fw.gradient(loss_value_, model_.trainable_weights)
                optimizer.apply_gradients(zip(grads, model_.trainable_weights))
                loss_[epoch] = float(loss_value_)

            if epoch == 0:
                bestLoss = loss_[epoch]

            if epoch % 500 == 0 and intermediate_results and epoch > 0:
                writeNextBestNN = True

            if loss_[epoch] < bestLoss:
                bestLoss = loss_[epoch]

                if writeNextBestNN:
                    outfname = NNname + '_batch' + str(batch_size) + '_epoch' + str(epoch) + \
                               '_loss' + str(np.round(loss_[epoch], ndigits))
                    model_.save(outpath + outfname + '.h5')

                    #
                    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
                    axes.semilogy(loss_, 'g-')
                    axes.set_xlabel('epoch number')
                    axes.set_ylabel('loss (mean squared error)')
                    fig.savefig(image_outpath + 'loss_' + outfname + '.png', dpi=200)
                    plt.close()

                    #
                    clearNN_pred = model_.predict(X_test)
                    # plot_Hist_categories(ytest=Y_test, ypred=clearNN_pred, output_label=NNtraining_metadata['output_label'],
                    #                      model_fn=outfname, epoch=epoch, plotPath=image_outpath, hist_range=(0.,7.))
                    # plot_Hist_categories(ytest=Y_test, ypred=clearNN_pred, output_label=NNtraining_metadata['output_label'],
                    #                      model_fn=outfname, epoch=epoch, plotPath=image_outpath)

                    writeNextBestNN =False


            if epoch % 100 == 0:
                print('Training loss (for one batch) at step %s: %s' % (epoch, float(loss_value_)))
                print('Seen so far: %s samples' % ((epoch + 1) * batch_size))


    outfname = NNname + '_batch'+ str(batch_size) + '_epoch' + str(epochs) +'_loss'+ str(np.round(loss_[-1], ndigits))
    model_.save(outpath + outfname + '.h5')

    clearNN_pred = model_.predict(X_test)

    print(X_test.shape)

    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    axes.semilogy(loss_, 'g-')
    fig.savefig(image_outpath + 'loss_'+outfname+'.png', dpi=200)
    plt.close()
    # plt.show()

    ###
    # validation
    # plot_Hist_categories(ytest=Y_test, ypred=clearNN_pred, output_label=NNtraining_metadata['output_label'],
    #                      model_fn=outfname, plotPath=outpath, hist_range=(0.,7.))


####
# setup Training data
###
def setup_trainingdata():
    fname = "E:\Documents\projects\EnsAD\data\EnMAP_NN_training\AANN_NorthSea\\trainingData_EnMAPSpectra_NorthSeaEnMAPSouth20230815_InvertibleCategory_selection20240405_v2.txt"
    rrs_inv_fname = "E:\Documents\projects\EnsAD\inversion\HZG_HEREON_groups\\results_MK\inverted_Rrs_bio_optics_HEREONfull_NorthSeaEnMAPSouth20230815_V7AHall_md_bd_v2.txt"
    iop_inv_fname = "E:\Documents\projects\EnsAD\inversion\HZG_HEREON_groups\\results_MK\inverted_IOP_bio_optics_HEREONfull_NorthSeaEnMAPSouth20230815_V7AHall_md_bd_v2.txt"

    df = pd.read_csv(fname, header=0, sep='\t') # already filtered for nan OWT
    rrs_inv = pd.read_csv(rrs_inv_fname, header=0, sep='\t') # remove the first 300 spectra (OWT nan)
    iop_inv = pd.read_csv(iop_inv_fname, header=0, sep='\t') # remove the first 300 spectra (OWT nan)

    rrscols = rrs_inv.columns.values
    invcols = ['inv_'+col for col in rrs_inv.columns.values]

    wavelengths = np.asarray([float(wl) for wl in rrs_inv.columns.values])
    ID = np.array(df['invertible'].values == 1)

    print(df.columns.values)
    print(rrs_inv.columns.values)
    print(iop_inv.columns.values)

    for col in rrs_inv.columns.values:
        rrs_inv = rrs_inv.rename(columns={col: 'inv_'+col})

    rrs_inv = rrs_inv.iloc[300:, :]
    iop_inv = iop_inv.iloc[300:, :]

    out = rrs_inv.copy()
    for cols in iop_inv.columns.values:
        out[cols] = iop_inv[cols].values
    print(df.shape, rrs_inv.shape, iop_inv.shape)
    print(out.shape)

    out = out.loc[ID,:]
    out.to_csv("E:\Documents\projects\EnsAD\data\EnMAP_NN_training\\NN_NorthSea\\trainingData_EnMAPSpectra_NorthSeaEnMAPSouth20230815_Model2IOP_invertible_v20240408_v2.txt", header=True, index=False, sep='\t')
    # return out, rrscols, invcols

    # NList = [1, 350, 602]
    # for i in NList:
    #     # plt.plot(wavelengths, df[rrscols].iloc[i,:], 'b-')
    #     # plt.plot(wavelengths, rrs_inv.iloc[i, :], 'r-')
    #     plt.plot(wavelengths, out[rrscols].iloc[i, :], 'b-')
    #     plt.plot(wavelengths, out[invcols].iloc[i, :], 'r-')
    #     plt.show()



NNdict = {
    'Ninput': 21,
    'Noutput': 10,
    'Nodes_layer': [], #set an architecture of fully connected hidden layers
    'model_type': 'regression_nadam',  # 'regression' or 'regression_nadam', uses optimizer 'nadam'
    'kernel_regularizer': 'l1', #  None,
    # 'regulizer_layer': 0.001, #old comment: ist noch hartcodiert in trainingNN_general, Mit regulizer trainiert das Netz nicht mehr!!
    'activation_function_layer': ['leakyRelu'], #relu # https://www.v7labs.com/blog/neural-networks-activation-functions: softmax should be the right choice #relu, tanh, sigmoid; for OLCI_Schiller: sigmoid, relu; 'leakyReLu'
    'activation_function_output_layer': '', #softmax for mutually exclusive categories!, sigmoid
    # 'gaussian_noise_std': 0.05, # für OLCI_Schiller ohne noise
    'test_size': 0.2,
    'callbacks': [],  # or csv_logger
    'learning_rate': [],
    'batch_size': 5, # 10 # 500; für OLCI_Schiller: 1 # für 6NodesOutput blocksize = 1000
    'scaling': False
}


NNtraining_metadata = {
    'training_data_path': "E:\Documents\projects\EnsAD\data\EnMAP_NN_training\\NN_NorthSea\\trainingData_EnMAPSpectra_NorthSeaEnMAPSouth20230815_Model2IOP_invertible_v20240408.txt",
    'input_names': ['C_0', 'C_2', 'C_5', 'C_6', 'C_7', 'C_Y', 'C_ism', 'L_fl_lambda0', 'b_ratio_md', 'b_ratio_bd', 'S_cdom'],
    'output_label': ['inv_418.24', 'inv_423.874', 'inv_429.294', 'inv_434.528', 'inv_439.603', 'inv_444.549', 'inv_449.391', 'inv_454.159', 'inv_458.884',
                    'inv_463.584', 'inv_468.265', 'inv_472.934', 'inv_477.599', 'inv_482.265', 'inv_486.941', 'inv_491.633', 'inv_496.349', 'inv_501.094',
                    'inv_505.87', 'inv_510.678', 'inv_515.519', 'inv_520.397', 'inv_525.313', 'inv_530.268', 'inv_535.265', 'inv_540.305', 'inv_545.391',
                    'inv_550.525', 'inv_555.71', 'inv_560.947', 'inv_566.239', 'inv_571.587', 'inv_576.995', 'inv_582.464', 'inv_587.997', 'inv_593.596',
                    'inv_599.267', 'inv_605.011', 'inv_610.833', 'inv_616.737', 'inv_622.732', 'inv_628.797', 'inv_634.919', 'inv_641.1', 'inv_647.341',
                    'inv_653.643', 'inv_660.007', 'inv_666.435', 'inv_672.927', 'inv_679.485', 'inv_686.11', 'inv_692.804', 'inv_699.567', 'inv_706.401',
                    'inv_713.307', 'inv_720.282', 'inv_727.324', 'inv_734.431', 'inv_741.601', 'inv_748.833'],
    'transformation_method': 'IOP_NorthSea_ranges', # 'log', 'sqrt', 'IOP_NorthSea_ranges'
    'architectures': [[110, 70, 60]],  # [80, 80, 80, 80], [110, 100, 90], [80, 80, 80]
    'outpath': "E:\Documents\projects\EnsAD\\NN_training\\NN_IOP2ModelRrs_fwHEREON\\fwnn_HEREON_20240409_v01\\",
    'maxEpochs': 100000, #30, bei block_size=1; 15000 bei block_size=1000
    'file_ending': '_l1RegTrans',
    'InputMin': '',
    'InputMax': '',
    'OutputMin': '',
    'OutputMax': ''
}

###
# TRAINING or FIND the BEST
trainingRun = True

for arch in NNtraining_metadata['architectures'][:]:
    print(arch)
    NNdict['Nodes_layer'] = arch            #architecture of the hidden layers in the NN.
    input_label = NNtraining_metadata['input_names'] # training data input, column names
    output_label = NNtraining_metadata['output_label']  # training data output, column names (1 or more columns)

    Ninput = len(input_label)
    file_ending = NNtraining_metadata['transformation_method'] + NNtraining_metadata['file_ending']

    outpath = NNtraining_metadata['outpath']
    folder_NNtraining = setup_folder_NNtraining(outpath, arch, file_ending, Ninput, len(output_label))
    plotPath = outpath + folder_NNtraining + '/'

    ###
    # Read data
    d = pd.read_csv(NNtraining_metadata['training_data_path'], header=0, sep='\t')

    if trainingRun:
        with open(plotPath + 'Metadata_' + folder_NNtraining + '.txt', 'w') as file:
            file.write(json.dumps(NNtraining_metadata))
        file.close()
        with open(plotPath + 'MetadataModel_' + folder_NNtraining + '.txt', 'w') as file:
            file.write(json.dumps(NNdict))
        file.close()

        NN_tensorflow_training_IOP2Model(d, NNdict, NNtraining_metadata, outpath=plotPath + 'temp/', image_outpath=plotPath,
                                    epochs=NNtraining_metadata['maxEpochs'], batch_size=NNdict['batch_size'], resume=True)

    # if testTheBest:
        # find_bestNN_and_threshold_for_output_categories(d, NNtraining_metadata, NNpath=plotPath+ 'temp/', outpath=outpath)
        # find_bestNN_and_threshold_for_output_categories_with_N_levels(d, NNtraining_metadata, NNpath=plotPath+'temp/',
        #                                                               outpath=outpath, sigmoid=True, levels=[0., 0.5, 1.],
        #                                                               plotThis=True)
        # find_bestNN_BinaryOutput(d, NNtraining_metadata, NNpath=plotPath+'temp/', outpath=outpath)

# setup_trainingdata()