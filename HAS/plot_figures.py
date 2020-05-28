
import pandas as pd
import io
import re
import csv
import json
import numpy as np
from tqdm import tqdm
import math
import ast
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import seaborn as sns
import keras
from keras import backend as K
from progressbar import ProgressBar
import matplotlib.pyplot as plt
from keras import Sequential, Model
from random import randrange
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as preprocessing
from keras.utils.vis_utils import plot_model
import os
import sys

def loss(target_power, predicted_power):
  return tf.math.abs((target_power - predicted_power)/target_power)

def plot_loss(history, path, save):
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    if save:
        plt.savefig(f'{path}/loss.png', dpi=300, bbox_inches='tight')


def predict(model,X_list, y,nb_predictions =100):
    y_pred_min = sys.float_info.max
    y_pred_max = 0
    nb_val = len(X_list[0])
    losses = []
    preds = []
    trues = []
    index_low=[]
    index_middle = []
    index_high = []
    for k in (range(nb_predictions)):
        i=randrange(nb_val)
        inputs=[]
        for x in range(len (X_list)): # whether one or multiple inputs
            input_x=np.array(X_list[x])[i]
            input_x=input_x.reshape((1,*input_x.shape))
            inputs.append(tf.convert_to_tensor(input_x))
        pred = model.predict(inputs)
        if pred > y_pred_max:
            y_pred_max=pred
        if pred < y_pred_min:
            y_pred_min = pred
        true = y[i]
        if true < 100:
            index_low.append(k)
        elif 100<=true <1000:
            index_middle.append(k)        
        else:
            index_high.append(k)
        preds.append(pred)
        trues.append(true)
        print(f'Index : {i}')
        print(f'pred : {pred}')
        print(f'true : {true}')
        losses.append(loss(true,pred))
        print(f'loss : {loss(true,pred)}')
        print()
    print()
    print(f'y_pred_min : {y_pred_min}')
    print(f'y_pred_max : {y_pred_max}')
    print(f'diff : {y_pred_max-y_pred_min}')
    print(f'Mean loss on predictions : {np.mean(losses)}')
    return losses, preds, trues, index_low, index_middle, index_high

def plot_predictions(losses, preds, trues, index_low, index_middle, index_high, prediction_path, save):
    nb_pred = len(preds)
    plt.figure()
    plt.plot(np.array(preds).reshape(nb_pred,-1))
    plt.plot(np.array(trues).reshape(nb_pred,-1))
    plt.title(f'All predictions. Mean loss on them : {round(np.mean(losses),2)}')
    plt.ylabel('predictions')
    plt.xlabel('range')
    plt.legend(['preds', 'trues'], loc='upper left')
    if save:
        plt.savefig(f'{prediction_path}/all_predictions.png', dpi=300, bbox_inches='tight')
    #plt.show()
    try :
        plt.figure()
        preds_low = np.array(preds)[index_low].reshape(len(index_low),-1)
        trues_low = np.array(trues)[index_low].reshape(len(index_low),-1)
        plt.plot(preds_low)
        plt.plot(trues_low)
        plt.title(f'Low Value Predictions. Mean loss on them : {round(np.mean(loss(trues_low,preds_low)),2)}')
        plt.ylabel('predictions')
        plt.xlabel('range')
        plt.legend(['preds', 'trues'], loc='upper left')
        if save:
            plt.savefig(f'{prediction_path}/low_predictions.png', dpi=300, bbox_inches='tight')
    except:
        print("No Low Value Predictions")
    #plt.show()
    try:
        plt.figure()
        preds_middle = np.array(preds)[index_middle].reshape(len(index_middle),-1)
        trues_middle = np.array(trues)[index_middle].reshape(len(index_middle),-1)
        plt.plot(preds_middle)
        plt.plot(trues_middle)
        plt.title(f'Middle Value Predictions. Mean loss on them : {round(np.mean(loss(trues_middle,preds_middle)),2)}')
        plt.ylabel('predictions')
        plt.xlabel('range')
        plt.legend(['preds', 'trues'], loc='upper left')
        if save :
            plt.savefig(f'{prediction_path}/middle_predictions.png', dpi=300, bbox_inches='tight')
    #plt.show()
    except:
        print("No Middle Value Predictions")

    try:
        plt.figure()
        preds_high = np.array(preds)[index_high].reshape(len(index_high),-1)
        trues_high = np.array(trues)[index_high].reshape(len(index_high),-1)
        plt.plot(preds_high)
        plt.plot(trues_high)
        plt.title(f'High Value Predictions. Mean loss on them : {round(np.mean(loss(trues_high,preds_high)),2)}')
        plt.ylabel('predictions')
        plt.xlabel('range')
        plt.legend(['preds', 'trues'], loc='upper left')
        if save:
            plt.savefig(f'{prediction_path}/high_predictions.png', dpi=300, bbox_inches='tight')
    except:
        print("No High Value Predictions")
    #plt.show()

def save_architecture(model_to_save, path,save):
    if save:
        tf.keras.utils.plot_model(
            model_to_save, to_file=f'{path}/architecture.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=70
        )

def save_model(path, model, history, X_list, y, name='', nb_predictions = 100, max_val_loss=0.30, nb_final_epochs_for_mean = 3, save=True):
    test_performance = np.mean((history.history["val_loss"][-nb_final_epochs_for_mean:]))
    if test_performance <max_val_loss:
        path = f'{path}/{name}_{round(test_performance,3)}_error'
        try :
            if save:
                os.mkdir(path)
        except:
            print('Directory already exists')
        plot_loss(history, path,save)
        losses, preds, trues, index_low, index_middle, index_high = predict(model, X_list, y,nb_predictions = nb_predictions)
        plot_predictions(losses, preds, trues, index_low, index_middle, index_high, f'{path}',save)
        save_architecture(model, path,save)
        model.save_weights(f'{path}/{name}')



