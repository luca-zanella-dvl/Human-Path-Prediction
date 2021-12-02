import numpy as np
import pandas as pd
import os
import cv2
from copy import deepcopy
from modeleth import YNet
from utils.preprocessing import load_and_window_ETH
import pickle
import yaml
import wandb

if __name__== "__main__":
    
    #:param df: df
    #:param window_size: sequence-length of one trajectory, mostly obs_len + pred_len
    #:param stride: timesteps to move from one trajectory to the next one ---> If stride=window_size then there is no overlap
    #:return: df with chunked trajectories
    CONFIG_FILE_PATH = "config/eth_trajnet.yaml"
    OBS_LEN = 8  # in timesteps
    PRED_LEN = 12  # in timesteps
    NUM_GOALS = 20  # K_e
    NUM_TRAJ = 1  # K_a
    BATCH_SIZE = 4
    EXPERIMENT_NAME = 'eth_trajnet'  # arbitrary name for this experiment
    DATASET_NAME = 'eth'
    
    TRAIN_IMAGE_PATH = 'data/ETH/train'
    VAL_IMAGE_PATH = 'data/ETH/val'
    
    
    train_data = load_and_window_ETH(step=1, path="eth" , mode = "train",  window_size=20, stride=20)
    val_data = load_and_window_ETH(step=1, path="eth" , mode = "val",  window_size=20, stride=20)

    train_data.to_pickle('data/ETH/train_eth.pickle')
    val_data.to_pickle('data/ETH/val_eth.pickle')
    
    #with open('data/ETH/train_eth.pickle', 'rb') as f:
    #    train_data = pickle.load(f)
    #    
    #with open('data/ETH/val_eth.pickle', 'rb') as f:
    #    val_data = pickle.load(f)
    
    with open(CONFIG_FILE_PATH) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1] + "01"
    
    
    wandb.init(project="YNET")
    model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
    
    
    model.train(train_data, val_data, params, train_image_path= TRAIN_IMAGE_PATH, val_image_path= VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
            device=None, dataset_name=DATASET_NAME)
    
    wandb.finish()