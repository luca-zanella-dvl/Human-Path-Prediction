import pandas as pd
import yaml
import argparse
import torch
from model import YNet
import wandb

from ynet.utils.preprocessing import load_and_window_MT

if __name__ == "__main__":
    #:param df: df
	#:param window_size: sequence-length of one trajectory, mostly obs_len + pred_len
	#:param stride: timesteps to move from one trajectory to the next one ---> If stride=window_size then there is no overlap
	#:return: df with chunked trajectories
	CONFIG_FILE_PATH = "config/mt_trajnet.yaml"  # yaml config file containing all the hyperparameters
	EXPERIMENT_NAME = "mt_trajnet"  # arbitrary name for this experiment
	DATASET_NAME = "mt"

	TRAIN_DATA_PATH = "data/MT/train_mt.pkl"
	TRAIN_IMAGE_PATH = "data/MT/train"
	VAL_DATA_PATH = "data/MT/val_mt.pkl"
	VAL_IMAGE_PATH = "data/MT/val"
	OBS_LEN = 8  # in timesteps
	PRED_LEN = 12  # in timesteps
	NUM_GOALS = 20  # K_e
	NUM_TRAJ = 1  # K_a

	BATCH_SIZE = 4

	with open(CONFIG_FILE_PATH) as file:
		params = yaml.load(file, Loader=yaml.FullLoader)
	experiment_name = CONFIG_FILE_PATH.split(".yaml")[0].split("config/")[1]

	df_train = load_and_window_MT(
		path="data/MT/", mode="train", step=5, window_size=20, stride=20
	)
	df_val = load_and_window_MT(
		path="data/MT/", mode="val", step=5, window_size=20, stride=20
	)

	df_train.to_pickle(TRAIN_DATA_PATH)
	df_val.to_pickle(VAL_DATA_PATH)

	# with open(TRAIN_DATA_PATH, 'rb') as f:
	#    df_train = pickle.load(f)
	#
	# with open(VAL_DATA_PATH, 'rb') as f:
	#    df_val = pickle.load(f)

	print(df_train.head())

	wandb.init(project="YNet")
	model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

	model.train(
	    df_train,
	    df_val,
	    params,
	    train_image_path=TRAIN_IMAGE_PATH,
	    val_image_path=VAL_IMAGE_PATH,
	    experiment_name=EXPERIMENT_NAME,
	    batch_size=BATCH_SIZE,
	    num_goals=NUM_GOALS,
	    num_traj=NUM_TRAJ,
	    device=None,
	    dataset_name=DATASET_NAME,
	)

	wandb.finish()