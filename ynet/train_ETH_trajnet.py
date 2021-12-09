import pandas as pd
import yaml
from model import YNet
from utils.preprocessing import load_and_window_ETH
import wandb

if __name__ == "__main__":

    #:param df: df
    #:param window_size: sequence-length of one trajectory, mostly obs_len + pred_len
    #:param stride: timesteps to move from one trajectory to the next one ---> If stride=window_size then there is no overlap
    #:return: df with chunked trajectories
    CONFIG_FILE_PATH = "config/eth_trajnet.yaml"  # yaml config file containing all the hyperparameters
    EXPERIMENT_NAME = "eth_trajnet"  # arbitrary name for this experiment
    DATASET_NAME = "eth"

    TRAIN_DATA_PATH = "data/ETH/train_eth.pkl"
    TRAIN_IMAGE_PATH = "data/ETH/train"
    VAL_DATA_PATH = "data/ETH/val_eth.pkl"
    VAL_IMAGE_PATH = "data/ETH/val"
    OBS_LEN = 8  # in timesteps
    PRED_LEN = 12  # in timesteps
    NUM_GOALS = 20  # K_e
    NUM_TRAJ = 1  # K_a

    BATCH_SIZE = 4

    with open(CONFIG_FILE_PATH) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    experiment_name = CONFIG_FILE_PATH.split(".yaml")[0].split("config/")[1]

    df_train = load_and_window_ETH(
        step=1, path="eth", mode="train", window_size=20, stride=20
    )
    df_val = load_and_window_ETH(
        step=1, path="eth", mode="val", window_size=20, stride=20
    )

    df_train.to_pickle("data/ETH/train_eth.pickle")
    df_val.to_pickle("data/ETH/val_eth.pickle")

    # with open(TRAIN_DATA_PATH, 'rb') as f:
    #    df_train = pickle.load(f)
    #
    # with open(VAL_DATA_PATH, 'rb') as f:
    #    df_val = pickle.load(f)

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
