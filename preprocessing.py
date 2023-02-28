import os, pathlib

import pandas as pd
import numpy as np

from common import get_config

def preprocess():
    root_path = pathlib.Path(__file__).parent.resolve()
    cfg = get_config()

    data_path = os.path.join(root_path, cfg["data_path"], cfg["file_name"])
    raw_data = pd.read_csv(data_path).values
    shift_data = raw_data - 1

    # for one-hot encoding
    eye = np.eye(45)
    onehot_data = eye[shift_data[:, :6]].sum(axis=1)
    if cfg["include_bonus"]:
        onehot_data += eye[shift_data[:, 6:]].sum(axis=1)*0.9 # bonus penalty
    # pd.DataFrame(onehot_data).to_csv(os.path.join(root_path, cfg["data_path"], "onehot_data.csv"), index=False)

    window_size = cfg["window_size"] + 1
    stack_data_list = []
    onehot_data_list = onehot_data.tolist()
    for i in range(len(onehot_data_list) - window_size + 1):
        stack_data_list += [onehot_data_list[i: i+window_size]]
    
    stack_data = np.array(stack_data_list)
    
    x = stack_data[:, :window_size-1, :]
    y = stack_data[:, window_size-1:, :]
    # for inference
    z = stack_data[-1:, 1:, :] 

    reshape = lambda x: x.reshape(len(x), -1)
    x, y, z = map(reshape, [x, y, z])
    # print(x.shape, y.shape, z.shape)

    pd.DataFrame(x).to_csv(os.path.join(root_path, cfg["dataset_path"], "x.csv"), index=False)
    pd.DataFrame(y).to_csv(os.path.join(root_path, cfg["dataset_path"], "y.csv"), index=False)
    pd.DataFrame(z).to_csv(os.path.join(root_path, cfg["dataset_path"], "z.csv"), index=False)


if __name__=="__main__":
    preprocess()

