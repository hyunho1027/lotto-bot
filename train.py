import os, pathlib, copy

import numpy as np
import pandas as pd
import torch

from common import get_config, Model

def train():
    cfg = get_config()
    root_path = pathlib.Path(__file__).parent.resolve()

    # load dataset
    load_dataset = lambda x: pd.read_csv(x).values
    all_x, all_y = map(load_dataset, 
                [os.path.join(root_path, cfg["dataset_path"], "x.csv"),
                os.path.join(root_path, cfg["dataset_path"], "y.csv")])

    num_train = int(len(all_x) * cfg["split_ratio"])
    
    if cfg["shuffle"]:
        perm = np.random.permutation(len(all_x))
        shuffle_x, shuffle_y = map(lambda x: x[perm], [all_x, all_y])
        train_x, train_y, valid_x, valid_y = shuffle_x[:num_train], shuffle_y[:num_train], shuffle_x[num_train:], shuffle_y[num_train:]
    else:
        train_x, train_y, valid_x, valid_y = all_x[:num_train], all_y[:num_train], all_x[num_train:], all_y[num_train:]
    
    to_tensor = lambda x : torch.FloatTensor(x)
    train_x, train_y, valid_x, valid_y = map(to_tensor, [train_x, train_y, valid_x, valid_y])

    m = Model(cfg["input_dim"] * cfg["window_size"], cfg["hidden_dim"], cfg["output_dim"], cfg["learning_rate"])
    early_stop_cnt, min_valid_loss = 0, np.inf
    for e in range(cfg['max_epoch']):
        batch_perm = np.random.permutation(len(train_x))
        train_x, train_y = map(lambda x: x[batch_perm], [train_x, train_y])
        train_loss_list = []
        for i in range(int(np.ceil(len(train_x)/cfg['batch_size']))):
            batch_x, batch_y = train_x[i*cfg['batch_size']: (i+1)*cfg['batch_size']], train_y[i*cfg['batch_size']: (i+1)*cfg['batch_size']]
            train_loss_list.append(m.optimize(batch_x, batch_y))

        train_loss = np.mean(train_loss_list)
        valid_loss = m.get_loss(valid_x, valid_y)
        train_hit = m.evaluate(train_x, train_y)
        valid_hit = m.evaluate(valid_x, valid_y)

        if cfg["debug"]:
            print(f"{e} Epoch - train loss: {train_loss:.4f} / valid_loss: {valid_loss:.4f} " +\
                            f"/ train_hit: {train_hit:.4f} / valid_hit: {valid_hit:.4f}")
        
        if min_valid_loss < valid_loss:
            early_stop_cnt += 1
        else:
            early_stop_cnt = 0
            min_valid_loss = valid_loss
            best_model = copy.deepcopy(m.net)

        if early_stop_cnt == cfg['early_stop_threshold']:
            break

    m.net = best_model
    m.save(cfg['model_path'])

    last_week_inference = m.inference(to_tensor(all_x[-1]), to_tensor(all_y[-1]))
    print("last week inference:", last_week_inference)


if __name__=="__main__":
    train()

