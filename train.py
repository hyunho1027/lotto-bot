import os, pathlib, copy

import numpy as np
import pandas as pd
import torch
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from util import get_config
from model import Model

def load_dataset(cfg):
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

    test_x, test_y = all_x[-1], all_y[-1]

    to_tensor = lambda x : torch.FloatTensor(x)
    train_x, train_y, valid_x, valid_y, test_x, test_y = map(to_tensor, [train_x, train_y, valid_x, valid_y, test_x, test_y])

    return train_x, train_y, valid_x, valid_y, test_x, test_y

def fit(params, train_x, train_y, valid_x, valid_y):
    mlflow.start_run(experiment_id=mlflow.get_experiment_by_name('lotto').experiment_id)

    cfg = get_config()

    learning_rate = params["learning_rate"]
    window_size = params["window_size"]
    batch_size = params["batch_size"]
    hidden_dim = params["hidden_dim"]
    early_stop_threshold = params["early_stop_threshold"]
    
    mlflow.log_params(params)

    train_x = train_x.clone()[:, -cfg["input_dim"]*window_size:]
    valid_x = valid_x.clone()[:, -cfg["input_dim"]*window_size:]

    m = Model(cfg["input_dim"] * window_size, hidden_dim, cfg["output_dim"], learning_rate)
    early_stop_cnt, min_valid_loss = 0, np.inf
    for e in range(cfg['max_epoch']):
        batch_perm = np.random.permutation(len(train_x))
        train_x, train_y = map(lambda x: x[batch_perm], [train_x, train_y])
        train_loss_list = []
        for i in range(int(np.ceil(len(train_x)/batch_size))):
            batch_x, batch_y = train_x[i*batch_size: (i+1)*batch_size], train_y[i*batch_size: (i+1)*batch_size]
            train_loss_list.append(m.optimize(batch_x, batch_y))

        train_loss = np.mean(train_loss_list)
        valid_loss = m.get_loss(valid_x, valid_y).item()
        train_hit = m.evaluate(train_x, train_y).item()
        valid_hit = m.evaluate(valid_x, valid_y).item()

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("valid_loss", valid_loss)
        mlflow.log_metric("train_hit", train_hit)
        mlflow.log_metric("valid_hit", valid_hit)

        if cfg["debug"]:
            print(f"{e} Epoch - train loss: {train_loss:.4f} / valid_loss: {valid_loss:.4f} " +\
                            f"/ train_hit: {train_hit:.4f} / valid_hit: {valid_hit:.4f}")
        
        if min_valid_loss < valid_loss:
            early_stop_cnt += 1
        else:
            early_stop_cnt = 0
            min_valid_loss = valid_loss
            best_model = copy.deepcopy(m.net)

        if early_stop_cnt == early_stop_threshold:
            break
    metrics = {'train_loss': train_loss, 'valid_loss': valid_loss, 'train_hit': train_hit, 'valid_hit': valid_hit}
    
    m.net = best_model
    mlflow.end_run()
    return {'loss': min_valid_loss, 'params': params, 'status': STATUS_OK, "metrics": metrics, 'model': m}


def train():
    cfg = get_config()
    mlflow.set_experiment("lotto")
    
    # load dataset
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_dataset(cfg)

    # hyper parameters tuning
    space = {
        'learning_rate': hp.quniform('learning_rate', 1e-4, 5e-4, 1e-4),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'hidden_dim': hp.choice('hidden_dim', [32, 64, 128]),
        'early_stop_threshold': hp.choice('early_stop_threshold', [3, 5, 7]),
        'window_size': hp.choice('window_size', [3, 5, 7]),
    }
    trials = Trials()
    objective = lambda params: fit(params, train_x, train_y, valid_x, valid_y)
    fmin(objective, space=space, algo=tpe.suggest, max_evals=cfg['max_evals'], trials=trials)
    
    best_result = trials.best_trial['result']
    best_params = best_result["params"]
    best_metrics = best_result["metrics"]
    best_model = best_result["model"]

    mlflow.log_params(best_params)
    mlflow.log_metrics(best_metrics)
    last_week_inference = best_model.inference(test_x[-cfg["input_dim"]*best_params["window_size"]:], test_y)
    mlflow.log_metric("last_week_hit", last_week_inference["Hits"])
    print("Best params:", best_params)
    print("Best metrics:", best_metrics)
    print("last week inference:", last_week_inference)

    # register model
    model_info = mlflow.pytorch.log_model(best_model.net, cfg['model_path'])
    mlflow.register_model(model_info.model_uri, "lotto")
    
if __name__=="__main__":
    train()

