import yaml
import os, pathlib

import torch
import torch.nn.functional as F

def get_config():
    root_path = pathlib.Path(__file__).parent.resolve()
    cfg_path = os.path.join(root_path, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg

class Network(torch.nn.Module):
    def __init__(self, i_dim, h_dim, o_dim):
        super(Network, self).__init__()
        self.l1 = torch.nn.Linear(i_dim, h_dim)
        self.l2 = torch.nn.Linear(h_dim, h_dim)
        self.l3 = torch.nn.Linear(h_dim, o_dim)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return torch.sigmoid(self.l3(x))

class Model:
    def __init__(self, i_dim, h_dim, o_dim, lr=3e-4):
        self.net = Network(i_dim, h_dim, o_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def get_loss(self, x, y):
        prob = self.net(x)
        loss = F.binary_cross_entropy(prob, y).mean()
        return loss
    
    def optimize(self, x, y):
        loss = self.get_loss(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        logit = self.net(x)
        # predict by weighed sampling
        weights = torch.clip(logit - logit.mean(axis=-1, keepdim=True), 0)
        pred_wins = torch.multinomial(weights, 6, replacement=False)

        pred = F.one_hot(pred_wins, num_classes=45).sum(axis=-2)
        return pred

    def evaluate(self, x, y):
        logit = self.net(x)
        # predict by argmax
        pred_wins = torch.sort(logit, descending=True).indices[..., :6]
        pred = F.one_hot(pred_wins, num_classes=45).sum(axis=-2)
        label = y > 0
        return (pred * label).sum(-1).float().mean()

    def inference(self, x, y=None):
        result = {}
        pred = self.predict(x)
        result["Pred"] = (pred.nonzero().squeeze() + 1).tolist()

        if y is not None:
            num_wins = ((y > 0).nonzero().squeeze() + 1).tolist()
            result["Wins"] = num_wins
            result["Hits"] = int(self.evaluate(x, y).item())

        return result
    
    def save(self, path):
        print(f"... Save Model to {path}/model.pt ...")
        torch.save({
            "net" : self.net.state_dict(),
        }, path+f'/model.pt')

    def load(self, path):
        print(f"... Load Model to {path}/model.pt ...")
        checkpoint = torch.load(path+f'/model.pt')
        self.net.load_state_dict(checkpoint["net"])
