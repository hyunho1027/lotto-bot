import argparse

from collect import collect
from preprocessing import preprocess
from train import train
from inference import inference

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id")
    parser.add_argument("--pw")
    args = parser.parse_args()

    preprocess()
    train()
    inference(args.id, args.pw)
