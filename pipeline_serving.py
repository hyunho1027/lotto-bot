import argparse

from inference import inference

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id")
    parser.add_argument("--pw")
    args = parser.parse_args()

    inference(args.id, args.pw)
