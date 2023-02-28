import argparse

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def eda(ani=False):
    df = pd.read_csv("data/raw_data.csv")
    if ani:
        fig, ax = plt.subplots()
        fig.set_size_inches(w=16,h=8)
        def update(frame):
            ax.cla()
            df.iloc[:frame].plot.hist(bins=45, ax=ax)
            ax.set_title(f'Histogram of Data in DataFrame (iteration {frame})')
            return ax,
        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(df), interval=1, blit=False)
    else:
        df.plot.hist(bins=45, figsize=(16, 8))
    
    # Show the animation
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ani", action='store_true')
    args = parser.parse_args()

    eda(args.ani)

