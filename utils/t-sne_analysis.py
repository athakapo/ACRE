import time

from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle

PATH = '../tensorboard/Swimmer-v2-acre-[2_256]-2022-06-17_14-44-54'

def main():
    #load dataset
    with open(f'{PATH}/all_states.pkl', 'rb') as f:
        all_states = pickle.load(f)

    #subsampling
    all_states = all_states[::20]

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(all_states)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    df = pd.DataFrame()
    df['x'] = tsne_results[:, 0]
    df['y'] = tsne_results[:, 1]
    #df['z'] = tsne_results[:, 2]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x='x', y='y',
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()

if __name__ == '__main__':
    main()