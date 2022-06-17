from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    #load dataset
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array(['0', '1', '0', '0'])
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    print(tsne_result.shape)

    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
    fig, ax = plt.subplots(1)
    sns.regplot(x=tsne_result_df['tsne_1'], y=tsne_result_df['tsne_2'], fit_reg=False)
    lim = (tsne_result.min() - 5, tsne_result.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    fig.show()


if __name__ == '__main__':
    main()