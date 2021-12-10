import gym
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import mixture
import time
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

MAX_LOG_PROB = 2
MIN_LOG_PROB = -20


class GMM:

    def __init__(self, list_state_type, n_components=7, with_plot=False, goal_position=None):
        # self.model = mixture.GaussianMixture(n_components=n_components)
        self.model = mixture.GaussianMixture(n_components=n_components,
                                             covariance_type='diag',
                                             reg_covar=1e-5,
                                             warm_start=True)
        self.trained = False
        self.goal_position=goal_position
        self.viewer = None

        self.n_components = n_components

        if with_plot:
            self.state_type = list_state_type[0]  # Get observation
            # for plotting
            xmax = self.state_type.high[0]
            xmin = self.state_type.low[0]
            xstep = (xmax - xmin) / 200.0

            ymax = self.state_type.high[1]
            ymin = self.state_type.low[1]
            ystep = (ymax - ymin) / 200.0
            self.x, self.y = np.mgrid[xmin:xmax:xstep, ymin:ymax:ystep]
            self.pos = np.dstack((self.x, self.y)).reshape(-1, 2)

    def update(self, states):
        self.model.fit(states)
        self.trained = True
        # print(f'Gaussian Mixture Model updated')

    def estimate(self, o):
        if self.trained:
            # score_samples() returns the log-likelihood of the samples
            if len(o.shape) == 1:
                o = o.reshape(1, -1)
            z = np.exp(self.log_prob(o))
        else:
            # fit some data first
            z = None
        return z

    def log_prob(self, o):
        if self.trained:
            clipped_log_prob = np.clip(self.model.score_samples(o), MIN_LOG_PROB, MAX_LOG_PROB)
            return clipped_log_prob
        else:
            return None

    def disp_info(self, plot3d=False):
        if plot3d:
            self.plot_3d_pdf()

    def plot_3d_pdf(self):
        z = self.estimate(self.pos)
        pdf_estimation = np.reshape(z, self.x.shape)/self.n_components
        matplotlib.rcParams.update({'font.size': 13})
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(self.x, self.y, pdf_estimation, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # Customize the x axis.
        ax.set_xlim(self.state_type.low[0], self.state_type.high[0])
        ax.xaxis.set_major_locator(LinearLocator(7))
        # A StrMethodFormatter is used automatically
        ax.xaxis.set_major_formatter('{x:.01f}')

        # Customize the y axis.
        ax.set_ylim(self.state_type.low[1], self.state_type.high[1])
        ax.yaxis.set_major_locator(LinearLocator(7))
        # A StrMethodFormatter is used automatically
        ax.yaxis.set_major_formatter('{x:.02f}')

        ax.view_init(55, 40)
        fig.tight_layout()
        fig.show()


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')

    x, y = np.mgrid[-10:10:.1, -10:10:.1]
    pos = np.dstack((x, y))

    rv_1 = multivariate_normal([5, -0.5], [[5, 0], [0, 5]])
    rv_2 = multivariate_normal([-2, 4], [[3, 0], [0, 3]])

    mult = 4.0
    alpha = mult / (mult + 1.0)
    beta = 1.0 / (mult + 1.0)
    pdf_true = (alpha * rv_1.pdf(pos) + beta * rv_2.pdf(pos))

    pos_samples = np.concatenate((rv_1.rvs(8), rv_2.rvs(2)), axis=0)
    gmm_func = GMM([env.observation_space])

    num_iters = 300
    all_mse = []

    for i in range(num_iters):
        pos_samples = np.concatenate((pos_samples, rv_1.rvs(80), rv_2.rvs(20)), axis=0)

        start = time.time()

        print(f'#########################################')
        print(f'Timestep : {i}')

        gmm_func.update(pos_samples)
        print(f'Update time : {time.time() - start}')
        start = time.time()
        set = pos.reshape(-1, 2)
        pdf_estimated = gmm_func.estimate(set)
        print(f'Time to estimate {set.shape[0]} values: {time.time() - start}')
        pdf_estimated = pdf_estimated.reshape(pdf_true.shape)

        mse = (np.square(pdf_estimated - pdf_true)).mean()
        all_mse.append(mse)
        print(f'MSE : {mse}')
        # plt.plot(all_mse)
        # plt.title('Mean Square Error')
        # plt.show()

        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title('Ground Truth Multi-variate PDF')
        surf = ax.plot_surface(x, y, pdf_true, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.view_init(30, 50)
        fig.colorbar(surf, shrink=0.5, aspect=10)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title('Estimated PDF by GMM')
        surf = ax.plot_surface(x, y, pdf_estimated, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.view_init(30, 50)
        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.show()
