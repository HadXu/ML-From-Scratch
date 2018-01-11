import numpy as np
import scipy.special as ss
import scipy.stats as st
import matplotlib.pyplot as plt


def circle(x, y):
    return (x+2)**2 + (y-2)**2/100 - 3**2


mus = np.array([5, 5])
sigmas = np.array([[1, .9], [.9, 1]])
def pgauss(x, y):
    return st.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)

def HM(p,iter=100000):
	x,y = 0.,0.
	samples = np.zeros((iter,2))
	for i in range(iter):
		x_star, y_star = np.array([x, y]) + np.random.normal(size=2)
		if np.random.rand() < p(x_star,y_star)/p(x,y):
			x, y = x_star, y_star
		samples[i] = np.array([x,y])
	return samples

if __name__ == '__main__':
	samples = HM(pgauss, iter=10000)
	plt.scatter(samples[:,0],samples[:,1])
	plt.show()

