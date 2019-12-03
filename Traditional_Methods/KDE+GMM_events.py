### Creating a generative model with kernel density estimation and gaussian mixture models for ttbar events ###
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn import mixture

#Reading in events
filename = 'ttb.csv'
ttb_df = pd.read_csv(filename, sep=' ', header=None)
data = ttb_df.values
data = data[:,0:26]		
#preprocessing events
max = np.empty(26)
for i in range(0,data.shape[1]):
	max[i] = np.max(np.abs(data[:,i]))
	if np.abs(max[i]) > 0: 
		data[:,i] = data[:,i]/max[i]
	else:
		pass
event_size = data.shape[1]
original_dim = event_size
data = np.reshape(data, [-1, original_dim])
data = data.astype('float32')
data = data[:100000]

# data_small = data[:10000]
# #KDE
# # use grid search cross-validation to optimize the bandwidth for 10k events
# params = {'bandwidth': np.logspace(-1, 1, 20)}
# grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
# grid.fit(data_small)

# print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
# h_opt=grid.best_estimator_.bandwidth

# #Do KDE with h_opt for all events
# # use the best estimator to compute the kernel density estimate
# kde = KernelDensity(bandwidth=h_opt, kernel='gaussian')
# kde.fit(data)

# # sample 1 200 000 new events from the generative model
# new_data = kde.sample(1200000, random_state=0)
# for i in range(0,data.shape[1]):
	# new_data[:,i] = new_data[:,i]*max[i]
# np.savetxt('ttb_KDE_events.csv', new_data, delimiter=' ')

#gaussian mixture model
gmm = mixture.GaussianMixture(n_components = 1000, covariance_type = 'full', max_iter = 500, verbose = 1)
gmm.fit(data)
gmm_data, y_data = gmm.sample(1200000)
print(np.shape(gmm_data))
for i in range(0, 26):
	gmm_data[:,i] = gmm_data[:,i]*max[i]
np.savetxt('ttb_GMM_events_500iter_1kcomp.csv', gmm_data, delimiter=' ')