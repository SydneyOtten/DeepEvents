import numpy as np
import pandas as pd

filename = 'ttb.csv'
ttb_df = pd.read_csv(filename, sep=' ', header=None)
data = ttb_df.values
data = data[:,0:26]		
			
max = np.empty(26)
for i in range(0,data.shape[1]):
	max[i] = np.max(np.abs(data[:,i]))
	if np.abs(max[i]) > 0: 
		data[:,i] = data[:,i]/max[i]
	else:
		pass

data = data[:100000]		

#events are max-normalized, i.e. they are restricted to [-1,1]

#smear the events
smeared_events = np.empty([1200000,26])
for i in range(0,100000):
	for j in range(0,12):
		for k in range(0,26):
			smeared_events[i*12+j,k] = data[i,k]*np.abs(np.random.normal(1,0.05))
for i in range(0,26):
	smeared_events[:,i] = smeared_events[:,i]*max[i]
np.savetxt('ttb_gaussian_smearing_5p.csv', smeared_events)
