#!/usr/bin/env python
# coding: utf-8

# # Physicality
# Computes a figure 
# between 0 and 1 that indicates the physicality of event samples. 
# The figure is 
# $$r_{\text{Phys}} = \frac{1}{N} \sum_i^N w_i$$
# where $w_i$ is an event weight that is given by 
# $$ w_i = \begin{cases}
#     1 - \frac{1}{2} \frac{m_{l1}}{E_{l1}} - \frac{1}{2} \frac{m_{l2}}{E_{l2}} & \text{if event passed selection cuts}\\
#     0,              & \text{otherwise}
# \end{cases}
# $$
# 
# # Density
# We make use of the following measures that quantify a distance between two probability distributions p(x) and q(x). For us, these distributions are represented by histograms with $n$ entries $p_i$ and $q_i$.
# 
# ### $\chi^2$ test
# The $\chi^2$ is a measurement of distance between histograms defined by
# $$ \chi^2 = \sum_i^n \frac{(p_i - q_i)^2}{p_i + q_i}$$
# 
# Ref: 0804.0380
# 
# ### Wasserstein Distance
# 
# ### Jensen-Shannon divergence
# The Jensen-Shannon divergence is defined as 
# $$ \text{JS}(p,q) = \frac{1}{2}\text{KL}(p|m) + \frac{1}{2}\text{KL}(q|m) $$
# where $m(x) = p(x) + q(x)$ and 
# $$ \text{KL}(p|q) = \int dx p(x) \log\left(\frac{p(x)}{q(x)}\right) \approx \sum_i^n p_i \log\left(\frac{p_i}{q_i}\right)$$
# 
# ## A note on histogram normalization
# For a histogram to represent a probability distribution, it has to integrate to $1$. 
# Thus, when the $x$-axis has a dimension, the $y$-axis has to have $1/d$. 
# Usually one normalizes a histogram by dividing by the sum of bin entries and by the length of the histogram domain. 
# However, the distance measures then become dimensionfull, making it hard to compare them. 
# We therefore only normalize by dividing by the sum of bin entries.
# ___



import numpy as np
import pandas as pd
import math as m
import sys
import scipy.stats as stats
from scipy.spatial import distance
from matplotlib import pyplot as plt
import os


# # Some Settings

# ## Number of events to check physicality of

nEventsTestPhys = 1000

# ## Number of bins used in 1d histograms

nBins = 25

# ## Number of bins used in 2d phi histograms

nBins2d = 400

# # Selection cuts definitions

ptMinJet = 20000
ptMaxJet = 7000000

ptMinLep = 10000
ptMaxLep = 7000000

EminJet = ptMinJet
EmaxJet = ptMaxJet

EminLep = ptMinLep
EmaxLep = ptMaxLep

METmin = 100
METmax = 7000000

etaMinJet = -5
etaMaxJet = 5

etaMinLep = -2.5
etaMaxLep = 2.5

phiMin = -m.pi
phiMax = m.pi

massMin = 10000
massMax = 1e7

histNames = ['MET', 'phi_MET',
             'Ej1', 'ptj1', 'etaj1', 'phij1',
             'Ej2', 'ptj2', 'etaj2', 'phij2',
             'Ej3', 'ptj3', 'etaj3', 'phij3',
             'Ej4', 'ptj4', 'etaj4', 'phij4',
             'El1', 'ptl1', 'etal1', 'phil1',
             'm2j', 'm3j', 'm4j', 'm4j1l', 'm4j2l']

minData = [METmin, phiMin, 
           EminJet, ptMinJet, etaMinJet, phiMin, 
           EminJet, ptMinJet, etaMinJet, phiMin,
           EminJet, ptMinJet, etaMinJet, phiMin,
           EminJet, ptMinJet, etaMinJet, phiMin,
           EminLep, ptMinLep, etaMinLep, phiMin,
           EminLep, ptMinLep, etaMinLep, phiMin]

maxData = [METmax, phiMax,
           EmaxJet, ptMaxJet, etaMaxJet, phiMax,
           EmaxJet, ptMaxJet, etaMaxJet, phiMax,
           EmaxJet, ptMaxJet, etaMaxJet, phiMax,
           EmaxJet, ptMaxJet, etaMaxJet, phiMax,
           EmaxLep, ptMaxLep, etaMaxLep, phiMax,
           EmaxLep, ptMaxLep, etaMaxLep, phiMax]

isLogSpace = [True, False,
              True, True, False, False,
              True, True, False, False,
              True, True, False, False,
              True, True, False, False,
              True, True, False, False,
              True, True, False, False]


# 
# ___
# # Load events
# 
# ## Monte Carlo data



fileNameMC = 'ttb.csv'
dfMC = pd.read_csv(fileNameMC, delimiter = ' ', header=None, index_col=False)
# Required for trailing delimiter in data file
dfMC = dfMC.drop(dfMC.columns[[26]], axis=1)
dMC = dfMC.to_numpy()
nEventsMC = len(dMC)
dMC = dMC.transpose()
nParmsMC = len(dMC)

f = open('result.txt', 'w+')
# ## Network data

dir = 'plots/'
f.write(dir + '\n')
os.mkdir(dir)
fileNameNN = 'B-VAE_events.csv'
dfNN = pd.read_csv(fileNameNN, delimiter = ' ', header=None, index_col=False)
# Required for trailing delimiter in data file
# dfNN = dfNN.drop(dfNN.columns[[26]], axis=1)
dNN = dfNN.to_numpy()
nEventsNN = len(dNN)
dNN = dNN.transpose()
nParmsNN = len(dNN)


# ## Check if data is same format


if (nParmsMC != nParmsNN):
	sys.exit("Data not in the same format")
else:
	nParms = nParmsMC


# ---
# # Do Physicality checks on NN Data


rPhysCuts = 0
rPhysCutsAndOrdering = 0
rPhysCutsAndMass = 0
rPhysAll = 0
for i in range(nEventsTestPhys):
	passedCuts = True
	passedOrdering = True
	passedMass = True
	weight = 1
	# Check cut regions
	for j in range(nParms-4):
		if dNN[j][i] < minData[j] or dNN[j][i] > maxData[j]:
			passedCuts = False
				
	# Check if all energies are larger than pt
	for j in range(5):
		if (dNN[2+4*j][i] < dNN[3+4*j][i]):
			passedMass = False
			
	# Check positivity of jet mass
	for j in range(0,4):
		m2 = dNN[2+4*j][i]**2 - (dNN[3+4*j][i]*np.cosh(dNN[4+4*j][i]))**2
		if m2 < 0:
			passedMass = False
		
	# Compute lepton masses
	for j in range(4,5):
		ml = np.sqrt(abs(dNN[2+4*j][i]**2 - (dNN[3+4*j][i]*np.cosh(dNN[4+4*j][i]))**2))
		if dNN[2+4*j][i] > 0:
			weight -= 0.5*ml/dNN[2+4*j][i]
		
	# Check pt ordering in jets
	if (dNN[3][i] < dNN[7][i] or dNN[7][i] < dNN[11][i] or dNN[11][i] < dNN[15][i]):
		passedOrdering = False
	
	# Check pt ordering in leptons
	#if (dNN[19][i] < dNN[23][i]):
		#passed = False
	
	if passedCuts:
		rPhysCuts += 1
		if passedOrdering:
			rPhysCutsAndOrdering += 1
		if passedMass:
			rPhysCutsAndMass += weight
	if passedCuts and passedOrdering and passedMass:
		rPhysAll += weight
		
rPhysCuts /= nEventsTestPhys
rPhysCutsAndOrdering /= nEventsTestPhys
rPhysCutsAndMass /= nEventsTestPhys
rPhysAll /= nEventsTestPhys

##print("Physicality Tests")
##print("=======================================================")
##print(format("Fraction of events that pass cuts", '46s'), format(rPhysCuts, 'f'))
##print(format("Fraction of events that pass cuts and ordering", '46s'), format(rPhysCutsAndOrdering, 'f'))
##print(format("Fraction of events that pass cuts and mass", '46s'), format(rPhysCutsAndMass, 'f'))
##print(format("Fraction of events that pass everything", '46s'), format(rPhysAll, 'f'))
##print("\n")


# # 1D Histograms of data parameters


histsMC = []
histsNN = []
for i in range(nParms-4):
	fileNameSave = dir + histNames[i] + '.png'
	weightsMC = np.ones_like(dMC[i])/float(len(dMC[i]))
	weightsNN = np.ones_like(dNN[i])/float(len(dNN[i]))
	if (isLogSpace[i]):
		histMC, binsMC, _ = plt.hist(dMC[i], bins = np.logspace(np.log10(minData[i]), np.log10(maxData[i]), nBins+1), weights = weightsMC)
		histNN, binsNN, _ = plt.hist(dNN[i], bins = np.logspace(np.log10(minData[i]), np.log10(maxData[i]), nBins+1), weights = weightsNN, histtype='step')
		
		plt.gca().set_xscale("log")
		plt.gca().set_yscale("log")
		plt.savefig(fileNameSave)
		#plt.show()   
		plt.clf()

		histsMC.append(histMC)
		histsNN.append(histNN)
	else:
		histMC, binsMC, _ = plt.hist(dMC[i], bins=nBins+1, range=(minData[i], maxData[i]), weights = weightsMC)
		histNN, binsNN, _ = plt.hist(dNN[i], bins=nBins+1, range=(minData[i], maxData[i]), weights = weightsNN, histtype='step')
		
		plt.savefig(fileNameSave)
		#plt.show()
		plt.clf()
		
		histsMC.append(histMC)
		histsNN.append(histNN)


# # 1D Histograms of combined parameters


# Compute cumulative masses
EMC = np.zeros(nEventsMC)
pxMC = np.zeros(nEventsMC)
pyMC = np.zeros(nEventsMC)
pzMC = np.zeros(nEventsMC)

ENN = np.zeros(nEventsNN)
pxNN = np.zeros(nEventsNN)
pyNN = np.zeros(nEventsNN)
pzNN = np.zeros(nEventsNN)
for j in range(6):
	EMC += dMC[2+4*j]
	pxMC += dMC[3+4*j]*np.cos(dMC[5+4*j])
	pyMC += dMC[3+4*j]*np.sin(dMC[5+4*j])
	pzMC += dMC[3+4*j]*np.sinh(dMC[4+4*j])
	
	ENN += dNN[2+4*j]
	pxNN += dNN[3+4*j]*np.cos(dNN[5+4*j])
	pyNN += dNN[3+4*j]*np.sin(dNN[5+4*j])
	pzNN += dNN[3+4*j]*np.sinh(dNN[4+4*j])
	
	if j > 0:
		m2MC = EMC**2 - pxMC**2 - pyMC**2 - pzMC**2
		m2NN = ENN**2 - pxNN**2 - pyNN**2 - pzNN**2
		
		for k in range(len(m2MC)):
			if m2MC[k] < 0: 
				m2MC[k] = 0
			else:
				m2MC[k] = np.sqrt(m2MC[k])
				
		for k in range(len(m2NN)):
			if m2NN[k] < 0: 
				m2NN[k] = 0
			else:
				m2NN[k] = np.sqrt(m2NN[k])
						
		weightsMC = np.ones_like(m2MC)/float(len(m2MC))
		weightsNN = np.ones_like(m2NN)/float(len(m2NN))

		histMC, binsMC, _ = plt.hist(m2MC, bins = np.logspace(np.log10(massMin), np.log10(massMax), nBins+1), weights = weightsMC)   
		histNN, binsNN, _ = plt.hist(m2NN, bins = np.logspace(np.log10(massMin), np.log10(massMax), nBins+1), weights = weightsNN, histtype='step')   
		plt.gca().set_xscale("log")
		plt.gca().set_yscale("log")
		
		# Save to file
		fileNameSave = dir + histNames[j+nParms-5] + '.png'
		plt.savefig(fileNameSave)
		
		#plt.show()
		plt.clf()       
		
		histsMC.append(histMC/histMC.sum()) 
		histsNN.append(histNN/histNN.sum()) 


# # 2D Hist of leading phi

# In[55]:


weightsPhiMC = np.ones_like(dMC[5])/float(len(dMC[5]))
phiHistMC, xbinsPhiHistMC, ybinsPhiHistMC, _ = plt.hist2d(dMC[5], dMC[9], nBins2d, weights = weightsPhiMC)
plt.clf()


# In[56]:


weightsPhiNN = np.ones_like(dNN[5])/float(len(dNN[5]))
phiHistNN, xbinsPhiHistNN, ybinsPhiHistNN, _ = plt.hist2d(dNN[5], dNN[9], nBins2d, weights = weightsPhiNN)
plt.clf()


weightsPhiMC100k = np.ones_like(dMC[5][:100000])/float(len(dMC[5][:100000]))
phiHistMC100k, xbinsPhiHistMC100k, ybinsPhiHistMC100k, _ = plt.hist2d(dMC[5][:100000], dMC[9][:100000], nBins2d, weights = weightsPhiMC100k)
plt.clf()




phiHistFlat = np.ones_like(phiHistMC)/float(nBins2d**2)


# # Check if we found the same number of histograms



if (len(histsMC) != len(histsMC)):
	sys.exit("Didn't get the same number of histograms")


# # Chi Square definition


def chisq(f_obs, f_exp):
	terms = np.zeros(f_obs.shape)
	nbins = terms.shape[0]
	for i in range(nbins): 
		if f_exp[i] == 0. and f_obs[i] == 0:
			terms[i] = 0
		else:
			terms[i] = (f_obs[i] - f_exp[i])**2 / (f_obs[i] + f_exp[i])

	stat = terms.sum()  

	return stat


# # Density comparison 1D

chisqMCNN = np.empty(len(histsMC))
wsMCNN = np.empty(len(histsMC))
jsMCNN = np.empty(len(histsMC))
total = np.empty(len(histsMC))

print("1D Density Comparison")
print("=======================================================")
print("Name    Chi-Square   Earth-Mover  Jensen-Shannon")
print("-------------------------------------------------")
for i in range(len(histsMC)):
	chisqMCNN[i] = format(chisq(histsMC[i], histsNN[i]), 'e')
	wsMCNN[i]    = format(stats.wasserstein_distance(histsMC[i], histsNN[i]), 'e')
	jsMCNN[i]    = format(distance.jensenshannon(histsMC[i], histsNN[i]), 'e')
	total[i]     = format(-np.log(chisqMCNN[i])-np.log(wsMCNN[i])-np.log(jsMCNN[i]), 'e')
	print(format(histNames[i], '7s'), chisqMCNN, wsMCNN, jsMCNN, total)
print("\n")
tot=0
for i in range(len(histsMC)):
	tot += total[i]
print(str(tot) + ' is the total 1d hist distance')
f.write(str(tot) + '\n')
# # Denstiny comparison 2D


nHolesMC = 0
nTotalMC = 0
for i in phiHistMC:
	for j in i:
		nTotalMC += 1
		if j==0:
			nHolesMC += 1

nHolesMC100k = 0
nTotalMC100k = 0
for i in phiHistMC100k:
	for j in i:
		nTotalMC100k +=1
		if j==0:
			nHolesMC100k +=1 
			
nHolesNN = 0
nTotalNN = 0
for i in phiHistNN:
	for j in i:
		nTotalNN += 1
		if j==0:
			nHolesNN += 1

rDen2D = 0
rDen2D += -np.log(chisq(phiHistMC.flatten(), phiHistNN.flatten()))/2
rDen2D += -np.log(distance.jensenshannon(phiHistMC.flatten(), phiHistNN.flatten()))/2

MC1string  = format('Monte Carlo 1.2M', '22s')
MC2string  = format('Monte Carlo 100k', '22s')
NN1string  = format('Neural Network 1.2M', '22s')
flatstring = format('Flat', '22s')
emptystring = format('', '22s')

fracstring  = format('Fraction of holes', '18s')
chisqstring = format('Chi-Square', '18s')
JSstring    = format('Jensen-Shannon', '18s')
emstring    = format('Earth-Mover', '18s')

print("2D Holes test")
print("==========================================================================")
print("Statistic          Dataset 1              Dataset 2              Result")
print("--------------------------------------------------------------------------")

print(fracstring, MC1string, emptystring, format(nHolesMC/nTotalMC, 'f'))
print(fracstring, MC2string, emptystring, format(nHolesMC100k/nTotalMC100k, 'f'))
print(fracstring, NN1string, emptystring, format(nHolesNN/nTotalNN, 'f'))

print("\n")

print(chisqstring, MC1string, flatstring, format(chisq(phiHistFlat.flatten(), phiHistMC.flatten()), 'f'))
print(chisqstring, MC2string, flatstring, format(chisq(phiHistFlat.flatten(), phiHistMC100k.flatten()), 'f'))
print(chisqstring, NN1string, flatstring, format(chisq(phiHistFlat.flatten(), phiHistNN.flatten()), 'f'))

print('\n')

print(JSstring, MC1string, flatstring, format(distance.jensenshannon(phiHistFlat.flatten(), phiHistMC.flatten()), 'f'))
print(JSstring, MC2string, flatstring, format(distance.jensenshannon(phiHistFlat.flatten(), phiHistMC100k.flatten()), 'f'))
print(JSstring, NN1string, flatstring, format(distance.jensenshannon(phiHistFlat.flatten(), phiHistNN.flatten()), 'f'))

print("\n")

print(chisqstring, MC1string, NN1string, format(chisq(phiHistMC.flatten(), phiHistNN.flatten()), 'f'))
print(JSstring, MC1string, NN1string, format(distance.jensenshannon(phiHistMC.flatten(), phiHistNN.flatten()), 'f'))
tot2d = (chisq(phiHistMC.flatten(), phiHistNN.flatten())+distance.jensenshannon(phiHistMC.flatten(), phiHistNN.flatten()))*np.abs(nHolesMC/nTotalMC-nHolesNN/nTotalNN)
print(tot2d)
f.write(str(tot2d) + '\n')
f.close()




