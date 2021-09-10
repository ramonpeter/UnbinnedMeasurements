import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

import omnifold as of
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# import data
tau1_gen = np.reshape(np.load("../data/tau1s_Pythia_gen.npy"), (-1,1))
tau2_gen = np.reshape(np.load("../data/tau2s_Pythia_gen.npy"), (-1,1))

tau1_sim = np.reshape(np.load("../data/tau1s_Pythia_sim.npy"), (-1,1))
tau2_sim = np.reshape(np.load("../data/tau2s_Pythia_sim.npy"), (-1,1))

tau1_gen_herwig = np.reshape(np.load("../data/tau1s_Herwig_gen.npy"), (-1,1))
tau2_gen_herwig = np.reshape(np.load("../data/tau2s_Herwig_gen.npy"), (-1,1))

tau1_sim_herwig = np.reshape(np.load("../data/tau1s_Herwig_sim.npy"), (-1,1))
tau2_sim_herwig = np.reshape(np.load("../data/tau2s_Herwig_sim.npy"), (-1,1))

train_gen, test_gen = np.split(np.concatenate([tau1_gen,tau2_gen], axis=-1), 2)
train_sim, test_sim = np.split(np.concatenate([tau1_sim,tau2_sim], axis=-1), 2)

train_gen_herwig, test_gen_herwig = np.split(np.concatenate([tau1_gen_herwig,tau2_gen_herwig], axis=-1), 2)
train_sim_herwig, test_sim_herwig = np.split(np.concatenate([tau1_sim_herwig,tau2_sim_herwig], axis=-1), 2)

#as a first test, let's call half of Pythia data and the other half MC.

inputs = Input((2, ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
model = Model(inputs=inputs, outputs=outputs)

theta_unknown_S = test_sim
theta_unknown_G = test_gen
theta0_G = train_gen
theta0 = np.stack([train_gen,train_sim], axis=1)
myweights = of.omnifold(theta0,theta_unknown_S,5,model,verbose=1)

fig = plt.figure()
_,_,_=plt.hist(theta0_G[:,1]/theta0_G[:,0],bins=np.linspace(0,1.2,20),color='blue',alpha=0.5,label="MC, true",density=True)
_,_,_=plt.hist(theta_unknown_G[:,1]/theta_unknown_G[:,0],bins=np.linspace(0,1.2,20),color='orange',alpha=0.5,label="Data, true",density=True)
_,_,_=plt.hist(theta0_G[:,1]/theta0_G[:,0],weights=myweights[-1, 0, :], bins=np.linspace(0,1.2,20),color='black',histtype="step",label="OmniFolded",lw="2",density=True)
plt.xlabel(r"$N$-subjettiness ratio $\tau_{21}$")
plt.ylabel("events")
plt.legend(frameon=False)
fig.savefig('result_omnifold.pdf',bbox_inches='tight')

np.save("weights_pythia",myweights)

#Now, for Herwig

inputs = Input((2, ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
model = Model(inputs=inputs, outputs=outputs)

theta_unknown_S = test_sim
theta_unknown_G = test_gen
theta0_G = train_gen_herwig
theta0 = np.stack([train_gen_herwig,train_sim_herwig], axis=1)
myweights = of.omnifold(theta0,theta_unknown_S,5,model,verbose=1)

fig = plt.figure()
_,_,_=plt.hist(theta0_G[:,1]/theta0_G[:,0],bins=np.linspace(0,1.2,20),color='blue',alpha=0.5,label="MC, true",density=True)
_,_,_=plt.hist(theta_unknown_G[:,1]/theta_unknown_G[:,0],bins=np.linspace(0,1.2,20),color='orange',alpha=0.5,label="Data, true",density=True)
_,_,_=plt.hist(theta0_G[:,1]/theta0_G[:,0],weights=myweights[-1, 0, :], bins=np.linspace(0,1.2,20),color='black',histtype="step",label="OmniFolded",lw="2",density=True)
plt.xlabel(r"$N$-subjettiness ratio $\tau_{21}$")
plt.ylabel("events")
plt.legend(frameon=False)
fig.savefig('result_omnifold_herwig.pdf',bbox_inches='tight')

np.save("weights_herwig",myweights)