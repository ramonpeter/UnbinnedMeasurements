import tensorflow as tf
import pandas as pd
import numpy as np
import sys

from cflow import ConditionalFlow
from MoINN.modules.subnetworks import DenseSubNet

from utils import train_density_estimation, plot_loss, shuffle, plot_tau_ratio

# import data
tau1_gen = np.reshape(np.load("../data/tau1s_Pythia_gen.npy"), (-1,1))
tau2_gen = np.reshape(np.load("../data/tau2s_Pythia_gen.npy"), (-1,1))

tau1_sim = np.reshape(np.load("../data/tau1s_Pythia_sim.npy"), (-1,1))
tau2_sim = np.reshape(np.load("../data/tau2s_Pythia_sim.npy"), (-1,1))

train_gen, test_gen = np.split(np.concatenate([tau1_gen,tau2_gen], axis=-1), 2)
train_sim, test_sim = np.split(np.concatenate([tau1_sim,tau2_sim], axis=-1), 2)

# Get the flow
meta = {
        "units": 20,
        "layers": 2,
        "initializer": "glorot_uniform",
        "activation": "leakyrelu",
        }

cflow = ConditionalFlow(dims_in=[2], dims_c=[[2]], n_blocks=4, subnet_meta=meta, subnet_constructor=DenseSubNet)


# train the network
EPOCHS = 200
BATCH_SIZE = 1000
LR = 1e-3
DECAY_RATE=0.1
ITERS = len(train_gen)//BATCH_SIZE
DECAY_STEP=ITERS


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)
opt = tf.keras.optimizers.Adam(lr_schedule)

train_losses = []
train_all = np.concatenate([train_gen, train_sim], axis=-1)
for e in range(EPOCHS):
    
    #train_gen, train_sim = shuffle(train_gen, train_sim)
    #np.random.shuffle(train_all)
    batch_train_losses = []
    for i in range(ITERS):
        #batch_gen = tf.constant(train_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :2], dtype=tf.float32)
        #batch_sim = tf.constant(train_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE, 2:], dtype=tf.float32)
        batch_gen = tf.constant(train_gen[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :], dtype=tf.float32)
        batch_sim = tf.constant(train_sim[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :], dtype=tf.float32)
        batch_loss = train_density_estimation(cflow, opt, batch_gen, [batch_sim])
        batch_train_losses.append(batch_loss)

    train_loss = tf.reduce_mean(batch_train_losses)
    train_losses.append(train_loss)

    if (e + 1) % 1 == 0:
        # Print metrics
        print(
            "Epoch #{}: Loss: {}, Learning_Rate: {}".format(
                e + 1, train_losses[-1], opt._decayed_lr(tf.float32)
            )
        )

plot_loss(train_losses, name="Log-likelihood", log_axis=False)

detector = tf.constant(test_sim, dtype=tf.float32)
unfold_gen = cflow.sample(int(5e5),[detector])

plot_tau_ratio(test_gen, unfold_gen)
