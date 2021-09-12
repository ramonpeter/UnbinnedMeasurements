import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import time

from cflow import ConditionalFlow
from MoINN.modules.subnetworks import DenseSubNet

from utils import train_density_estimation, plot_loss, plot_tau_ratio

# import data
tau1_gen = np.reshape(np.load("../data/tau1s_Pythia_gen.npy"), (-1,1))
tau2_gen = np.reshape(np.load("../data/tau2s_Pythia_gen.npy"), (-1,1))

tau1_sim = np.reshape(np.load("../data/tau1s_Pythia_sim.npy"), (-1,1))
tau2_sim = np.reshape(np.load("../data/tau2s_Pythia_sim.npy"), (-1,1))

data_gen = tf.convert_to_tensor(np.concatenate([tau1_gen,tau2_gen], axis=-1), dtype=tf.float32)
data_sim = tf.convert_to_tensor(np.concatenate([tau1_sim,tau2_sim], axis=-1), dtype=tf.float32)

train_gen, test_gen = np.split(data_gen, 2)
train_sim, test_sim = np.split(data_sim, 2)

detector = tf.constant(test_sim, dtype=tf.float32)

#Now, for Herwig

# import data
tau1_gen_herwig = np.reshape(np.load("../data/tau1s_Herwig_gen.npy"), (-1,1))
tau2_gen_herwig = np.reshape(np.load("../data/tau2s_Herwig_gen.npy"), (-1,1))

tau1_sim_herwig = np.reshape(np.load("../data/tau1s_Herwig_sim.npy"), (-1,1))
tau2_sim_herwig = np.reshape(np.load("../data/tau2s_Herwig_sim.npy"), (-1,1))

data_gen_herwig = tf.convert_to_tensor(np.concatenate([tau1_gen_herwig,tau2_gen_herwig], axis=-1), dtype=tf.float32)
data_sim_herwig = tf.convert_to_tensor(np.concatenate([tau1_sim_herwig,tau2_sim_herwig], axis=-1), dtype=tf.float32)

train_gen_herwig, test_gen_herwig = np.split(data_gen_herwig, 2)
train_sim_herwig, test_sim_herwig = np.split(data_sim_herwig, 2)

# Get the flow
meta = {
        "units": 16,
        "layers": 4,
        "initializer": "glorot_uniform",
        "activation": "leakyrelu",
        }

cflow_herwig = ConditionalFlow(dims_in=[2], dims_c=[[2]], n_blocks=12, subnet_meta=meta, subnet_constructor=DenseSubNet)


# train the network
EPOCHS = 50
BATCH_SIZE = 1000
LR = 5e-3
DECAY_RATE=0.1
ITERS = len(train_gen_herwig)//BATCH_SIZE
DECAY_STEP=ITERS

#Prepare the tf.dataset
train_dataset_herwig = tf.data.Dataset.from_tensor_slices((train_gen_herwig, train_sim_herwig))
train_dataset_herwig = train_dataset_herwig.shuffle(buffer_size=500000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)
opt = tf.keras.optimizers.Adam(lr_schedule)

train_losses_herwig = []
#train_all = np.concatenate([train_gen, train_sim], axis=-1)
start_time = time.time()
for e in range(EPOCHS):
    
    batch_train_losses = []
    # Iterate over the batches of the dataset.
    for step, (batch_gen, batch_sim) in enumerate(train_dataset_herwig):
        batch_loss = train_density_estimation(cflow_herwig, opt, batch_gen, [batch_sim])
        batch_train_losses.append(batch_loss)

    train_loss = tf.reduce_mean(batch_train_losses)
    train_losses_herwig.append(train_loss)

    if (e + 1) % 1 == 0:
        # Print metrics
        print(
            "Epoch #{}: Loss: {}, Learning_Rate: {}".format(
                e + 1, train_losses_herwig[-1], opt._decayed_lr(tf.float32)
            )
        )
end_time = time.time()
print("--- Run time: %s hour ---" % ((end_time - start_time)/60/60))
print("--- Run time: %s mins ---" % ((end_time - start_time)/60))
print("--- Run time: %s secs ---" % ((end_time - start_time)))


# Make plots and sample
plot_loss(train_losses_herwig, name="Log-likelihood-Herwig", log_axis=False)

unfold_gen = cflow_herwig.sample(int(5e5),[detector])
plot_tau_ratio(test_gen, unfold_gen, detector, name="tau_ratio")

unfold_gen_herwig = {}
for i in range(10):
    unfold_gen_herwig[i] = cflow_herwig.sample(int(5e5),[detector])
unfold_herwig = np.stack([unfold_gen_herwig[i] for i in range(10)])

np.save("inn_herwig",unfold_herwig)