#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Tuning with the HParams Dashboard
# 
# https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams

# ## 0. Preparing the environment and data

# ### Import TensorFlow and the TensorBoard HParams plugin:

# In[ ]:


import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import sys
# ### Download the FashionMNIST dataset and scale it:

# In[ ]:


fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# ## 1. Experiment Setup and HParams experiment summary

# In[ ]:


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

params = []
for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
        params.append([num_units, dropout_rate, optimizer])

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )


# ## 2. Adapt TensorFlow runs to log hyperparameters and metrics

# In[ ]:


def train_test_model(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  ])
  model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )
  model.fit(x_train, y_train, epochs=1) # Run with 1 epoch to speed things up for demo purposes
  _, accuracy = model.evaluate(x_test, y_test)
  return accuracy
# ### For each run, log an hparams summary with the hyperparameters and final accuracy
# In[ ]:
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
# ## 3. Start runs and log them all under one parent directory
# In[ ]:
session_num = int(sys.argv[1])
hparams = {
      HP_NUM_UNITS: params[session_num][0],
      HP_DROPOUT: params[session_num][1],
      HP_OPTIMIZER: params[session_num][2],
}
run_name = "run-%d" % session_num
print('--- Starting trial: %s' % run_name)
print({h.name: hparams[h] for h in hparams})
run('logs/hparam_tuning/' + run_name, hparams)
