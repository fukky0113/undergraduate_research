from traindemo import Train
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.backend import tensorflow_backend 
    
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

t = Train()
t.train(epochs = 10000, batch_size = 8, sample_interval = 18)
