import numpy as np
import tensorflow as tf

def positional_embedding(pos, model_size):
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE

def get_posenc(length,dim):
    indices = np.linspace(start=0.0,stop=100.0, num=length, endpoint=True)

    pes = []
    for i in range(length):
        pes.append(positional_embedding(indices[i], dim))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)
    return pes
