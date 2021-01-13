import tensorflow as tf

class CountingAbstraction(tf.keras.Model):
    def __init__(self, model_size):
        super(Counting, self).__init__()
        
        self.counter_expander = tf.keras.layers.Dense(model_size,activation=tf.nn.softplus)
        self.counter_merger = tf.keras.layers.Dense(model_size,use_bias=False)
        
    def call(self, data):
        #cosine similarity
        counter = tf.math.l2_normalize(data, axis=-1)
        counter = tf.matmul(counter, counter, transpose_b=True)
        #relu to filter out bads (this could be improved upon)
        counter = tf.nn.relu(counter)

        #do the fixed-v abstraction
        fixed_v = get_posenc(data.shape[-2],data.shape[-1])
        v_output = tf.matmul(fixed_v,counter)

        #do the count itself
        counter = tf.reduce_sum(counter,axis=-1,keepdims=True)
        
        #combine them
        counter = tf.concat([counter,v_output],axis=-1)
        
        #do postprocessing of the count.
        counter = self.counter_expander(counter)
        #merge the data here.
        data = self.counter_merger(tf.concat([data,counter],axis=-1))
        return data

class Counting(tf.keras.Model):
    def __init__(self, model_size):
        super(Counting, self).__init__()
        
        self.counter_expander = tf.keras.layers.Dense(model_size,activation=tf.nn.softplus)
        self.counter_merger = tf.keras.layers.Dense(model_size,use_bias=False)
        
    def call(self, data):
        #cosine similarity
        counter = tf.math.l2_normalize(data, axis=-1)
        counter = tf.matmul(counter, counter, transpose_b=True)
        #relu to filter out bads (this could be improved upon)
        counter = tf.nn.relu(counter)
        #do the count itself
        counter = tf.reduce_sum(counter,axis=-1,keepdims=True)
        #do postprocessing of the count.
        counter = self.counter_expander(counter)
        #merge the data here.
        data = self.counter_merger(tf.concat([data,counter],axis=-1))
        return data
