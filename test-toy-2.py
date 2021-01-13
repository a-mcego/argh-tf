import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
np.set_printoptions(precision=3,suppress=True)
import tensorflow as tf
import time
import enum
import matplotlib.pyplot as plt

VEC_AMOUNT = 32
MODEL_SIZE = 96
XF_LAYERS = 1
XF_HEADS = 1
LEARNING_RATE = 0.0005
ADAM_EPSILON = 1e-4
BATCH_SIZE = 32
USE_COUNTER = True
N_COLORS = 10
NORMALIZE_COLUMNS = False

print(f"VEC_AMOUNT={VEC_AMOUNT} ",end="")
print(f"MODEL_SIZE={MODEL_SIZE} ",end="")
print(f"XF_LAYERS={XF_LAYERS} ",end="")
print(f"XF_HEADS={XF_HEADS} ",end="")
print(f"LEARNING_RATE={LEARNING_RATE} ",end="")
print(f"ADAM_EPSILON={ADAM_EPSILON} ",end="")
print(f"BATCH_SIZE={BATCH_SIZE} ",end="")
print(f"USE_COUNTER={USE_COUNTER} ",end="")
print(f"N_COLORS={N_COLORS} ",end="")
print(f"NORMALIZE_COLUMNS={NORMALIZE_COLUMNS} ",end="")
print()

steps = 0

def positional_embedding(pos, model_size):
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE

def get_posenc(length):

    indices = np.linspace(start=0.0,stop=100.0, num=length, endpoint=True)

    pes = []
    for i in range(length):
        pes.append(positional_embedding(indices[i], MODEL_SIZE))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)
    return pes
    
#max_length = 256
#pes = get_posenc(max_length)


def reduce(matrix, axis):
    mean = tf.math.reduce_mean(matrix, axis=axis, keepdims=True)
    stddev = tf.math.reduce_std(matrix, axis=axis, keepdims=True)
    return (matrix - mean)/stddev

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

    
class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.model_size = model_size
        self.head_size = model_size // h
        self.n_heads = h

        self.w_q = tf.keras.layers.Dense(model_size)
        self.w_kv = tf.keras.layers.Dense(model_size*2)

        self.wo = tf.keras.layers.Dense(model_size)
        self.normalizer = 1.0/tf.math.sqrt(tf.dtypes.cast(self.head_size, tf.float32))

    def call(self, query, value):
        out_kv = self.w_kv(value)
        out_q = self.w_q(query)

        if NORMALIZE_COLUMNS:
            out_kv = reduce(out_kv,axis=-2)
            out_q = reduce(out_q,axis=-2)
        
        out_kv = tf.split(out_kv, self.n_heads*2, axis=-1)
        out_q = tf.split(out_q, self.n_heads, axis=-1)
        
        heads = []
        for i in range(self.n_heads):
            wqd = out_q[i]
            wkd = out_kv[i]

            score = tf.matmul(wqd, wkd, transpose_b=True)*self.normalizer
            alignment = tf.nn.softmax(score, axis=-1)

            wvd = out_kv[i+self.n_heads]
            head = tf.matmul(alignment, wvd)
            heads.append(head)

        heads = tf.concat(heads, axis=-1)
        heads = tf.reshape(heads, [BATCH_SIZE,-1,MODEL_SIZE])
        heads = self.wo(heads)
        
        return heads

#no positional encoding, no classification layers, no frills.
#basic transformer is all you need :D
#
#this version of transformer attends to the source sequence *first*
#only then to the target sequence
class Transformer(tf.keras.Model):
    def __init__(self, model_size, num_layers, heads, use_causal_mask):
        super(Transformer, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.heads = heads

        self.src_att = [MultiHeadAttention(model_size, heads) for _ in range(num_layers)]
        self.src_att_norm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

        self.tgt_att = [MultiHeadAttention(model_size, heads) for _ in range(num_layers)]
        self.tgt_att_norm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]
       
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation=tf.math.softplus) for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

        self.use_causal_mask = use_causal_mask
        
        if USE_COUNTER:
            self.counting = [Counting(model_size) for _ in range(num_layers)]
        
    def call(self, target_sequence, source_sequence=None):
    
        if self.use_causal_mask:
            mask_size = target_sequence.shape[1]
            look_left_only_mask = tf.linalg.band_part(tf.ones((mask_size,mask_size)), -1, 0)
        else:
            look_left_only_mask = None
            
        src_att_in = target_sequence
        for i in range(self.num_layers):

            #src_att_in has the data now
            if USE_COUNTER:
                src_att_in = self.counting[i](src_att_in)
            
            if source_sequence is not None:
                src_att_out = self.src_att[i](src_att_in, source_sequence)
                
                src_att_out = src_att_in + src_att_out
                src_att_out = self.src_att_norm[i](src_att_out)
            else:
                src_att_out = src_att_in
                            
            tgt_att_in = src_att_out
            #tgt_att_in has the data now
            

            tgt_att_out = self.tgt_att[i](tgt_att_in, tgt_att_in)
            tgt_att_out = tgt_att_in + tgt_att_out
            tgt_att_out = self.tgt_att_norm[i](tgt_att_out)
                        
            ffn_in = tgt_att_out
            #ffn_in has the data now 
            ffn_mid = self.dense_1[i](ffn_in)
            ffn_out = self.dense_2[i](ffn_mid)

            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            src_att_in = ffn_out
            #src_att_in has the data now

        return src_att_in
    
class TaskSolver(tf.keras.Model):
    def __init__(self):
        super(TaskSolver,self).__init__()

        self.xformer_A = Transformer(model_size=MODEL_SIZE, num_layers=XF_LAYERS, heads=XF_HEADS, use_causal_mask=False)
        
        self.logit_outputs = tf.keras.layers.Dense(N_COLORS)
        
    def call(self, n_input, posenc):
        n_output = n_input
        
        n_output = n_output+posenc[:n_output.shape[1], :]
        n_output = self.xformer_A(n_output)
        
        n_output = self.logit_outputs(n_output)
        return n_output
        
tasksolver = TaskSolver()

def lr_scheduler():
    global steps
    minimum = LEARNING_RATE*0.05
    calculated = LEARNING_RATE*(2.0**(-steps/400.0))
    return max(minimum,calculated)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,amsgrad=True,epsilon=ADAM_EPSILON)

start_time = time.time()

#@tf.function
def do_step(n_input, target, training, posenc):
    losses = []
    accuracies = []
    
    with tf.GradientTape() as tape:
        n_output = tasksolver(n_input, posenc)
        loss_c = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
          target, n_output, from_logits=True))
        pred_ids = tf.keras.backend.argmax(n_output,axis=-1)
        losses.append(loss_c)

    if training:
        gradients = tape.gradient(losses, tasksolver.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer.apply_gradients(zip(gradients, tasksolver.trainable_variables))

    corrects = tf.reduce_all(tf.equal(pred_ids,target),axis=-1)
    accuracy = tf.squeeze(tf.reduce_mean(tf.cast(corrects,dtype=tf.float32)))
    accuracies.append(accuracy)

    return losses, accuracies

def augment_colors(g, subst_table):
    grid_out = g
    grid_out = tf.one_hot(grid_out, depth=10, on_value=1, off_value=0, dtype=tf.int32, axis=0)
    grid_out = tf.gather(grid_out, subst_table)
    grid_out = tf.math.argmax(grid_out,axis=0)
    grid_out = tf.cast(grid_out,tf.int32)
    return grid_out
    
lookup = tf.random.uniform(shape=[N_COLORS,MODEL_SIZE], minval=-1.0, maxval=1.0)
while True:
    all_num = []
    targets = []
    n_vecs = tf.random.uniform(shape=[],minval=3,maxval=VEC_AMOUNT+1,dtype=tf.int64)
    
    new_posenc = get_posenc(n_vecs)
    
    for _ in range(BATCH_SIZE):
        num_hots = tf.random.uniform(shape=[n_vecs],minval=0,maxval=N_COLORS,dtype=tf.int64)
        n_input = tf.gather(lookup, num_hots)
        all_num.append(n_input)
        targets.append(tf.reverse(num_hots,axis=[-1]))
        
    all_num = tf.stack(all_num,axis=0)
    targets = tf.stack(targets,axis=0)
    losses, accuracies_c = do_step(all_num, target=targets, training=True, posenc=new_posenc)
    steps += 1

    print(f"{steps} {round(time.time()-start_time,4)} s ", end='')

    #HACK
    #for some reason np.set_printoptions doesnt affect scalars
    #so we print a dumdum array instead
    losses = np.array([(tf.add_n(losses) / len(losses)).numpy()])
    accs_c = np.array([(tf.add_n(accuracies_c) / len(accuracies_c)).numpy()])
    
    print(f"loss {losses} ", end='')
    print(f"acc {accs_c} ", end='')
    print(f"n_vecs {n_vecs.numpy()} ", end='')
    
    print()
    
    start_time = time.time()
