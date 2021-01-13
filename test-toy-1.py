import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
np.set_printoptions(precision=3,suppress=True)
import tensorflow as tf
import time
import enum
import matplotlib.pyplot as plt

from counting import Counting
from transformer import Transformer
import posenc

prm = {} #parameters
prm['VEC_AMOUNT'] = 14
prm['MODEL_SIZE'] = 96
prm['XF_LAYERS'] = 1
prm['XF_HEADS'] = 1
prm['LEARNING_RATE'] = 0.0005
prm['LEARNING_RATE_MIN'] = 0.00005
prm['ADAM_EPSILON'] = 1e-4
prm['BATCH_SIZE'] = 32
prm['USE_COUNTER'] = True
prm['N_COLORS'] = 2
prm['NORMALIZE_COLUMNS'] = False

for key in prm:
    print(f"{key}={prm[key]}",end=" ")
print()

steps = 0

class TaskSolver(tf.keras.Model):
    def __init__(self):
        super(TaskSolver,self).__init__()

        self.xformer_A = Transformer(model_size=prm['MODEL_SIZE'], num_layers=prm['XF_LAYERS'], heads=prm['XF_HEADS'], use_causal_mask=False, use_counter=prm['USE_COUNTER'], normalize_columns=prm['NORMALIZE_COLUMNS'])
        
        self.logit_outputs = tf.keras.layers.Dense(2)
        
    def call(self, n_input):
        n_output = n_input
        
        pes = posenc.get_posenc(n_output.shape[-2],n_output.shape[-1])
        
        n_output = n_output+pes
        n_output = self.xformer_A(n_output)
        
        n_output = self.logit_outputs(n_output)
        return n_output
        
tasksolver = TaskSolver()

def lr_scheduler():
    global steps
    minimum = prm['LEARNING_RATE_MIN']
    calculated = prm['LEARNING_RATE']*(2.0**(-steps/400.0))
    return max(minimum,calculated)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,amsgrad=True,epsilon=prm['ADAM_EPSILON'])

start_time = time.time()

#@tf.function
def do_step(n_input, target, training):
    losses = []
    accuracies = []
    
    with tf.GradientTape() as tape:
        n_output = tasksolver(n_input)
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
    
while True:
    all_num = []
    targets = []
    for _ in range(prm['BATCH_SIZE']):
        lookup = tf.random.uniform(shape=[prm['N_COLORS'],prm['MODEL_SIZE']], minval=-1.0, maxval=1.0)
        num_nums = tf.random.uniform(shape=[],minval=1,maxval=(prm['VEC_AMOUNT']-1)//2+1,dtype=tf.int64)
        allnums = tf.random.shuffle(tf.range(0,prm['VEC_AMOUNT'],dtype=tf.int64))
        chosen_nums = allnums[:num_nums]
        
        num_hots = tf.zeros([prm['VEC_AMOUNT']],dtype=tf.int64)
        num_hots = tf.tensor_scatter_nd_update(num_hots,tf.expand_dims(chosen_nums,axis=-1),tf.ones_like(chosen_nums))
        
        color_subst = tf.random.shuffle(tf.range(0,prm['N_COLORS'],dtype=tf.int32))        
        n_input = tf.gather(lookup, augment_colors(num_hots,color_subst))

        #print(augment_colors(num_hots,color_subst))
        
        all_num.append(n_input)
        targets.append(num_hots)
        
    all_num = tf.stack(all_num,axis=0)
    targets = tf.stack(targets,axis=0)
    losses, accuracies_c = do_step(all_num, target=targets, training=True)
    steps += 1

    print(f"{steps} {round(time.time()-start_time,4)} s ", end='')

    #HACK
    #for some reason np.set_printoptions doesnt affect scalars
    #so we print a dumdum array instead
    losses = np.array([(tf.add_n(losses) / len(losses)).numpy()])
    accs_c = np.array([(tf.add_n(accuracies_c) / len(accuracies_c)).numpy()])
    
    print(f"loss {losses} ", end='')
    print(f"acc {accs_c} ", end='')
    
    print()
    
    start_time = time.time()
