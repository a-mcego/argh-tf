import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
import time
import enum
import zipfile

class SizePredType(enum.Enum):
    CLASSIFICATION = 1
    REAL_NUMBER = 2

MODEL_SIZE = 128
XF_LAYERS = 3
XF_HEADS = 2
LEARNING_RATE = 0.0005
N_TASKS = 2
ADAM_EPSILON = 1e-2
NORMALIZE_COLUMNS = True
SIZEPREDTYPE = SizePredType.CLASSIFICATION
#SIZEPREDTYPE = SizePredType.REAL_NUMBER
USE_MULTIPLE_TRANSFORMERS = True
FFN_DROPOUT = 0.0 #doesn't work yet
AUGMENT = True
AUGMENT_TWISTS = True
AUGMENT_COLORS = False
USE_COUNTER = True
BATCH_SIZE = 4

print(f"MODEL_SIZE={MODEL_SIZE} ",end="")
print(f"XF_LAYERS={XF_LAYERS} ",end="")
print(f"XF_HEADS={XF_HEADS} ",end="")
print(f"LEARNING_RATE={LEARNING_RATE} ",end="")
print(f"N_TASKS={N_TASKS} ",end="")
print(f"ADAM_EPSILON={ADAM_EPSILON} ",end="")
print(f"NORMALIZE_COLUMNS={NORMALIZE_COLUMNS} ",end="")
print(f"SIZEPREDTYPE={SIZEPREDTYPE} ",end="")
print(f"USE_MULTIPLE_TRANSFORMERS={USE_MULTIPLE_TRANSFORMERS} ",end="")
print(f"AUGMENT={AUGMENT} ",end="")
print(f"AUGMENT_TWISTS={AUGMENT_TWISTS} ",end="")
print(f"AUGMENT_COLORS={AUGMENT_COLORS} ",end="")
print(f"USE_COUNTER={USE_COUNTER} ",end="")
print(f"BATCH_SIZE={BATCH_SIZE} ",end="")
print()

INS_TRAIN = 0
OUTS_TRAIN = 1
IN_TEST = 2
OUT_TEST = 3

if MODEL_SIZE % 2 != 0:
    print(f"MODEL_SIZE should be even, but is {MODEL_SIZE}")
    exit(0)

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

        if USE_COUNTER:
            self.counter_expander = tf.keras.layers.Dense(model_size,activation=tf.nn.softplus)
            self.counter_merger = tf.keras.layers.Dense(model_size,use_bias=False)
            self.counter_norm = tf.keras.layers.LayerNormalization()
        
        
    def reduce(self, matrix):
        mean = tf.math.reduce_mean(matrix, axis=0, keepdims=True)
        stddev = tf.math.reduce_std(matrix, axis=0, keepdims=True)
        return (matrix - mean)/stddev
        
    @tf.function(input_signature=[tf.TensorSpec(shape=[None,MODEL_SIZE], dtype=tf.float32),tf.TensorSpec(shape=[None,MODEL_SIZE], dtype=tf.float32)])
    def call(self, query, value):
        heads = []

        if USE_COUNTER:
            #cosine similarity
            counter = tf.math.l2_normalize(value, axis=-1)
            counter = tf.matmul(counter, counter, transpose_b=True)
            #relu to filter out bads (this could be improved upon)
            counter = tf.nn.relu(counter)
            #do the count itself
            counter = tf.reduce_sum(counter,axis=-1)
            counter = tf.expand_dims(counter,axis=-1)
            #do postprocessing of the count.
            counter = self.counter_expander(counter)
            
            value = self.counter_merger(tf.concat([value,counter],axis=-1))
            value = self.counter_norm(value)

        
        out_kv = self.w_kv(value)
        out_q = self.w_q(query)
        
        if NORMALIZE_COLUMNS:
            out_kv = self.reduce(out_kv)
            out_q = self.reduce(out_q)

        out_kv = tf.split(out_kv, self.n_heads*2, axis=-1)
        out_q = tf.split(out_q, self.n_heads, axis=-1)
        
        for i in range(self.n_heads):
            wqd = out_q[i]
            wkd = out_kv[i]
            wvd = out_kv[i+self.n_heads]
            
            score = tf.matmul(wqd, wkd, transpose_b=True)*self.normalizer

            alignment = tf.nn.softmax(score, axis=-1)
            head = tf.matmul(alignment, wvd)
            heads.append(head)

        heads = tf.concat(heads, axis=-1)
        heads = tf.reshape(heads, [-1,MODEL_SIZE])
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
        self.ffn_dropout = tf.keras.layers.Dropout(FFN_DROPOUT)
        
        self.use_causal_mask = use_causal_mask
        
        
    def call(self, target_sequence, source_sequence=None, src_as_list=False):
        if self.use_causal_mask:
            mask_size = target_sequence.shape[1]
            look_left_only_mask = tf.linalg.band_part(tf.ones((mask_size,mask_size)), -1, 0)
        else:
            look_left_only_mask = None
            
        src_att_in = target_sequence
        for i in range(self.num_layers):
            #src_att_in has the data now
        
            if source_sequence is not None:
                if src_as_list:
                    #src_att_out = []
                    #for seq_item in source_sequence:
                    #    src_att_out.append(self.src_att[i](src_att_in, seq_item))
                    #src_att_out = tf.add_n(src_att_out)
                    
                    processed_seq = tf.concat(source_sequence,axis=0)
                    src_att_out = self.src_att[i](src_att_in, processed_seq)
                    
                else:
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
            if FFN_DROPOUT > 0.0:
                ffn_mid = self.ffn_dropout(ffn_mid,training=True)
            
            ffn_out = self.dense_2[i](ffn_mid)

            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            src_att_in = ffn_out
            #src_att_in has the data now

        return src_att_in
    
class Task:
    def __init__(self):
        self.filename = ""
        self.ins_train = []
        self.outs_train = []
        self.ins_test = []
        self.outs_test = []
        
def get_tasks(name):
    tasks = []
    
    #HACK (sorta)
    #you couldnt drag-and-drop *folders* into colab
    #so i had to write this zip reader
    if name[-4:] == '.zip':
        zipname = name
        zip = zipfile.ZipFile(name)
        
        namelist = zip.namelist()
        for name in namelist:
            task = Task()
            task.filename = name
            data = zip.read(name)
            task.size = len(data)
            data = json.loads(data)
            
            for e in data['train']:
                task.ins_train.append(tf.convert_to_tensor(e['input']))
                task.outs_train.append(tf.convert_to_tensor(e['output']))

            for e in data['test']:
                task.ins_test.append(tf.convert_to_tensor(e['input']))
                task.outs_test.append(tf.convert_to_tensor(e['output']))
                
            tasks.append(task)
        
    else:
        directory = name
        onlyfiles = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        onlyfiles = [f for f in onlyfiles if f[-4:] == 'json']
        for f in onlyfiles:
            task = Task()
            task.filename = f
            task.size = os.path.getsize(os.path.join(directory,f))
            data = json.load(open(os.path.join(directory,f),'r'))
            
            for e in data['train']:
                task.ins_train.append(tf.convert_to_tensor(e['input']))
                task.outs_train.append(tf.convert_to_tensor(e['output']))

            for e in data['test']:
                task.ins_test.append(tf.convert_to_tensor(e['input']))
                task.outs_test.append(tf.convert_to_tensor(e['output']))
                
            tasks.append(task)
            
    return tasks

CONST_INPUT = 0
CONST_OUTPUT = 1
CONST_SHAPE = 0
CONST_PIXEL = 1

class TaskSolver(tf.keras.Model):
    def __init__(self):
        super(TaskSolver,self).__init__()
        
        self.input_dense = tf.keras.layers.Dense(MODEL_SIZE)
        
        self.color_embedding = tf.keras.layers.Embedding(10, MODEL_SIZE)
        self.color_embedding.trainable = True
        
        #sizes *and* coordinates
        self.coord_embedding = tf.keras.layers.Embedding(31, MODEL_SIZE//2)
        self.coord_embedding.trainable = True
        
        #embedding for input vs. output
        self.io_embedding = tf.keras.layers.Embedding(2, MODEL_SIZE)
        self.io_embedding.trainable = True
        
        #embedding for shape vs. pixel
        self.sp_embedding = tf.keras.layers.Embedding(2, MODEL_SIZE)
        self.sp_embedding.trainable = True
        
        #embedding for c_shape_input
        self.c_input_embedding = tf.keras.layers.Embedding(2, MODEL_SIZE)
        self.c_input_embedding.trainable = True
        
        #self.xformer_COLOR = Transformer(model_size=MODEL_SIZE, num_layers=XF_LAYERS, heads=XF_HEADS, use_causal_mask=False)
        self.xformer_A = Transformer(model_size=MODEL_SIZE, num_layers=XF_LAYERS, heads=XF_HEADS, use_causal_mask=False)
        
        if USE_MULTIPLE_TRANSFORMERS:
            self.xformer_B = Transformer(model_size=MODEL_SIZE, num_layers=XF_LAYERS, heads=XF_HEADS, use_causal_mask=False)
            self.xformer_C = Transformer(model_size=MODEL_SIZE, num_layers=XF_LAYERS, heads=XF_HEADS, use_causal_mask=False)
        else:
            self.xformer_B = self.xformer_A
            self.xformer_C = self.xformer_A
        
        self.c_shape_output_logits = tf.keras.layers.Dense(31)
        self.c_pixel_output_logits = tf.keras.layers.Dense(10)
        
        self.c_shape_output_real = tf.keras.layers.Dense(1)

    def do_coord_embedding(self, ys, xs):
        xs = self.coord_embedding(xs)
        ys = self.coord_embedding(ys)
        coords = tf.concat([ys,xs],axis=-1)
        return coords
        
    def embed_grid_with_shape(self,shape):
        xs = tf.range(start=0, limit=shape[1])
        ys = tf.range(start=0, limit=shape[0])
        
        xs,ys = tf.meshgrid(xs,ys)
        return self.do_coord_embedding(ys,xs)
        
    def embed_arc_grid(self, grid, grid_type):
        grid_out = [self.color_embedding(grid)]
        grid_out.append(self.embed_grid_with_shape(grid.shape))
        grid_out.append(self.io_embedding(tf.constant(grid_type,shape=grid.shape)))
        grid_out.append(self.sp_embedding(tf.constant(CONST_PIXEL,shape=grid.shape)))
        
        grid_out = self.input_dense(tf.concat(grid_out,axis=-1))
        
        grid_out_shape = [tf.constant(0.0,shape=[MODEL_SIZE])]
        grid_out_shape.append(self.do_coord_embedding(grid_out.shape[0],grid_out.shape[1]))
        grid_out_shape.append(self.io_embedding(tf.constant(grid_type,shape=[])))
        grid_out_shape.append(self.sp_embedding(tf.constant(CONST_SHAPE,shape=[])))
        grid_out_shape = self.input_dense(tf.expand_dims(tf.concat(grid_out_shape,axis=-1),axis=0))
        
        grid_out = tf.reshape(grid_out,[grid_out.shape[0]*grid_out.shape[1],grid_out.shape[2]])
        
        grid_out = tf.concat([grid_out,grid_out_shape],axis=0)
        return grid_out

    def embed_empty_arc_grid(self, grid_shape, grid_type):
        grid_out = self.embed_grid_with_shape(grid_shape)
        grid_out += self.sp_embedding(tf.tile(tf.constant(CONST_PIXEL,shape=[1,1]),multiples=grid_shape))
        grid_out += self.io_embedding(tf.tile(tf.constant(grid_type,shape=[1,1]),multiples=grid_shape))
        
        grid_out_shape = self.do_coord_embedding(grid_out.shape[0],grid_out.shape[1])
        grid_out_shape += self.io_embedding(tf.constant(grid_type,shape=[]))
        grid_out_shape += self.sp_embedding(tf.constant(CONST_SHAPE,shape=[]))
        grid_out_shape = tf.expand_dims(grid_out_shape,axis=0)
        
        grid_out = tf.reshape(grid_out,[grid_out.shape[0]*grid_out.shape[1],grid_out.shape[2]])
        grid_out = tf.concat([grid_out,grid_out_shape],axis=0)
        return grid_out
        
    def call(self, task, is_training=True):
        in_grids = []
        for grid in task[INS_TRAIN]:
            embedded_grid = self.embed_arc_grid(grid, grid_type=CONST_INPUT)
            in_grids.append(embedded_grid)

        out_grids = []
        for grid in task[OUTS_TRAIN]:
            embedded_grid = self.embed_arc_grid(grid, grid_type=CONST_OUTPUT)
            out_grids.append(embedded_grid)           
            
        together = zip(in_grids,out_grids)
        pairs = [tf.concat([x[0],x[1]],axis=0) for x in together]
           
        a_outputs = []
        for pair in pairs:
            a_output = self.xformer_A(pair)
            a_output = tf.concat([a_output,pair],axis=-2)
            a_outputs.append(a_output)
            
        test_in_grid = self.embed_arc_grid(task[IN_TEST], grid_type=CONST_INPUT)
        
        b_output = self.xformer_B(test_in_grid, a_outputs, True)
        
        c_input = self.c_input_embedding(tf.convert_to_tensor([0,1]))
            
        c_output = self.xformer_C(c_input, tf.concat([b_output,test_in_grid],axis=-2), False)

        if SIZEPREDTYPE == SizePredType.CLASSIFICATION:
            c_output = self.c_shape_output_logits(c_output)
            predicted_output_shape = tf.keras.backend.argmax(c_output,axis=-1)
        
        elif SIZEPREDTYPE == SizePredType.REAL_NUMBER:
            c_output = 7.5*(self.c_shape_output_real(c_output)+2.0)
            predicted_output_shape = tf.cast(tf.math.round(tf.squeeze(c_output)),tf.int64)
            predicted_output_shape = tf.clip_by_value(predicted_output_shape,1,30)

        if is_training:
            d_input = self.embed_empty_arc_grid(task[OUT_TEST].shape, grid_type=CONST_OUTPUT)
        else:
            d_input = self.embed_empty_arc_grid(predicted_output_shape, grid_type=CONST_OUTPUT)

        b_output = tf.concat([b_output,test_in_grid],axis=-2)
            
        d_output = self.xformer_C(d_input, b_output, False)
        
        d_output = self.c_pixel_output_logits(d_output[:-1])
        
        return c_output, d_output
    

tasksolver = TaskSolver()

steps = 0

def lr_scheduler():
    global steps
    minimum = LEARNING_RATE*0.05
    calculated = LEARNING_RATE*(2.0**(-steps/100.0))
    return max(minimum,calculated)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,amsgrad=True,epsilon=ADAM_EPSILON)

start_time = time.time()

def augment_twists(g, twists):
    grid_out = g
    grid_out = tf.transpose(grid_out) if twists[0] else grid_out
    grid_out = tf.reverse(grid_out,axis=[0]) if twists[1] else grid_out
    grid_out = tf.reverse(grid_out,axis=[1]) if twists[2] else grid_out
    return grid_out

def augment_colors(g, subst_table):
    grid_out = g
    grid_out = tf.one_hot(grid_out, depth=10, on_value=1, off_value=0, dtype=tf.int32, axis=0)
    grid_out = tf.gather(grid_out, subst_table)
    grid_out = tf.math.argmax(grid_out,axis=0)
    grid_out = tf.cast(grid_out,tf.int32)
    return grid_out
    
def augment(t):
    twists = tf.cast(tf.random.uniform([2,3], minval=0, maxval=2, dtype=tf.int64),tf.bool)
    twists = twists.numpy()
    
    def do_augment(g, twist_id):
        ret = g
        if AUGMENT_TWISTS:
            ret = augment_twists(ret,twists[twist_id])
        if AUGMENT_COLORS:
            ret = augment_colors(ret,color_subst)
        return ret
    
    color_subst = tf.random.shuffle(tf.range(0,10,dtype=tf.int32))
    
    ins_train = []
    for g in t[INS_TRAIN]:
        ins_train.append(do_augment(g,twist_id=0))
    in_test = do_augment(t[IN_TEST],twist_id=0)

    outs_train = []
    for g in t[OUTS_TRAIN]:
        outs_train.append(do_augment(g,twist_id=1))
    out_test = do_augment(t[OUT_TEST],twist_id=1)
    return [ins_train, outs_train, in_test, out_test]

#@tf.function
def do_step(chosen_tasks, training):
    losses = []
    accuracies_c = []
    accuracies_d = []
    gradient_accum = None
    for t_id,t in enumerate(chosen_tasks):
        corrects_c = []
        corrects_d = []
        with tf.GradientTape() as tape:
            if training and AUGMENT:
                t = augment(t)
                
            target_shape = t[OUT_TEST].shape
            target_grid = t[OUT_TEST]
            
            result_c, result_d = tasksolver(t)

            if SIZEPREDTYPE == SizePredType.CLASSIFICATION:
                loss_c = tf.reduce_mean(
                  tf.keras.losses.sparse_categorical_crossentropy(
                  target_shape, result_c, from_logits=True))
                pred_ids = tf.keras.backend.argmax(result_c,axis=-1)

            elif SIZEPREDTYPE == SizePredType.REAL_NUMBER:
                result_c = tf.squeeze(result_c)
                loss_c = tf.reduce_mean(tf.square(tf.cast(target_shape,tf.float32)-result_c))
                pred_ids = tf.cast(tf.math.round(result_c),tf.int64)
            
            correct_c = tf.reduce_all(tf.equal(pred_ids,target_shape))
            
            pred_grid = tf.keras.backend.argmax(result_d,axis=-1)
            
            if correct_c.numpy() == True:
                pred_grid = tf.reshape(pred_grid, target_grid.shape) #yes i'm doing it twice.
                correct_d = tf.reduce_all(tf.equal(pred_grid,tf.cast(target_grid,tf.int64)))
            else:
                correct_d = tf.constant(False)

            if training:
                pred_grid = tf.reshape(pred_grid, target_grid.shape) #yes i'm doing it twice.
                loss_d = tf.reduce_mean(
                  tf.keras.losses.sparse_categorical_crossentropy(
                  target_grid, result_d, from_logits=True))
            else:
                loss_d = tf.constant(0.0)
               
              
            corrects_c.append(correct_c)
            corrects_d.append(correct_d)
            #print(pred_ids.numpy(), target_shape, correct.numpy())
            losses.append(loss_c)
            losses.append(loss_d)
        
        if training:
            gradients = tape.gradient(losses, tasksolver.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            
            if gradient_accum is None:
                gradient_accum = [tf.convert_to_tensor(tens) for tens in gradients]
            else:
                for t_id2 in range(0,len(gradient_accum)):
                    gradient_accum[t_id2] = tf.add(gradient_accum[t_id2],tf.convert_to_tensor(gradients[t_id2]))
        
        corrects_c = tf.reduce_all(tf.stack(corrects_c,axis=0))
        corrects_d = tf.reduce_all(tf.stack(corrects_d,axis=0))
        accuracy_c = tf.squeeze(tf.reduce_mean(tf.cast(corrects_c,dtype=tf.float32)))
        accuracy_d = tf.squeeze(tf.reduce_mean(tf.cast(corrects_d,dtype=tf.float32)))
        accuracies_c.append(accuracy_c)
        accuracies_d.append(accuracy_d)
        print(f"{t_id}/{len(chosen_tasks)}",end="     \r")

    totalparams = tf.reduce_sum([tf.size(tens) for tens in tasksolver.trainable_variables])
    

    if training:
        #print(f"             #gt {len(gradient_accum)} #p {totalparams} ", end='')
        optimizer.apply_gradients(zip(gradient_accum, tasksolver.trainable_variables))
    
    return losses, accuracies_c, accuracies_d

training_tasks = get_tasks("training.zip")
training_tasks = sorted(training_tasks,key=lambda task: task.size)

evaluation_tasks = get_tasks("evaluation.zip")
evaluation_tasks = sorted(evaluation_tasks,key=lambda task: task.size)

chosen_tasks = []
validation_tasks = []
for t_id in range(1,N_TASKS+1):
    t = training_tasks[t_id]
    for _ in range(BATCH_SIZE):
        for test_id in range(len(t.ins_test)):
            chosen_tasks.append([t.ins_train, t.outs_train, t.ins_test[test_id], t.outs_test[test_id]])

    t = training_tasks[t_id]
    for test_id in range(len(t.ins_test)):
        validation_tasks.append([t.ins_train, t.outs_train, t.ins_test[test_id], t.outs_test[test_id]])

np.set_printoptions(precision=3,suppress=True)

while True:
    losses, accuracies_c, accuracies_d = do_step(chosen_tasks,training=True)
    steps += 1
    
    if (steps%5 == 0):
        losses2, accuracies2_c, accuracies2_d = do_step(validation_tasks,training=False)

    print(f"{steps} {round(time.time()-start_time,4)} s ", end='')

    #HACK
    #for some reason np.set_printoptions doesnt affect scalars
    #so we print a dumdum array instead
    losses = np.array([(tf.add_n(losses) / len(losses)).numpy()])
    accs_c = np.array([(tf.add_n(accuracies_c) / len(accuracies_c)).numpy()])
    accs_d = np.array([(tf.add_n(accuracies_d) / len(accuracies_d)).numpy()])
    
    print(f"loss {losses} ", end='')
    print(f"acc {accs_c} {accs_d} ", end='')
        
    if (steps%5 == 0):
        print(f"VALID ", end='')
        
        losses = np.array([(tf.add_n(losses2) / len(losses2)).numpy()])
        accs_c = np.array([(tf.add_n(accuracies2_c) / len(accuracies2_c)).numpy()])
        accs_d = np.array([(tf.add_n(accuracies2_d) / len(accuracies2_d)).numpy()])
        print(f"loss {losses} ", end='')
        print(f"acc {accs_c} {accs_d} ", end='')
    
    print()
    
    start_time = time.time()
