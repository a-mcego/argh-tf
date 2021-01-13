import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
import time
import enum
import zipfile

from counting import Counting
from transformer import Transformer
import posenc
from augment import augment

class SizePredType(enum.Enum):
    CLASSIFICATION = 1
    REAL_NUMBER = 2

prm = {}
prm['MODEL_SIZE'] = 160
prm['XF_LAYERS'] = 2
prm['XF_HEADS'] = 2
prm['LEARNING_RATE'] = 0.001
prm['LEARNING_RATE_MIN'] = 0.0001
prm['N_TASKS'] = 20
prm['ADAM_EPSILON'] = 1e-2
prm['NORMALIZE_COLUMNS'] = True
prm['SIZEPREDTYPE'] = SizePredType.CLASSIFICATION
#prm['SIZEPREDTYPE'] = SizePredType.REAL_NUMBER
prm['USE_MULTIPLE_TRANSFORMERS'] = True
prm['AUGMENT'] = True
prm['AUGMENT_TWISTS'] = True
prm['AUGMENT_COLORS'] = True
prm['USE_COUNTER'] = True
prm['BATCH_SIZE'] = 2

for key in prm:
    print(f"{key}={prm[key]}",end=" ")
print()

INS_TRAIN = 0
OUTS_TRAIN = 1
IN_TEST = 2
OUT_TEST = 3

if prm['MODEL_SIZE'] % 2 != 0:
    print(f"MODEL_SIZE should be even, but is {prm['MODEL_SIZE']}")
    exit(0)

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
        
        self.input_dense = tf.keras.layers.Dense(prm['MODEL_SIZE'])
        
        self.color_embedding = tf.keras.layers.Embedding(10, prm['MODEL_SIZE'])
        self.color_embedding.trainable = False
        
        #sizes *and* coordinates
        self.coord_embedding = tf.keras.layers.Embedding(31, prm['MODEL_SIZE']//2)
        self.coord_embedding.trainable = True
        
        #embedding for input vs. output
        self.io_embedding = tf.keras.layers.Embedding(2, prm['MODEL_SIZE'])
        self.io_embedding.trainable = True
        
        #embedding for shape vs. pixel
        self.sp_embedding = tf.keras.layers.Embedding(2, prm['MODEL_SIZE'])
        self.sp_embedding.trainable = True
        
        #embedding for c_shape_input
        self.c_input_embedding = tf.keras.layers.Embedding(2, prm['MODEL_SIZE'])
        self.c_input_embedding.trainable = True
        
        self.xformer_A = Transformer(model_size=prm['MODEL_SIZE'], num_layers=prm['XF_LAYERS'], heads=prm['XF_HEADS'], use_causal_mask=False, use_counter=prm['USE_COUNTER'], normalize_columns=prm['NORMALIZE_COLUMNS'])
        
        if prm['USE_MULTIPLE_TRANSFORMERS']:
            self.xformer_B = Transformer(model_size=prm['MODEL_SIZE'], num_layers=prm['XF_LAYERS'], heads=prm['XF_HEADS'], use_causal_mask=False, use_counter=prm['USE_COUNTER'], normalize_columns=prm['NORMALIZE_COLUMNS'])
            self.xformer_C = Transformer(model_size=prm['MODEL_SIZE'], num_layers=prm['XF_LAYERS'], heads=prm['XF_HEADS'], use_causal_mask=False, use_counter=prm['USE_COUNTER'], normalize_columns=prm['NORMALIZE_COLUMNS'])
        else:
            self.xformer_B = self.xformer_A
            self.xformer_C = self.xformer_A
        
        self.c_shape_output_logits = tf.keras.layers.Dense(31)
        self.c_pixel_output_logits = tf.keras.layers.Dense(10)
        
        self.c_shape_output_real = tf.keras.layers.Dense(1)

    def do_coord_embedding(self, ys, xs, size_shape=None):
        if size_shape is None:
            xs = self.coord_embedding(xs)
            ys = self.coord_embedding(ys)
            coords = tf.concat([ys,xs],axis=-1)
            return coords
        ype = posenc.get_posenc(size_shape[0],dim=prm['MODEL_SIZE']//2)
        xpe = posenc.get_posenc(size_shape[1],dim=prm['MODEL_SIZE']//2)
        y_result = tf.gather(ype, ys)
        x_result = tf.gather(xpe, xs)
        coords = tf.concat([y_result,x_result],axis=-1)
        return coords
        
    def embed_grid_with_shape(self,shape):
        xs = tf.range(start=0, limit=shape[1])
        ys = tf.range(start=0, limit=shape[0])
        
        xs,ys = tf.meshgrid(xs,ys)
        return self.do_coord_embedding(ys,xs, shape)
        
    def embed_arc_grid(self, grid, grid_type):
        grid_out = [self.color_embedding(grid)]
        grid_out.append(self.embed_grid_with_shape(grid.shape))
        grid_out.append(self.io_embedding(tf.constant(grid_type,shape=grid.shape)))
        grid_out.append(self.sp_embedding(tf.constant(CONST_PIXEL,shape=grid.shape)))
        
        grid_out = self.input_dense(tf.concat(grid_out,axis=-1))
        
        grid_out_shape = [tf.constant(0.0,shape=[prm['MODEL_SIZE']])]
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

        if prm['SIZEPREDTYPE'] == SizePredType.CLASSIFICATION:
            c_output = self.c_shape_output_logits(c_output)
            predicted_output_shape = tf.keras.backend.argmax(c_output,axis=-1)
        
        elif prm['SIZEPREDTYPE'] == SizePredType.REAL_NUMBER:
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
    minimum = prm['LEARNING_RATE_MIN']
    calculated = prm['LEARNING_RATE']*(2.0**(-steps/100.0))
    return max(minimum,calculated)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,amsgrad=True,epsilon=prm['ADAM_EPSILON'])

start_time = time.time()

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
            if training and prm['AUGMENT']:
                t = augment(t,prm)
                
            target_shape = t[OUT_TEST].shape
            target_grid = t[OUT_TEST]
            
            result_c, result_d = tasksolver(t)

            if prm['SIZEPREDTYPE'] == SizePredType.CLASSIFICATION:
                loss_c = tf.reduce_mean(
                  tf.keras.losses.sparse_categorical_crossentropy(
                  target_shape, result_c, from_logits=True))
                pred_ids = tf.keras.backend.argmax(result_c,axis=-1)

            elif prm['SIZEPREDTYPE'] == SizePredType.REAL_NUMBER:
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
for t_id in range(0,prm['N_TASKS']):
    t = training_tasks[t_id]
    for _ in range(prm['BATCH_SIZE']):
        for test_id in range(len(t.ins_test)):
            chosen_tasks.append([t.ins_train, t.outs_train, t.ins_test[test_id], t.outs_test[test_id]])

    t = evaluation_tasks[t_id]
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
