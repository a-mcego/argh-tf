import tensorflow as tf
import numpy as np

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
    
def augment(t,prm):
    twists = tf.cast(tf.random.uniform([2,3], minval=0, maxval=2, dtype=tf.int64),tf.bool)
    twists = twists.numpy()
    color_subst = tf.random.shuffle(tf.range(0,10,dtype=tf.int32))
    
    def do_augment(g, twist_id):
        ret = g
        if prm['AUGMENT_TWISTS']:
            ret = augment_twists(ret,twists[twist_id])
        if prm['AUGMENT_COLORS']:
            ret = augment_colors(ret,color_subst)
        return ret

    twist_ids = [0,0,1,1]
        
    outputs = []
    for thing_id in range(len(t)):
        thing = t[thing_id]
        if type(thing) == list:
            output = []
            for g in thing:
                output.append(do_augment(g,twist_id=twist_ids[thing_id]))
            outputs.append(output)
        else:
            outputs.append(do_augment(thing,twist_id=twist_ids[thing_id]))
            
    return outputs
    """        
    ins_train = []
    for g in t[INS_TRAIN]:
        ins_train.append(do_augment(g,twist_id=0))
    in_test = do_augment(t[IN_TEST],twist_id=0)

    outs_train = []
    for g in t[OUTS_TRAIN]:
        outs_train.append(do_augment(g,twist_id=1))
    out_test = do_augment(t[OUT_TEST],twist_id=1)
    return [ins_train, outs_train, in_test, out_test]"""
