import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def myscovblk(out_f,inp_1,inp_2):
  v = layers.Conv2D(out_f,(1,1),strides=1,padding='same', activation='tanh')(inp_1)
  v = layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='same')(v)
  w = layers.Conv2D(out_f,kernel_size=(3,3),strides=(2,2),padding='same',activation = tf.keras.activations.relu)(inp_2)
  #w = layers.Conv2D(out_f,kernel_size=(3,3),strides=1,padding='same',activation = tf.keras.activations.relu)(w)
  w = layers.Add()([w,v])
  return w
def bblock(im,xim):
  w = myscovblk(2,im,xim)
  w = myscovblk(3,w,w)
  vw = myscovblk(4,w,w)
  return vw
def convertTRT(tf_model_dir,converted_model_name):
  '''input argument: the tensorflow model directory and the converted model name'''
  conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
  conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<32))
  conversion_params = conversion_params._replace(precision_mode="FP16") # set the floating point operation precision
  conversion_params = conversion_params._replace(maximum_cached_engines=100)

  converter = trt.TrtGraphConverterV2(input_saved_model_dir=tf_model_dir,conversion_params=conversion_params)

  converter.convert()

  converter.save(output_saved_model_dir=converted_model_name)
#build the NN model
im_input= layers.Input(shape=[240,320,3])
x = layers.Conv2D(3,(1,1),strides=1,padding='same', activation='tanh')(im_input)
x = layers.Add()([x,im_input])
end_feat = []
for i in range(7):
  h = bblock(im_input,x)
  end_feat.append(h)
x = layers.Concatenate()(end_feat)

x = layers.Conv2D(32,kernel_size=(3,3),strides=(2,2),padding='same',activation = tf.keras.activations.relu)(x)
x = layers.Conv2D(16,kernel_size=(3,3),strides=(2,2),padding='same',activation = tf.keras.activations.relu)(x)
x = layers.Conv2D(1,kernel_size=[1,1],padding='same',strides=1,activation='sigmoid')(x)
out = layers.GlobalMaxPooling2D()(x)
model = tf.keras.Model(inputs=im_input, outputs=out, name="custom_hor_grey_fat3_plus")
model.summary()
#load the trained weights
model.load_weights('/home/joshjet/practice-jetson/mymodel/grayfat3/custom_hor_greyfat3_plus_ne400.h5')
model.save('mymodel_tf', include_optimizer=False,save_format='tf') #save in TF format
convertTRT('mymodel_tf','mymodel_horgreyfat3p_ep400_TFTRT_fp16') #convert the model



