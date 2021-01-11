#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import numpy as np

def load_mnist_data():
   mnist = tf.keras.datasets.mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_train = x_train.astype('float32')
   x_test = x_test.astype('float32')
   x_train, x_test = x_train / 255.0, x_test / 255.0
   return x_train,y_train,x_test,y_test

def mnist_model():
   '''
   mnist model build
   '''
   x_train,y_train,x_test,y_test = load_mnist_data()

   model = tf.keras.models.Sequential()
   model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
   model.add(tf.keras.layers.Dense(512, activation='relu'))
   model.add(tf.keras.layers.Dropout(0.2))
   model.add(tf.keras.layers.Dense(10, activation='softmax'))

   model.summary()
   model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
   model.fit(x_train, y_train, batch_size=64,epochs=10)
   score = model.evaluate(x_test,  y_test, verbose=2)
   print('loss:',score[0])
   print('accuracy:',score[1])
   # model.save('./model/tf_model',save_format = 'tf')
   tf.saved_model.save(model,'./model/tf_model')
   tf.keras.utils.plot_model(model,'model_info.png',show_shapes = True)

def trt(trt_opt):
   '''
   使用tensorRT优化模型
   '''
   converter = tf.experimental.tensorrt.Converter(input_saved_model_dir='./model/tf_model')
   converter.convert()#完成转换,但是此时没有进行优化,优化在执行推理时完成
   if trt_opt == True:
      mnist = tf.keras.datasets.mnist
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      x_test = x_test.astype('float32')
      x_test = x_test / 255.0
      def input_fn():
         yield (x_test[:1])
      converter.build(input_fn) #优化后保存
      converter.save('trt_model_opt')
   else:
      converter.save('trt_model')

def trt_test(model_path):
   x_train,y_train,x_test,y_test = load_mnist_data()
   model_loaded = tf.saved_model.load(model_path)#读取模型

   graph_func = model_loaded.signatures['serving_default']#获取推理函数
   t=time.time()
   #output = graph_func(tf.constant(x_test))
   output = model_loaded(x_test)
   print(output[0],'\n',time.time()-t)

def test(model_path):
   x_train,y_train,x_test,y_test = load_mnist_data()

   model = tf.keras.models.load_model(model_path)

   t=time.time()
   output = model(x_test)
   print(output[0],'\n',time.time()-t)

def tflite():
   'SavedModel to TensorFlow Lite'
   converter = tf.lite.TFLiteConverter.from_saved_model("./model/tf_model")
   converter.optimizations = [tf.lite.Optimize.DEFAULT] # 量化
   tflite_quantized_model = converter.convert()

   open("./model/quantized_converted_model.tflite", "wb").write(tflite_quantized_model)

def tflite_run(model_path):
   # Load the TFLite model and allocate tensors.
   interpreter = tf.lite.Interpreter(model_path)
   interpreter.allocate_tensors()
   # Get input and output tensors.
   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()
   # Test the model on random input data.
   input_shape = input_details[0]['shape']
   input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
   interpreter.set_tensor(input_details[0]['index'], input_data)

   interpreter.invoke()

   # The function `get_tensor()` returns a copy of the tensor data.
   # Use `tensor()` in order to get a pointer to the tensor.
   output_data = interpreter.get_tensor(output_details[0]['index'])
   print(output_data)

if __name__ == '__main__':
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   # mnist_model()
   # trt(True)
   # tflite()
   # test('./model/tf_model')
   tflite_run("./model/quantized_converted_model.tflite")