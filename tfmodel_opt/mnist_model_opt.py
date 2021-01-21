import logging
import os
import tempfile
import zipfile

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

logging.getLogger("tensorflow").setLevel(logging.DEBUG)

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images  = test_images / 255.0

def mnist_model():
    # Define the model architecture.
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(
        train_images,
        train_labels,
        validation_split=0.1,
        epochs=1
    )
    model.summary()
    tf.keras.models.save_model(model,'./tmp/model')

def prune():
    model = tf.keras.models.load_model('./tmp/model')
    model.summary()

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 128
    epochs = 1
    validation_split = 0.1 # 10% of training set will be used for validation set. 

    num_images = train_images.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                            final_sparsity=0.80,
                                                            begin_step=0,
                                                            end_step=end_step)
    }
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
    model_for_export.summary()
    tf.keras.models.save_model(model_for_export,'./tmp/model_pruned')

def cluster():
    model = tf.keras.models.load_model('./tmp/model_pruned')
    clustering_params = {
        'number_of_clusters': 16,
        'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.LINEAR
    }

    # Cluster a whole model
    clustered_model = tfmot.clustering.keras.cluster_weights(model, **clustering_params)
    clustered_model.summary()
    final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
    final_model.summary()
    tf.keras.models.save_model(final_model,'./tmp/model_clustered')

def quant():
    model = tf.keras.models.load_model('./tmp/model_clustered')
    q_aware_model = tfmot.quantization.keras.quantize_model(model)
    tf.keras.models.save_model(q_aware_model,'./tmp/model_quanted')

def tflite():
    model = tf.keras.models.load_model('./tmp/model_quanted')
    # 为TFLite后端创建量化模型,获得一个具有int8权重和uint8激活的实际量化模型
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('./tmp/model.tflite', 'wb') as f:
        f.write(tflite_model)

def get_gzipped_model_size(path):
    # Returns size of gzipped model
    import os
    import zipfile
    
    startdir = path  # 要压缩的文件夹路径
    _, zipped_file = tempfile.mkstemp('.zip')# 压缩后文件夹的名字

    z = zipfile.ZipFile(zipped_file, 'w', zipfile.ZIP_DEFLATED)  # 参数一：文件夹名
    for dirpath, dirnames, filenames in os.walk(startdir):
        fpath = dirpath.replace(startdir, '')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath+filename)
    z.close()

    print(path+' in MB:',os.path.getsize(zipped_file)/ float(2**20))
    return os.path.getsize(zipped_file)

if __name__ == "__main__":
    # mnist_model()
    get_gzipped_model_size('./tmp/model')
    get_gzipped_model_size('./tmp/model_pruned')
    get_gzipped_model_size('./tmp/model_clustered')
    get_gzipped_model_size('./tmp/model_quanted')
    get_gzipped_model_size('./tmp/model.tflite')