# 导入TensorFlow和TensorBoard HParams插件
import datetime
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

def train_test_model(hparams):
    '''
    该模型将非常简单：两个密集层之间有一个辍学层。不再对超参数进行硬编码，相反，超参数在hparams字典中提供，并在整个训练功能中使用.
    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(x_train, y_train, 
                epochs=1,

    ) 
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy

def run(run_dir, hparams):
    '每次运行记录具有超参数和最终精度的hparams摘要'
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

'''超参数调整'''
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32])) # 第一密集层中的单元数
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2)) # dropout层中的dropout比例
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd'])) # 优化器

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('./logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

session_num = 0 
for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)

      print({h.name: hparams[h] for h in hparams})
      
      run('./logs/hparam_tuning/' + run_name, hparams)

      session_num += 1