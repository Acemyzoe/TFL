	i9�Cm�@i9�Cm�@!i9�Cm�@	���萣@���萣@!���萣@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$i9�Cm�@�"j��G�?Aa5��6&@YF^��_�?*	gfffft@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�`��p�?!n��>@)�H���γ?1~�_�8@:Preprocessing2F
Iterator::Model��0�*�?!�����C@)<0��?1�/oV34@:Preprocessing2U
Iterator::Model::ParallelMapV2���f�?!����&3@)���f�?1����&3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapo+�6�?!՜;f47@)���K�'�?1q&�2,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	]��?!��P��H"@)	]��?1��P��H"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorx` �C��?!Q6K�n^@)x` �C��?1Q6K�n^@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS��:�?!Ndn�]N@):��ٕ?1�.h�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9���萣@Ic�x�ZX@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�"j��G�?�"j��G�?!�"j��G�?      ��!       "      ��!       *      ��!       2	a5��6&@a5��6&@!a5��6&@:      ��!       B      ��!       J	F^��_�?F^��_�?!F^��_�?R      ��!       Z	F^��_�?F^��_�?!F^��_�?b      ��!       JCPU_ONLYY���萣@b qc�x�ZX@