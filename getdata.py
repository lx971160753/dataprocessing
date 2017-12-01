
from parameter import *

def getTFrecorddata(data_dir,minafterdequeue,batchsize,capacity):
    files = tf.train.match_filenames_once(data_dir)
    filename_queue = tf.train.string_input_producer(files,shuffle =True)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'data_raw': tf.FixedLenFeature([], tf.string),
                'lable_raw[0]': tf.FixedLenFeature([], tf.int64),
                'lable_raw[1]': tf.FixedLenFeature([], tf.int64)
            }
    )
    data = features['data_raw']
    label = [features['lable_raw[0]'],features['lable_raw[1]']]

    label1 = tf.cast(label, tf.float32)
    label2 = tf.reshape(label1,[2,])
    data1 = tf.cast(tf.decode_raw(data, tf.float64), tf.float32)
    data2 = tf.reshape(data1, [3,7,48])
    image_batch1, label_batch = tf.train.shuffle_batch(
        [data2, label2], batch_size=batchsize, capacity=capacity, min_after_dequeue=minafterdequeue)
    return image_batch1,label_batch