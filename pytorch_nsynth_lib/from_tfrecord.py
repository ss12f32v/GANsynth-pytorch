"""
File: from_tfrecord.py
Author: Kwon-Young Choi
Email: kwon-young.choi@hotmail.fr
Date: 2018-11-12
Description: read nsynth dataset from tfrecord file
"""
import tensorflow as tf
import autodebug


def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'note': tf.FixedLenFeature([], tf.int64),
            'instrument': tf.FixedLenFeature([], tf.int64),
            'pitch': tf.FixedLenFeature([], tf.int64),
            'velocity': tf.FixedLenFeature([], tf.int64),
            'sample_rate': tf.FixedLenFeature([], tf.int64),
            'audio': tf.FixedLenSequenceFeature(
                shape=[], dtype=tf.float32, allow_missing=True),
            'qualities': tf.FixedLenSequenceFeature(
                shape=[], dtype=tf.int64, allow_missing=True),
            'instrument_family': tf.FixedLenFeature([], tf.int64),
            'instrument_source': tf.FixedLenFeature([], tf.int64),
        })
    return features


data_path = ['data/nsynth-test.tfrecord']
dataset = tf.data.TFRecordDataset(data_path)
dataset = dataset.map(parser)
dataset = dataset.batch(32)
iterator = dataset.make_one_shot_iterator()
batch_notes = iterator.get_next()

with tf.Session() as sess:
    cpt = 0
    while True:
        print(cpt)
        try:
            out = sess.run(batch_notes)
            for key, value in out.items():
                print(key, value.dtype, value.shape)
        except tf.errors.OutOfRangeError:
            break
        cpt += 1
