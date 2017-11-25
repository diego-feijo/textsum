"""Example of Converting TextSum model data.
Usage:
python data_convert_example.py --command binary_to_text --in_file data/data --out_file data/text_data
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
python data_convert_example.py --command binary_to_text --in_file data/binary_data --out_file data/text_data2
diff data/text_data2 data/text_data
"""

import struct
import sys
import collections
import ast

import tensorflow as tf

from tensorflow.core.example import example_pb2

import json

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'binary_to_text',
                           'Either binary_to_text or text_to_binary.'
                           'Specify FLAGS.in_file accordingly.')
tf.app.flags.DEFINE_string('in_file', '', 'path to file')
tf.app.flags.DEFINE_string('out_file', '', 'path to file')

def _binary_to_text():
  reader = open(FLAGS.in_file, 'rb')
  writer = open(FLAGS.out_file, 'w')
  while True:
    len_bytes = reader.read(8)
    if not len_bytes:
      sys.stderr.write('Done reading\n')
      return
    str_len = struct.unpack('q', len_bytes)[0]
    tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    tf_example = example_pb2.Example.FromString(tf_example_str)
    examples = []
    for key in tf_example.features.feature:
      examples.append('%s=%s' % (key, tf_example.features.feature[key].bytes_list.value[0]))
    writer.write('%s\n' % '\t'.join(examples))
  reader.close()
  writer.close()


def _text_to_binary():
  inputs = open(FLAGS.in_file, 'r').readlines()
  writer = open(FLAGS.out_file, 'wb')
  for inp in inputs:
    tf_example = example_pb2.Example()
    for feature in inp.strip().split('\t'):
      (k, v) = feature.split('=')
      tf_example.features.feature[k].bytes_list.value.extend([v])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))
  writer.close()

def _news_to_binary():
  with open(FLAGS.out_file, mode='wb') as w:
    with open(FLAGS.in_file, mode='r') as r:
      while True:
        s = r.readline()
        if len(s) == 0:
          break
        tf_example = example_pb2.Example()
        dict = json.loads(s)
        for k,v in dict.items():
          tf_example.features.feature[k].bytes_list.value.extend([tf.compat.as_bytes(v)])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        w.write(struct.pack('q', str_len))
        w.write(struct.pack('%ds' % str_len, tf_example_str))

def _build_vocab():
  with open(FLAGS.in_file, mode='r') as r:
    counter = collections.Counter()
    for line in r:
      counter.update(line.split())

  with open(FLAGS.out_file, mode='w') as w:
    for word,count in counter.most_common(200000-2):
      w.write(word + ' ' + str(count) + '\n')
    w.write('<UNK> 0\n')
    w.write('<PAD> 0\n')

def main(unused_argv):
  assert FLAGS.command and FLAGS.in_file and FLAGS.out_file
  if FLAGS.command == 'binary_to_text':
    _binary_to_text()
  elif FLAGS.command == 'text_to_binary':
    _text_to_binary()
  elif FLAGS.command == 'news_to_binary':
    _news_to_binary()
  elif FLAGS.command == 'build_vocab':
    _build_vocab()


if __name__ == '__main__':
  tf.app.run()
