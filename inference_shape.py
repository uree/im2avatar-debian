import tensorflow.compat.v1 as tf
import numpy as np
import os
import h5py
import sys
sys.path.append('./utils')
sys.path.append('./models')

import dataset as dataset
import model_shape as model

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', default='./train_shape', help="""Directory where to write summaries and checkpoint.""")

parser.add_argument('--base_dir', default='./data/ShapeNetCore_im2avatar', help="""The path containing all the samples.""")

parser.add_argument('--cat_id', default='04379243', help="""The category id for each category: 02958343, 03001627, 03467517, 04379243""")

parser.add_argument('--data_list_path', default='./data_list', help="""The path containing data lists.""")

parser.add_argument('--output_dir', default='./output_shape', help="""Directory to save generated volume.""")

args = parser.parse_args()

TRAIN_DIR = os.path.join(args.train_dir, args.cat_id)
OUTPUT_DIR = os.path.join(args.output_dir, args.cat_id)

if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

# The views' size
BATCH_SIZE = 12

IM_DIM = 128
VOL_DIM = 64

def inference(dataset_):
  is_train_pl = tf.placeholder(tf.bool)
  img_pl, _, = model.placeholder_inputs(BATCH_SIZE, IM_DIM, VOL_DIM)
  pred = model.get_model(img_pl, is_train_pl)
  pred = tf.sigmoid(pred)

  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True

  with tf.Session(config=config) as sess:
    model_path = os.path.join(TRAIN_DIR, "trained_models")
    ckpt = tf.train.get_checkpoint_state(model_path)
    restorer = tf.train.Saver()
    restorer.restore(sess, ckpt.model_checkpoint_path)

    test_samples = dataset_.getTestSampleSize()

    for batch_idx in range(test_samples):
      imgs, view_names = dataset_.next_test_batch(batch_idx, 1)

      feed_dict = {img_pl: imgs, is_train_pl: False}
      pred_res = sess.run(pred, feed_dict=feed_dict)

      instance_id = dataset_.getId('test', batch_idx)

      for i in range(len(view_names)):
        vol_ = pred_res[i] # (vol_dim, vol_dim, vol_dim, 1)
        name_ = view_names[i][:-4] # xx.xxx.png

        save_path = os.path.join(OUTPUT_DIR, instance_id)
        if not os.path.exists(save_path):
          os.makedirs(save_path)

        save_path_name = os.path.join(save_path, name_+".h5")
        if os.path.exists(save_path_name):
          os.remove(save_path_name)

        h5_fout = h5py.File(save_path_name)
        h5_fout.create_dataset(
                'data', data=vol_,
                compression='gzip', compression_opts=4,
                dtype='float32')
        h5_fout.close()


def main():
  tf.disable_eager_execution()
  test_dataset = dataset.Dataset(base_path=args.base_dir,
                                  cat_id=args.cat_id,
                                  data_list_path=args.data_list_path)
  inference(test_dataset)

if __name__ == '__main__':
  main()
