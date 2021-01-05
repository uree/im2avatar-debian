import tensorflow.compat.v1 as tf
import numpy as np
import os
import h5py
import sys
sys.path.append('./utils')
sys.path.append('./models')

import model_shape as model

import argparse

import img_utils

parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', default='./train_shape', help="""Directory where to write summaries and checkpoint.""")

parser.add_argument('--base_dir', default='./data/ShapeNetCore_im2avatar', help="""The path containing all the samples.""")

parser.add_argument('--cat_id', default='02958343', help="""The category id for each category: 02958343, 03001627, 03467517, 04379243""")

parser.add_argument('--data_list_path', default='./data_list', help="""The path containing data lists.""")

parser.add_argument('--output_dir', default='./output_shape', help="""Directory to save generated volume.""")

parser.add_argument('--prediction_input', default='', help="""Path to image.""")

args = parser.parse_args()

TRAIN_DIR = os.path.join(args.train_dir, args.cat_id)
OUTPUT_DIR = os.path.join(args.output_dir, args.cat_id)

if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

# The views' size
BATCH_SIZE = 1

IM_DIM = 128
VOL_DIM = 64

def inference(input_image, save_folder):

    print(input_image)
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

        imgs = []
        imgs.append(img_utils.imread(input_image))
        print(imgs)
        print(imgs[0].shape)

        feed_dict = {img_pl: imgs, is_train_pl: False}
        pred_res = sess.run(pred, feed_dict=feed_dict)

        name = input_image.split("/")[-1].split(".")[0]
        # print(name)
        save_path_name = os.path.join(save_folder, name+".h5")

        h5_fout = h5py.File(save_path_name, "a")
        h5_fout.create_dataset(
                'data', data=pred_res,
                compression='gzip', compression_opts=4,
                dtype='float32')
        h5_fout.close()
        print("<Done> h5 saved to ", save_path_name)
        return save_path_name

def main():
    tf.disable_eager_execution()

if __name__ == '__main__':
    main()
