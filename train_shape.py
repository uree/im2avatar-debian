import tensorflow.compat.v1 as tf
import numpy as np
import os
import sys
sys.path.append('./utils')
sys.path.append('./models')

import dataset as dataset
import model_shape as model

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', default='./train_shape', help= """Directory where to write summaries and checkpoint.""")

parser.add_argument('--base_dir', default='./data/ShapeNetCore_im2avatar', help="""The path containing all the samples.""")

parser.add_argument('--cat_id', default='02958343', help= """The category id for each category: 02958343, 03001627, 03467517, 04379243""")

parser.add_argument('--data_list_path', default='./data_list', help= """The path containing data lists.""")

parser.add_argument('--train_epochs', default=501, help= """The path containing data lists.""", type=int)

parser.add_argument('--batch_size', default=30, help= """Batch size.""", type=int)

parser.add_argument('--gpu', default=0, help= """""", type=int)

parser.add_argument('--learning_rate', default=0.0003, help= """""", type=float)

parser.add_argument('--wd', default=0.00001, help= """""", type=float)

parser.add_argument('--epochs_to_save', default=20, help="""""", type=int)

parser.add_argument('--decay_step', default=2000, help="""for lr""", type=int)

parser.add_argument('--decay_rate', default=0.7, help="""for lr""", type=int)


args = parser.parse_args()

IM_DIM = 128
VOL_DIM = 64

BATCH_SIZE = args.batch_size
TRAIN_EPOCHS = args.train_epochs
GPU_INDEX = args.gpu
BASE_LEARNING_RATE = args.learning_rate
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

TRAIN_DIR = os.path.join(args.train_dir, args.cat_id)
if not os.path.exists(TRAIN_DIR):
  os.makedirs(TRAIN_DIR)
LOG_FOUT = open(os.path.join(TRAIN_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(parser.parse_args())+'\n')

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)


def get_learning_rate(batch):
  learning_rate = tf.train.exponential_decay(
                      BASE_LEARNING_RATE,  # Base learning rate.
                      batch * BATCH_SIZE,  # Current index into the dataset.
                      DECAY_STEP,          # Decay step.
                      DECAY_RATE,          # Decay rate.
                      staircase=True)
  learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
  return learning_rate


def get_bn_decay(batch):
  bn_momentum = tf.train.exponential_decay(
                    BN_INIT_DECAY,
                    batch*BATCH_SIZE,
                    BN_DECAY_DECAY_STEP,
                    BN_DECAY_DECAY_RATE,
                    staircase=True)
  bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
  return bn_decay


def train(dataset_):
  with tf.Graph().as_default():
    with tf.device('/gpu:'+str(GPU_INDEX)):
      is_train_pl = tf.placeholder(tf.bool)
      img_pl, vol_pl = model.placeholder_inputs(BATCH_SIZE, IM_DIM, VOL_DIM)

      global_step = tf.Variable(0)
      bn_decay = get_bn_decay(global_step)
      tf.summary.scalar('bn_decay', bn_decay)

      pred = model.get_model(img_pl, is_train_pl, weight_decay=args.wd, bn_decay=bn_decay)
      loss = model.get_MSFE_cross_entropy_loss(pred, vol_pl)
      tf.summary.scalar('loss', loss)

      learning_rate = get_learning_rate(global_step)
      tf.summary.scalar('learning_rate', learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      train_op = optimizer.minimize(loss, global_step=global_step)

      summary_op = tf.summary.merge_all()

      saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
      model_path = os.path.join(TRAIN_DIR, "trained_models")
      if tf.gfile.Exists(os.path.join(model_path, "checkpoint")):
        ckpt = tf.train.get_checkpoint_state(model_path)
        restorer = tf.train.Saver()
        restorer.restore(sess, ckpt.model_checkpoint_path)
        print ("Load parameters from checkpoint.")
      else:
        sess.run(tf.global_variables_initializer())

      train_summary_writer = tf.summary.FileWriter(model_path, graph=sess.graph)

      train_sample_size = dataset_.getTrainSampleSize()
      train_batches = train_sample_size // BATCH_SIZE # The number of batches per epoch

      val_sample_size = dataset_.getValSampleSize()
      val_batches = val_sample_size // BATCH_SIZE

      for epoch in range(TRAIN_EPOCHS):
        ####################
        # For training
        ####################
        dataset_.shuffleIds()
        for batch_idx in range(train_batches):
          imgs, vols_clr = dataset_.next_batch(batch_idx * BATCH_SIZE, BATCH_SIZE, vol_dim=VOL_DIM)
          vols_occu = np.prod(vols_clr > -0.5, axis=-1, keepdims=True) # (batch, vol_dim, vol_dim, vol_dim, 1)
          vols_occu = vols_occu.astype(np.float32)

          feed_dict = {img_pl: imgs, vol_pl: vols_occu, is_train_pl: True}

          step = sess.run(global_step)
          _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)

          log_string("<TRAIN> Epoch {} - Batch {}: loss: {}.".format(epoch, batch_idx, loss_val))


        #####################
        # For validation
        #####################
        loss_sum = 0.0
        for batch_idx in range(val_batches):
          imgs, vols_clr = dataset_.next_batch(batch_idx * BATCH_SIZE, BATCH_SIZE, vol_dim=VOL_DIM,  process="val")
          vols_occu = np.prod(vols_clr > -0.5, axis=-1, keepdims=True) # (batch, vol_dim, vol_dim, vol_dim, 1)
          vols_occu = vols_occu.astype(np.float32)

          feed_dict = {img_pl: imgs, vol_pl: vols_occu, is_train_pl: False}

          loss_val = sess.run(loss, feed_dict=feed_dict)
          loss_sum += loss_val
        log_string("<VAL> Epoch {}: loss: {}.".format(epoch, loss_sum/val_batches))

        #####################
        # Save model parameters.
        #####################
        if epoch % args.epochs_to_save == 0:
          checkpoint_path = os.path.join(model_path, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=epoch)


def main():
    train_dataset = dataset.Dataset(base_path=args.base_dir,
                                  cat_id=args.cat_id,
                                  data_list_path=args.data_list_path)

    train(train_dataset)

if __name__ == '__main__':
    print("--- init train_shape ---")
    main()
