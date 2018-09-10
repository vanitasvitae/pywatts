import tensorflow as tf
import subprocess

writer = tf.summary.FileWriter("tensorboard")
checkpoint = tf.train.get_checkpoint_state('tf_pywatts_model_best')
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
    saver.restore(sess, checkpoint.model_checkpoint_path)
writer.add_graph(sess.graph)

subprocess.check_output(['tensorboard', '--logdir', 'tensorboard'])