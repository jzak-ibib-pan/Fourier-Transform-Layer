import tensorflow as tf


def ssim(y_true, y_pred):
    # TF SSIM returns value range (-1; 1]
    result = tf.image.ssim(y_true, y_pred, max_val=255)
    # range (0; 2]
    result = tf.add(result, 1)
    # range (0; 1]
    result = tf.divide(result, 2)
    # flip the values to minimize loss
    return tf.subtract(tf.cast(1, dtype=tf.float32), tf.cast(result, dtype=tf.float32))