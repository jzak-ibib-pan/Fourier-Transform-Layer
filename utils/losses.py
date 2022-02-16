import tensorflow as tf

def ssim(y_true, y_pred):
    return 1 - tf.image.ssim(y_true, y_pred, max_val=255)