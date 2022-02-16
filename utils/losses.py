from skimage.metrics import structural_similarity
from cv2 import resize
from numpy import expand_dims

def ssim(y_true, y_pred):
    _true = resize(y_true, (64, 64))
    _true = expand_dims(_true, axis=-1)
    index = structural_similarity(y_true, y_pred, gaussian_weights=True, sigma=1.5,  use_sample_covariance=False)
    return 1 - index