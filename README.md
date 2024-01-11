# Fourier-Transform-Layer
Fourier Tranform Layer for fast image processing in Neural Networks. 
This Layer performs FFT and optional iFFT on images, with weights being multiplied with Fourier spectrum. 
The code was developed in Python 3.8 and 3.9, Tensorflow 2.5. 
To run the code of published works, please refer to corresponding READMEs. 

# Input parameters of the Layer
Parameters are presented in a specific format: name [type : default value] - description
+ activation [str : None] - as any activation in Keras (https://keras.io/api/layers/activations/). Implemented activations: relu, softmax, sigmoid, tanh, selu. 
+ kernel_initializer [str : 'he_normal'] - as any kernel initializer in Keras (https://keras.io/api/layers/initializers/). Implemented all Keras initializers. 
+ train_imaginary [bool : True] - whether to train the imaginary part of Fourier spectrum. If True, will double the number of weights and return two arrays unless calculate_abs is True.
+ inverse [bool : False] - whether to inverse the Fourier spectrum at the end. If True, will return two arrays unless calculate_abs is True. 
+ use_bias [bool: False] - as in Keras Layer.
+ bias_initializer [str : 'zeros'] - as any bias initializer in Keras (https://keras.io/api/layers/initializers/). Implemented all Keras initializers. 
+ calculate_abs [bool : True] - whether to return absolute of complex number. If True, will return one array. If False, will return two arrays.
+ normalize_to_image_shape [bool : False] - whether to normalize the Fourier spectrum to image shape (https://stackoverflow.com/questions/20165193/fft-normalization). Planned further normalizations, according to https://pyfar.readthedocs.io/en/stable/concepts/pyfar.fft.html. 
+ already_fft [bool : False] - whether the input is the result of previous FFT. This way, the layer expects two arrays as input. This allows stacking FTLs.

# Please cite as: 
@article{ZAK2023110607,
title = {Fourier Transform Layer: A proof of work in different training scenarios},
journal = {Applied Soft Computing},
volume = {145},
pages = {110607},
year = {2023},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2023.110607},
url = {https://www.sciencedirect.com/science/article/pii/S1568494623006257},
author = {Jakub Zak and Anna Korzynska and Antonina Pater and Lukasz Roszkowiak},
keywords = {Neural networks, Convolutional neural networks, Fourier transform, Image classification},
abstract = {In this work two established methods were merged: Fourier Transform and Convolutional Neural Network to classify images in several datasets. Fourier Transform is commonly used in signal processing as well as in image processing. Convolutional Neural Networks are well-suited to image analysis tasks, yet require a lot of processing power and/or time during training process. Fourier Transform Layer is introduced to increase the processing speed without sacrificing accuracy. The motivation is to present and compare an alternative approach to Convolutional Neural Networks, which could reduce the need for GPU training. Models containing only the novel layer were trained to classify images from widely accepted datasets and compared to classification results of simple models containing one convolutional layer. The comparison was performed in terms of test accuracy, Area-Under-Curve and training times. The results showed that, for images of size 128 × 128 and larger, models with the proposed layer reached test accuracy comparable to that reached by convolutional models (accuracy: 96% and 98% respectively), with at least 27% decrease in training time per one epoch on Central Processing Unit.}
}
