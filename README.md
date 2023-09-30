# Fourier-Transform-Layer
Fourier Tranform Layer for fast image processing in Neural Networks
Please cite as:

Jakub Zak, Anna Korzynska, Antonina Pater, Lukasz Roszkowiak,
Fourier Transform Layer: A proof of work in different training scenarios,
Applied Soft Computing,
Volume 145,
2023,
110607,
ISSN 1568-4946,
https://doi.org/10.1016/j.asoc.2023.110607.
(https://www.sciencedirect.com/science/article/pii/S1568494623006257)
Abstract: In this work two established methods were merged: Fourier Transform and Convolutional Neural Network to classify images in several datasets. Fourier Transform is commonly used in signal processing as well as in image processing. Convolutional Neural Networks are well-suited to image analysis tasks, yet require a lot of processing power and/or time during training process. Fourier Transform Layer is introduced to increase the processing speed without sacrificing accuracy. The motivation is to present and compare an alternative approach to Convolutional Neural Networks, which could reduce the need for GPU training. Models containing only the novel layer were trained to classify images from widely accepted datasets and compared to classification results of simple models containing one convolutional layer. The comparison was performed in terms of test accuracy, Area-Under-Curve and training times. The results showed that, for images of size 128 × 128 and larger, models with the proposed layer reached test accuracy comparable to that reached by convolutional models (accuracy: 96% and 98% respectively), with at least 27% decrease in training time per one epoch on Central Processing Unit.
Keywords: Neural networks; Convolutional neural networks; Fourier transform; Image classification
