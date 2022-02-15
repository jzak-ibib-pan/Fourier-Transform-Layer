from utils.data_loader import prepare_data_for_sampling


def test_minors():
    builder = FourierBuilder(model_type='fourier', input_shape=(32, 32, 3), noof_classes=10,
                              filename='temp', filepath='../temp')
    # builder = CNNBuilder(model_type='mobilenet', input_shape=(32, 32, 3), noof_classes=10, weights='imagenet', freeze=5,
    #                      filename='temp', filepath='../temp')
    # layers = [{'ftl': {}}, {'flatten': {}}, {'dense': {'units': 128}}, {'dense': {}}]
    # builder = CustomBuilder(layers, input_shape=(32, 32, 3), noof_classes=10,
    #                           filename='temp', filepath='../temp')
    builder.compile_model('adam', 'categorical_crossentropy', metrics=[CategoricalAccuracy(),
                                                                       TopKCategoricalAccuracy(k=5, name='top-5')])
    builder.train_model(2, x_data=x_train, y_data=y_train, batch=128, validation_split=0.1,
                        call_stop=True, call_time=True, call_checkpoint=True,
                        call_stop_kwargs={'baseline': 0.5,
                                          'monitor': 'categorical_accuracy',
                                          'patience': 3,
                                          },
                        call_checkpoint_kwargs={'monitor': 'categorical_accuracy',
                                                }, save_memory=True
                        )
    builder.evaluate_model(x_data=x_test, y_data=y_test)
    builder.save_model_info('Testing training pipeline', summary=True)


def test_sampling():
    from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
    from utils.data_loader import prepare_data_for_sampling
    data_channels = 1
    classes = [1, 3]
    noof_classes = len(classes)
    (x_train, y_train), (x_test, y_test), x_test_resized =  prepare_data_for_sampling(classes, data_channels, 64)

    builder = FourierBuilder(model_type='fourier', input_shape=(32, 32, data_channels), noof_classes=noof_classes,
                              filename='temp', filepath='../temp')
    # builder = CNNBuilder(model_type='mobilenet', input_shape=(32, 32, 3), noof_classes=10, weights='imagenet', freeze=5,
    #                      filename='temp', filepath='../temp')
    layers = [{'ftl': {'kernel_initializer': 'glorot_uniform', 'activation': 'relu', 'use_bias': False}},
              # {'conv2d': {'filters': 256, 'activation': 'relu', 'padding': 'valid'}},
              {'flatten': {}},
              {'dense': {'units': noof_classes, 'kernel_initializer': 'glorot_uniform'}}]
    builder = CustomBuilder(layers, input_shape=(32, 32, data_channels), noof_classes=noof_classes,
                              filename='temp', filepath='../temp')
    builder.compile_model('adam', 'categorical_crossentropy', metrics=[CategoricalAccuracy(),
                                                                       TopKCategoricalAccuracy(k=5, name='top-5')])
    builder.train_model(100, x_data=x_train, y_data=y_train, batch=16, validation_split=0.1,
                        call_stop=True, call_time=True, call_checkpoint=True,
                        call_stop_kwargs={'baseline': 0.90,
                                          'monitor': 'val_categorical_accuracy',
                                          'patience': 2,
                                          },
                        call_checkpoint_kwargs={'monitor': 'val_categorical_accuracy',
                                                }, save_memory=True
                        )
    builder.evaluate_model(x_data=x_test, y_data=y_test)
    builder.save_model_info(f'Trained model. Classes {classes}', summary=True)

    builder_comparison = CustomBuilder(layers, input_shape=(64, 64, data_channels), noof_classes=noof_classes,
                                       filename='temp', filepath='../temp')
    # builder_comparison = FourierBuilder(model_type='fourier', input_shape=(64, 64, data_channels), noof_classes=noof_classes,
    #                           filename='temp', filepath='../temp')
    builder_comparison.compile_model('adam', 'categorical_crossentropy', metrics=[CategoricalAccuracy(),
                                                                       TopKCategoricalAccuracy(k=5, name='top-5')])
    builder_comparison.evaluate_model(x_data=x_test_resized, y_data=y_test)
    builder_comparison.save_model_info(f'Non-trained upsampled model. Classes {classes}', summary=True)

    sampled = builder.sample_model(shape=(64, 64), compile=True, replace_value=1e-9)
    sampled.evaluate_model(x_data=x_test_resized, y_data=y_test)
    sampled.save_model_info(f'Trained upsampled model. Classes {classes}', summary=True)
