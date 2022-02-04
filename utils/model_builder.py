from tensorflow.keras.models import Model


class ModelBuilder(Model):
    def __init__(self):
        super(ModelBuilder, self).__init__(self)

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass

    def _save_model_info(self):
        return self.input_shape