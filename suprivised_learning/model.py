import tensorflow as tf
import tensorflow_hub as hub


def create_model():
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None, input_shape=(225, 400, 4),
                                                              alpha=0.5,
                                                              include_top=False,
                                                              input_tensor=None,
                                                              pooling=None,
                                                              )
    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10, activation='sigmoid',
                              use_bias=True, name='Logits')(x)
    model = tf.keras.Model(inputs=backbone.inputs, outputs=x)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam())
    return model
if __name__ == '__main__':

    model = create_model()
    model.summary()