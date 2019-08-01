import tensorflow as tf

#===========================
#CREATING NETWORK ARCHITECTURE TO LOAD WEIGHTS
#===========================
def res_block(model, filters, strides):
    gen = model

    model = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)

    model = tf.keras.layers.add([gen, model])

    return model

def up_sampling_block(model, filters, strides):
    model = tf.keras.layers.Conv2DTranspose(filters, 3, strides=strides, padding="same")(model)
    model = tf.keras.layers.LeakyReLU()(model)

    return model

def create_generator():
    input = tf.keras.layers.Input(shape=[256, 256, 3])

    conv1 = tf.keras.layers.Conv2D(64, 9, strides=1, padding="same")(input)
    #prelu1 = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(conv1)
    prelu1 = tf.keras.layers.PReLU()(conv1)

    gen_model = prelu1

    block1 = res_block(prelu1, 64, 1)
    block2 = res_block(block1, 64, 1)
    block3 = res_block(block2, 64, 1)
    block4 = res_block(block3, 64, 1)
    block5 = res_block(block4, 64, 1)

    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(block5)
    batch1 = tf.keras.layers.BatchNormalization(momentum = 0.5)(conv2)
    skip1 = tf.keras.layers.add([gen_model, batch1])

    block6 = up_sampling_block(skip1, 256, 1)
    block7 = up_sampling_block(block6, 256, 1)

    last = tf.keras.layers.Conv2D(3, 9, strides=1, padding="same", activation="tanh")(block7)

    return tf.keras.Model(inputs=input, outputs=last)

print("Creating generator model...")
generator = create_generator()
print("Loading model weights...")
generator.load_weights("")
print("Predicting...")
#TODO implement image input pipeline
