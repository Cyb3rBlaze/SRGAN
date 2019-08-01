import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from tqdm import tqdm
from PIL import Image

#===========================
#LOAD AND PREPROCESS IMAGES
#===========================
def save_training_data(directory):
    list = listdir(directory)
    images = []
    size = (64, 64)

    for image_name in tqdm(list):
        image = Image.open(directory + "/" + image_name).convert("RGB").resize(size)
        image.save("training_data/high_res/" + image_name)
        image = image.resize((32, 32))
        image = image.resize((64, 64))
        image.save("training_data/low_res/" + image_name)

def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1

    return image

def load_and_preprocess_data(directory):
    low_resList = listdir(directory + "/low_res")
    high_resList = listdir(directory + "/high_res")

    low_resImages = []
    high_resImages = []

    for image in tqdm(low_resList):
        low_resImages += [normalize(np.array(Image.open(directory + "/low_res/" + image)).reshape((1, 64, 64, 3)))]
        high_resImages += [normalize(np.array(Image.open(directory + "/high_res/" + image)).reshape((1, 64, 64, 3)))]

    return low_resImages, high_resImages

#===========================
#CREATE MODELS
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

def disc_block(filters, strides):
    #initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    #result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result

def create_generator():
    input = tf.keras.layers.Input(shape=[64, 64, 3])

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


def create_discriminator():
    input = tf.keras.layers.Input(shape=[64, 64, 3])

    conv1 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same')(input)
    lrelu1 = tf.keras.layers.LeakyReLU()(conv1)

    block1 = disc_block(64, 2)(lrelu1)
    block2 = disc_block(128, 1)(block1)
    block3 = disc_block(128, 2)(block2)
    block4 = disc_block(256, 1)(block3)
    block5 = disc_block(256, 2)(block4)
    block6 = disc_block(512, 1)(block5)
    block7 = disc_block(512, 2)(block6)

    flatten = tf.keras.layers.Flatten()(block7)
    dense1 = tf.keras.layers.Dense(1024)(flatten)
    lrelu2 = tf.keras.layers.LeakyReLU()(dense1)
    dense2 = tf.keras.layers.Dense(1, activation="sigmoid")(lrelu2)

    return tf.keras.Model(inputs=input, outputs=dense2)

discriminator = create_discriminator()
#discriminator.summary()

generator = create_generator()
#generator.summary()

#===========================
#LOSS FUNCTIONS
#===========================
LAMBDA = 100

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)

    return model

content_layers = ["block5_conv2"]

vgg_model = vgg_layers(content_layers)

def generator_loss(fake_output, generated_image, target):
    gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    target_features = tf.keras.applications.vgg19.preprocess_input((target+1)*127.5)
    target_map = vgg_model(target_features)
    generated_features = tf.keras.applications.vgg19.preprocess_input((generated_image+1)*127.5)
    generated_map = vgg_model(generated_features)
    l1_loss = tf.reduce_mean(tf.losses.mean_squared_error(target_map, generated_map))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

#===========================
#CREATING MODEL OPTIMIZER OBJECTS
#===========================
def create_adam_optimizer():
    return tf.keras.optimizers.Adam(1e-4)

generator_optimizer = create_adam_optimizer()
discriminator_optimizer = create_adam_optimizer()

#===========================
#FUNCTION USED TO SAVE DATA
#===========================
def saveEpochImage(model, test_input, epoch):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig("./output/" + str(epoch) + ".jpg")

#===========================
#CHECKPOINTS TO SAVE MODEL
#===========================
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

#===========================
#TRAIN MODEL
#===========================
EPOCHS = 150

def one_train_step(input, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input, training=True)

        disc_real_output = discriminator(target, training=True)
        disc_generated_output = discriminator(gen_output, training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(low_resData, high_resData, epochs, seed):
    saveEpochImage(generator, seed, 0)

    for epoch in range(EPOCHS):
        for low, high in tqdm(zip(low_resData, high_resData)):
            one_train_step(low, high)
        print("Epoch " + str(epoch) + " finished")
        if epoch % 5 == 0:
            saveEpochImage(generator, seed, (epoch+1))
            checkpoint.save(file_prefix = checkpoint_prefix)

#===========================
#FUNCTION CALLS
#===========================
#save_training_data("./raw_data")
print("Beginning train data preprocessing...")
low_res, high_res = load_and_preprocess_data("./training_data")

print("Beginning training...")

    #===========================
    #CREATING SEED TO SEE PROGRESS
    #===========================
seed = low_res[0]

train(low_res, high_res, EPOCHS, seed)
