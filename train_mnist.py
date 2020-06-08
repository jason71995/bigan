import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Concatenate,Dense,LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import numpy as np
from PIL import Image

def build_generator(image_size, latent_code_length):
    x = Input(latent_code_length)
    y = Dense(128)(x)
    y = LeakyReLU()(y)
    y = Dense(256)(y)
    y = LeakyReLU()(y)
    y = Dense(512)(y)
    y = LeakyReLU()(y)
    y = Dense(1024)(y)
    y = LeakyReLU()(y)
    y = Dense(image_size)(y)
    return Model(x, y)

def build_encoder(image_size, latent_code_length):
    x = Input(image_size)
    y = Dense(1024)(x)
    y = LeakyReLU()(y)
    y = Dense(512)(y)
    y = LeakyReLU()(y)
    y = Dense(256)(y)
    y = LeakyReLU()(y)
    y = Dense(128)(y)
    y = LeakyReLU()(y)
    y = Dense(latent_code_length)(y)
    return Model(x, y)

def build_discriminator(image_size, latent_code_length):
    x = Input(image_size)
    z = Input(latent_code_length)
    y = Concatenate()([x,z])
    y = Dense(1024)(y)
    y = LeakyReLU()(y)
    y = Dense(512)(y)
    y = LeakyReLU()(y)
    y = Dense(256)(y)
    y = LeakyReLU()(y)
    y = Dense(128)(y)
    y = LeakyReLU()(y)
    y = Dense(1,activation="sigmoid")(y)
    return Model([x, z], [y])

def build_train_step(generator, encoder, discriminator):
    g_optimizer = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9)
    e_optimizer = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9)
    d_optimizer = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9)

    @tf.function
    def train_step(real_image, real_code):
        tf.keras.backend.set_learning_phase(True)

        fake_image = generator(real_code)
        fake_code  = encoder(real_image)

        d_inputs = [tf.concat([fake_image, real_image], axis=0),
                    tf.concat([real_code, fake_code], axis=0)]
        d_preds = discriminator(d_inputs)
        pred_g, pred_e = tf.split(d_preds,num_or_size_splits=2, axis=0)

        d_loss = tf.reduce_mean(-tf.math.log(pred_g + 1e-8)) + \
                 tf.reduce_mean(-tf.math.log(1 - pred_e + 1e-8))
        g_loss = tf.reduce_mean(-tf.math.log(1 - pred_g + 1e-8))
        e_loss = tf.reduce_mean(-tf.math.log(pred_e + 1e-8))

        d_gradients = tf.gradients(d_loss, discriminator.trainable_variables)
        g_gradients = tf.gradients(g_loss, generator.trainable_variables)
        e_gradients = tf.gradients(e_loss, encoder.trainable_variables)

        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        e_optimizer.apply_gradients(zip(e_gradients, encoder.trainable_variables))

        return d_loss, g_loss, e_loss

    return train_step

def train():
    check_point = 1000
    iters = 200 * check_point
    image_size = 784
    latent_code_length = 100
    batch_size = 128

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_of_data = x_train.shape[0]
    x_train = np.reshape(x_train, (-1, image_size))
    x_train = (x_train.astype("float32") / 255) * 2 - 1

    z_train = np.random.uniform(-1.0, 1.0, (num_of_data, latent_code_length)).astype("float32")
    z_test = np.random.uniform(-1.0, 1.0, (100, latent_code_length)).astype("float32")

    # ==================== save x images ====================
    image = np.reshape(x_train[:100], (10, 10, 28, 28))
    image = np.transpose(image, (0, 2, 1, 3))
    image = np.reshape(image, (10 * 28, 10 * 28))
    image = 255 * (image + 1) / 2
    image = np.clip(image, 0, 255)
    image = image.astype("uint8")
    Image.fromarray(image, "L").save("x.png")
    # =======================================================

    generator = build_generator(image_size, latent_code_length)
    encoder = build_encoder(image_size, latent_code_length)
    discriminator = build_discriminator(image_size, latent_code_length)
    train_step = build_train_step(generator, encoder, discriminator)

    for i in range(iters):
        real_images = x_train[np.random.permutation(num_of_data)[:batch_size]]
        real_code   = z_train[np.random.permutation(num_of_data)[:batch_size]]

        d_loss, g_loss, e_loss = train_step(real_images, real_code)
        print("\r[{}/{}]  d_loss: {:.4}, g_loss: {:.4}, e_loss: {:.4}".format(i,iters, d_loss, g_loss, e_loss),end="")


        if (i+1)%check_point == 0:

            # save G(x) images
            image = generator.predict(encoder.predict(x_train[:100]))
            image = np.reshape(image, (10, 10, 28, 28))
            image = np.transpose(image, (0, 2, 1, 3))
            image = np.reshape(image, (10 * 28, 10 * 28))
            image = 255 * (image + 1) / 2
            image = np.clip(image,0,255)
            image = image.astype("uint8")
            Image.fromarray(image, "L").save("G_E_x-{}.png".format(i//check_point))

            # save G(z) images
            image = generator.predict(z_test)
            image = np.reshape(image, (10, 10, 28, 28))
            image = np.transpose(image, (0, 2, 1, 3))
            image = np.reshape(image, (10 * 28, 10 * 28))
            image = 255 * (image + 1) / 2
            image = np.clip(image,0,255)
            image = image.astype("uint8")
            Image.fromarray(image, "L").save("G_z-{}.png".format(i//check_point))

if __name__ == "__main__":
    train()