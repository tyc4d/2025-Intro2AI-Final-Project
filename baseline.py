from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, RepeatVector, Conv2DTranspose, concatenate, LeakyReLU, PReLU, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

#keras mixed precision
from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')


def unet_vgg16(learning_rate=0.0001, loss_function_name='mse'):
    print("*****unet_vgg16*****")
    encoder_input = Input(shape=(512, 512, 1,))

    encoder_c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_input)
    encoder_c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_c1)
    encoder_p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c1)

    encoder_c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_p1)
    encoder_c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_c2)
    encoder_p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c2)

    encoder_c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_p2)
    encoder_c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_c3)
    encoder_p3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c3)

    encoder_c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_p3)
    encoder_c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_c4)
    encoder_p4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c4)

    middle_in = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_p4)
    middle_in = Conv2D(256, (3, 3), activation='relu', padding='same')(middle_in)
    embed_input = Input(shape=(1000,))
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([middle_in, fusion_output], axis=3)
    fusion_output = Conv2D(512, (1, 1), activation='relu', padding='same')(fusion_output)
    middle_out = Conv2D(256, (3, 3), activation='relu', padding='same')(fusion_output)

    decoder_c4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(middle_out)
    decoder_c4 = concatenate([decoder_c4, encoder_c4])
    decoder_c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_c4)
    decoder_c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_c4)

    decoder_c3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(decoder_c4)
    decoder_c3 = concatenate([decoder_c3, encoder_c3])
    decoder_c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_c3)
    decoder_c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_c3)

    decoder_c2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(decoder_c3)
    decoder_c2 = concatenate([decoder_c2, encoder_c2])
    decoder_c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_c2)
    decoder_c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_c2)

    decoder_c1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(decoder_c2)
    decoder_c1 = concatenate([decoder_c1, encoder_c1])
    decoder_c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_c1)
    decoder_c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_c1)

    decoder_output = Conv2D(2, (1, 1), activation='tanh', padding='same')(decoder_c1)
    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    
    if loss_function_name.lower() in ['mae', 'l1']:
        loss_func = MeanAbsoluteError()
        print(f"Compiling unet_vgg16 with Mean Absolute Error (L1) loss. Learning rate: {learning_rate}")
    else:
        loss_func = MeanSquaredError()
        print(f"Compiling unet_vgg16 with Mean Squared Error (L2) loss. Learning rate: {learning_rate}")
        
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_func, metrics=[])
    return model

def unet_relu_leaky(learning_rate=0.0001, loss_function_name='mse', use_zero_embedding=True):
    print("*****unet_relu_leaky*****")
    use_zero_embedding=True
    encoder_input = Input(shape=(512, 512, 1,))

    encoder_c1 = Conv2D(16, (3, 3), padding='same')(encoder_input)
    encoder_c1 = LeakyReLU(alpha=0.2)(encoder_c1)
    encoder_c1 = Conv2D(16, (3, 3), padding='same')(encoder_c1)
    encoder_c1 = LeakyReLU(alpha=0.2)(encoder_c1)
    encoder_p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c1)

    encoder_c2 = Conv2D(32, (3, 3), padding='same')(encoder_p1)
    encoder_c2 = LeakyReLU(alpha=0.2)(encoder_c2)
    encoder_c2 = Conv2D(32, (3, 3), padding='same')(encoder_c2)
    encoder_c2 = LeakyReLU(alpha=0.2)(encoder_c2)
    encoder_p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c2)

    encoder_c3 = Conv2D(64, (3, 3), padding='same')(encoder_p2)
    encoder_c3 = LeakyReLU(alpha=0.2)(encoder_c3)
    encoder_c3 = Conv2D(64, (3, 3), padding='same')(encoder_c3)
    encoder_c3 = LeakyReLU(alpha=0.2)(encoder_c3)
    encoder_p3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c3)

    encoder_c4 = Conv2D(128, (3, 3), padding='same')(encoder_p3)
    encoder_c4 = LeakyReLU(alpha=0.2)(encoder_c4)
    encoder_c4 = Conv2D(128, (3, 3), padding='same')(encoder_c4)
    encoder_c4 = LeakyReLU(alpha=0.2)(encoder_c4)
    encoder_p4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_c4)

    middle_in = Conv2D(256, (3, 3), padding='same')(encoder_p4)
    middle_in = LeakyReLU(alpha=0.2)(middle_in)
    middle_in = Conv2D(256, (3, 3), padding='same')(middle_in)
    middle_in = LeakyReLU(alpha=0.2)(middle_in)
    embed_input_original = Input(shape=(1000,))

    if use_zero_embedding:
        # Create a zero tensor with the same shape as embed_input_original
        # Note: Keras functional API requires tensors. We use a Lambda layer to create a constant tensor.
        # The Lambda layer will take the original embed_input but ignore it and output zeros.
        embed_input = Lambda(lambda x: x * 0)(embed_input_original)
        print("Using zero embedding for unet_relu_leaky.")
    else:
        embed_input = embed_input_original
        print("Using standard embedding for unet_relu_leaky.")

    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([middle_in, fusion_output], axis=3)
    fusion_output = Conv2D(512, (1, 1), padding='same')(fusion_output)
    fusion_output = LeakyReLU(alpha=0.2)(fusion_output)
    middle_out = Conv2D(256, (3, 3), padding='same')(fusion_output)
    middle_out = LeakyReLU(alpha=0.2)(middle_out)

    decoder_c4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(middle_out)
    decoder_c4 = concatenate([decoder_c4, encoder_c4])
    decoder_c4 = Conv2D(128, (3, 3), padding='same')(decoder_c4)
    decoder_c4 = LeakyReLU(alpha=0.2)(decoder_c4)
    decoder_c4 = Conv2D(128, (3, 3), padding='same')(decoder_c4)
    decoder_c4 = LeakyReLU(alpha=0.2)(decoder_c4)

    decoder_c3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(decoder_c4)
    decoder_c3 = concatenate([decoder_c3, encoder_c3])
    decoder_c3 = Conv2D(64, (3, 3), padding='same')(decoder_c3)
    decoder_c3 = LeakyReLU(alpha=0.2)(decoder_c3)
    decoder_c3 = Conv2D(64, (3, 3), padding='same')(decoder_c3)
    decoder_c3 = LeakyReLU(alpha=0.2)(decoder_c3)

    decoder_c2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(decoder_c3)
    decoder_c2 = concatenate([decoder_c2, encoder_c2])
    decoder_c2 = Conv2D(32, (3, 3), padding='same')(decoder_c2)
    decoder_c2 = LeakyReLU(alpha=0.2)(decoder_c2)
    decoder_c2 = Conv2D(32, (3, 3), padding='same')(decoder_c2)
    decoder_c2 = LeakyReLU(alpha=0.2)(decoder_c2)

    decoder_c1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(decoder_c2)
    decoder_c1 = concatenate([decoder_c1, encoder_c1])
    decoder_c1 = Conv2D(16, (3, 3), padding='same')(decoder_c1)
    decoder_c1 = LeakyReLU(alpha=0.2)(decoder_c1)
    decoder_c1 = Conv2D(16, (3, 3), padding='same')(decoder_c1)
    decoder_c1 = LeakyReLU(alpha=0.2)(decoder_c1)

    decoder_output = Conv2D(2, (1, 1), activation='tanh', padding='same')(decoder_c1)
    
    model = Model(inputs=[encoder_input, embed_input_original], outputs=decoder_output)

    if loss_function_name.lower() in ['mae', 'l1']:
        loss_func = MeanAbsoluteError()
        print(f"Compiling best_version with Mean Absolute Error (L1) loss. Learning rate: {learning_rate}")
    else:
        loss_func = MeanSquaredError()
        print(f"Compiling best_version with Mean Squared Error (L2) loss. Learning rate: {learning_rate}")

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_func, metrics=[])
    return model

# Example usage (for testing purposes, will be integrated into train.py)
if __name__ == '__main__':
    # Test Generator (using one of the existing U-Net models)
    # Make sure the generator's output shape is (batch, 512, 512, 2)
    # And its inputs are (batch, 512, 512, 1) and (batch, 1000)
    
    # generator = unet_relu_leaky() # Or any other U-Net generator
    # generator.summary()

    # Test Discriminator
    # Input to discriminator will be (batch, 512, 512, 3) -> (L, a, b)
    # from model import build_discriminator # Example of how to import if needed here
    # discriminator = build_discriminator(input_shape=(512, 512, 3))
    # discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    # discriminator.summary()
    
    # Test Combined GAN
    # from model import define_gan # Example of how to import if needed here
    # gan_model = define_gan(generator, discriminator)
    # gan_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5)) # Adversarial loss
    # gan_model.summary()
    print("GAN components (Generator, Discriminator, Combined GAN) can be defined in model.py.")
    print("Actual compilation with optimizers and losses will be handled in the training script.")
    print("unet_vgg16 and unet_relu_leaky remain in baseline.py.")