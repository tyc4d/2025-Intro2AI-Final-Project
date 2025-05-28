from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, RepeatVector, Conv2DTranspose, concatenate, LeakyReLU, PReLU, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, MeanAbsoluteError


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

def unet_relu_leaky(learning_rate=0.0001, loss_function_name='mse'):
    print("*****unet_relu_leaky*****")
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
    embed_input = Input(shape=(1000,))
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
    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

    if loss_function_name.lower() in ['mae', 'l1']:
        loss_func = MeanAbsoluteError()
        print(f"Compiling best_version with Mean Absolute Error (L1) loss. Learning rate: {learning_rate}")
    else:
        loss_func = MeanSquaredError()
        print(f"Compiling best_version with Mean Squared Error (L2) loss. Learning rate: {learning_rate}")

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_func, metrics=[])
    return model

def unet_advanced_prelu(learning_rate=0.0001, loss_function_name='mse'):
    print("*****unet_advanced_prelu (using PReLU)*****")
    encoder_input = Input(shape=(512, 512, 1,))

    # Encoder Path
    layers = [
        Conv2D(16, (3, 3), padding='same'), PReLU(),
        Conv2D(16, (3, 3), padding='same'), PReLU(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(32, (3, 3), padding='same'), PReLU(),
        Conv2D(32, (3, 3), padding='same'), PReLU(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(64, (3, 3), padding='same'), PReLU(),
        Conv2D(64, (3, 3), padding='same'), PReLU(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(128, (3, 3), padding='same'), PReLU(),
        Conv2D(128, (3, 3), padding='same'), PReLU(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    ]

    # Build encoder path and store skip connections
    x = encoder_input
    skip_connections = []
    for i, layer in enumerate(layers):
        x = layer(x)
        # Store outputs of Conv layers before MaxPooling for skip connections
        # In this VGG-style block, it's the output of the second Conv2D of each block
        if isinstance(layer, MaxPooling2D):
             # The output of the layer before MaxPooling was the one to skip
             # This logic needs to be more precise if we want to match U-Net exactly.
             # For U-Net, skip is usually the output of the block before pooling.
             # Let's refine this: we need to grab the output of the conv block.
             pass # Will capture skip connections more explicitly below

    # Explicitly define encoder path to capture skip connections correctly
    enc_c1 = Conv2D(16, (3, 3), padding='same')(encoder_input); enc_c1 = PReLU()(enc_c1)
    enc_c1 = Conv2D(16, (3, 3), padding='same')(enc_c1); enc_c1 = PReLU()(enc_c1)
    enc_p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(enc_c1)
    skip_connections.append(enc_c1)

    enc_c2 = Conv2D(32, (3, 3), padding='same')(enc_p1); enc_c2 = PReLU()(enc_c2)
    enc_c2 = Conv2D(32, (3, 3), padding='same')(enc_c2); enc_c2 = PReLU()(enc_c2)
    enc_p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(enc_c2)
    skip_connections.append(enc_c2)

    enc_c3 = Conv2D(64, (3, 3), padding='same')(enc_p2); enc_c3 = PReLU()(enc_c3)
    enc_c3 = Conv2D(64, (3, 3), padding='same')(enc_c3); enc_c3 = PReLU()(enc_c3)
    enc_p3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(enc_c3)
    skip_connections.append(enc_c3)

    enc_c4 = Conv2D(128, (3, 3), padding='same')(enc_p3); enc_c4 = PReLU()(enc_c4)
    enc_c4 = Conv2D(128, (3, 3), padding='same')(enc_c4); enc_c4 = PReLU()(enc_c4)
    enc_p4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(enc_c4)
    skip_connections.append(enc_c4)
    
    # Bottleneck / Middle Path
    middle_in = Conv2D(256, (3, 3), padding='same')(enc_p4); middle_in = PReLU()(middle_in)
    middle_in = Conv2D(256, (3, 3), padding='same')(middle_in); middle_in = PReLU()(middle_in)
    
    embed_input = Input(shape=(1000,))
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([middle_in, fusion_output], axis=3)
    
    fusion_output = Conv2D(512, (1, 1), padding='same')(fusion_output); fusion_output = PReLU()(fusion_output)
    middle_out = Conv2D(256, (3, 3), padding='same')(fusion_output); middle_out = PReLU()(middle_out)

    # Decoder Path
    dec_c4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(middle_out)
    dec_c4 = concatenate([dec_c4, skip_connections[3]]) # Skip connection from enc_c4
    dec_c4 = Conv2D(128, (3, 3), padding='same')(dec_c4); dec_c4 = PReLU()(dec_c4)
    dec_c4 = Conv2D(128, (3, 3), padding='same')(dec_c4); dec_c4 = PReLU()(dec_c4)

    dec_c3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(dec_c4)
    dec_c3 = concatenate([dec_c3, skip_connections[2]]) # Skip connection from enc_c3
    dec_c3 = Conv2D(64, (3, 3), padding='same')(dec_c3); dec_c3 = PReLU()(dec_c3)
    dec_c3 = Conv2D(64, (3, 3), padding='same')(dec_c3); dec_c3 = PReLU()(dec_c3)

    dec_c2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(dec_c3)
    dec_c2 = concatenate([dec_c2, skip_connections[1]]) # Skip connection from enc_c2
    dec_c2 = Conv2D(32, (3, 3), padding='same')(dec_c2); dec_c2 = PReLU()(dec_c2)
    dec_c2 = Conv2D(32, (3, 3), padding='same')(dec_c2); dec_c2 = PReLU()(dec_c2)

    dec_c1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(dec_c2)
    dec_c1 = concatenate([dec_c1, skip_connections[0]]) # Skip connection from enc_c1
    dec_c1 = Conv2D(16, (3, 3), padding='same')(dec_c1); dec_c1 = PReLU()(dec_c1)
    dec_c1 = Conv2D(16, (3, 3), padding='same')(dec_c1); dec_c1 = PReLU()(dec_c1)

    decoder_output = Conv2D(2, (1, 1), activation='tanh', padding='same')(dec_c1)
    
    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

    if loss_function_name.lower() in ['mae', 'l1']:
        loss_func = MeanAbsoluteError()
        print(f"Compiling unet_advanced_prelu with Mean Absolute Error (L1) loss. Learning rate: {learning_rate}")
    else:
        loss_func = MeanSquaredError()
        print(f"Compiling unet_advanced_prelu with Mean Squared Error (L2) loss. Learning rate: {learning_rate}")

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_func, metrics=[])
    return model

# Alias for best_version, as it was previously named unet_relu_leaky
# def best_version(learning_rate=0.0001, loss_function_name='mse'):
# return unet_relu_leaky(learning_rate=learning_rate, loss_function_name=loss_function_name)


def build_discriminator(input_shape=(512, 512, 3)):
    """
    建立一個 PatchGAN 判別器模型。
    輸入是 (L + ab) 通道圖像。
    輸出是一個 30x30x1 的 patch，其中每個值代表對應圖像區域為真實的機率。
    """
    print("*****Building Discriminator (PatchGAN)*****")
    init = 'glorot_uniform' # Kernel initializer

    # 輸入 L + ab 通道 (512x512x3)
    in_src_image = Input(shape=input_shape) # L channel
    # in_target_image = Input(shape=(input_shape[0], input_shape[1], 2)) # ab channels
    # merged = concatenate([in_src_image, in_target_image]) # Concatenate L and ab

    # C64: 4x4 stride 2x2
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_src_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128: 4x4 stride 2x2
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256: 4x4 stride 2x2
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512: 4x4 stride 1x1 (stride was 2x2 in some implementations, but for 30x30 output from 256, maybe 1x1 or adjust padding)
    # Let's adjust stride to 1 for the last two conv layers before the patch output, or adjust padding.
    # Given 512 -> 256 -> 128 -> 64. If next is stride 2 -> 32. Patch output usually 30x30 or 16x16.
    
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d) # Stride 1 after 64x64
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    # Second last layer:
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Patch output
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init, activation='sigmoid')(d) # Output is a 64x64 patch prediction
    
    # Define model
    model = Model(in_src_image, patch_out)
    # Compile model with Adam optimizer and binary cross-entropy loss
    # Learning rate for discriminator is often different from generator
    # model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model

def define_gan(g_model, d_model, image_shape_l=(512,512,1), embed_dim=1000):
    """
    定義組合的 GAN 模型，用於訓練生成器。
    判別器的權重在此模型中設定為不可訓練。
    """
    print("*****Defining Combined GAN model*****")
    # Make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization): # BN layers should be trainable
            layer.trainable = False

    # Generator input (L channel and embedding)
    gen_input_l = Input(shape=image_shape_l, name="gan_gen_input_l")
    gen_input_embed = Input(shape=(embed_dim,), name="gan_gen_input_embed")
    
    # Generator output (ab channels)
    # This is the actual output from the generator model G
    gen_output_ab = g_model([gen_input_l, gen_input_embed]) # This is 512x512x2
    
    # Concatenate L channel (from generator input) with generated ab channels
    # This forms the input for the discriminator D
    gan_input_for_discriminator = concatenate([gen_input_l, gen_output_ab], name="gan_concat_for_disc") # Should be 512x512x3
        
    # Discriminator output (the "adversarial" part)
    # This is D(G(L, embed))
    discriminator_output_on_fake = d_model(gan_input_for_discriminator)
    
    # Define GAN model: inputs are G's inputs, outputs are [D's output on fake, G's direct output]
    # This allows us to have two losses: one for fooling D, one for L1 reconstruction on G's output.
    model = Model(
        inputs=[gen_input_l, gen_input_embed], 
        outputs=[discriminator_output_on_fake, gen_output_ab], 
        name="gan_model"
    )
    
    # Compile model - this will be done in train.py with appropriate optimizers and loss weights
    # Example (actual compilation in train.py):
    # opt_gan = Adam(learning_rate=0.0002, beta_1=0.5)
    # model.compile(loss=['binary_crossentropy', 'mae'], 
    #               loss_weights=[1, 100], # Example: adversarial_weight=1, l1_weight=100
    #               optimizer=opt_gan)
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
    # discriminator = build_discriminator(input_shape=(512, 512, 3))
    # discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    # discriminator.summary()
    
    # Test Combined GAN
    # gan_model = define_gan(generator, discriminator)
    # gan_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5)) # Adversarial loss
    # gan_model.summary()
    print("GAN components (Generator, Discriminator, Combined GAN) can be defined.")
    print("Actual compilation with optimizers and losses will be handled in the training script.")