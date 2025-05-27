from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, RepeatVector, Conv2DTranspose, concatenate, LeakyReLU, PReLU
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
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError(), metrics=[])
    return model

def best_version(learning_rate=0.0001):
    print("*****best_version*****")
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
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError(), metrics=[])
    return model

def unet_advanced_prelu(learning_rate=0.0001):
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
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError(), metrics=[])
    return model