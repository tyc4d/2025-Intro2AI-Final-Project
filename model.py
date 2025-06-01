from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, RepeatVector, Conv2DTranspose, concatenate, LeakyReLU, PReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

#keras mixed precision
from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')


def unet_advanced_prelu(learning_rate=0.0001, loss_function_name='mse'):
    print("*****unet_advanced_prelu (using PReLU)*****")
    encoder_input = Input(shape=(512, 512, 1,))

    # Encoder Path
    enc_c1 = Conv2D(16, (3, 3), padding='same')(encoder_input); enc_c1 = PReLU()(enc_c1)
    enc_c1 = Conv2D(16, (3, 3), padding='same')(enc_c1); enc_c1 = PReLU()(enc_c1)
    enc_p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(enc_c1)

    enc_c2 = Conv2D(32, (3, 3), padding='same')(enc_p1); enc_c2 = PReLU()(enc_c2)
    enc_c2 = Conv2D(32, (3, 3), padding='same')(enc_c2); enc_c2 = PReLU()(enc_c2)
    enc_p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(enc_c2)

    enc_c3 = Conv2D(64, (3, 3), padding='same')(enc_p2); enc_c3 = PReLU()(enc_c3)
    enc_c3 = Conv2D(64, (3, 3), padding='same')(enc_c3); enc_c3 = PReLU()(enc_c3)
    enc_p3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(enc_c3)

    enc_c4 = Conv2D(128, (3, 3), padding='same')(enc_p3); enc_c4 = PReLU()(enc_c4)
    enc_c4 = Conv2D(128, (3, 3), padding='same')(enc_c4); enc_c4 = PReLU()(enc_c4)
    enc_p4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(enc_c4)
    
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
    # Skip connections are taken from the output of the second Conv2D layer in each encoder block (before PReLU and MaxPooling)
    # However, the original U-Net paper suggests skip connections from the feature maps of the convolutional layers.
    # The provided code structure for skip connections in unet_advanced_prelu was:
    # skip_connections.append(enc_c1) where enc_c1 is after PReLU. Let's stick to the provided logic.

    dec_c4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(middle_out)
    dec_c4 = concatenate([dec_c4, enc_c4]) # Skip connection from enc_c4 (output of PReLU block)
    dec_c4 = Conv2D(128, (3, 3), padding='same')(dec_c4); dec_c4 = PReLU()(dec_c4)
    dec_c4 = Conv2D(128, (3, 3), padding='same')(dec_c4); dec_c4 = PReLU()(dec_c4)

    dec_c3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(dec_c4)
    dec_c3 = concatenate([dec_c3, enc_c3]) # Skip connection from enc_c3
    dec_c3 = Conv2D(64, (3, 3), padding='same')(dec_c3); dec_c3 = PReLU()(dec_c3)
    dec_c3 = Conv2D(64, (3, 3), padding='same')(dec_c3); dec_c3 = PReLU()(dec_c3)

    dec_c2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(dec_c3)
    dec_c2 = concatenate([dec_c2, enc_c2]) # Skip connection from enc_c2
    dec_c2 = Conv2D(32, (3, 3), padding='same')(dec_c2); dec_c2 = PReLU()(dec_c2)
    dec_c2 = Conv2D(32, (3, 3), padding='same')(dec_c2); dec_c2 = PReLU()(dec_c2)

    dec_c1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(dec_c2)
    dec_c1 = concatenate([dec_c1, enc_c1]) # Skip connection from enc_c1
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

def build_discriminator(input_shape=(512, 512, 3)):
    print("*****Building Discriminator (PatchGAN)*****")
    init = 'glorot_uniform' 

    in_src_image = Input(shape=input_shape) 

    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_src_image)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d) 
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init, activation='sigmoid')(d) 
    
    model = Model(in_src_image, patch_out)
    # Discriminator compilation (e.g., learning rate) is typically handled in the training script
    # model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model

def define_gan(g_model, d_model, image_shape_l=(512,512,1), embed_dim=1000):
    print("*****Defining Combined GAN model*****")
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization): 
            layer.trainable = False

    gen_input_l = Input(shape=image_shape_l, name="gan_gen_input_l")
    gen_input_embed = Input(shape=(embed_dim,), name="gan_gen_input_embed")
    
    gen_output_ab = g_model([gen_input_l, gen_input_embed]) 
    
    gan_input_for_discriminator = concatenate([gen_input_l, gen_output_ab], name="gan_concat_for_disc")
        
    discriminator_output_on_fake = d_model(gan_input_for_discriminator)
    
    model = Model(
        inputs=[gen_input_l, gen_input_embed], 
        outputs=[discriminator_output_on_fake, gen_output_ab], 
        name="gan_model"
    )
    # GAN compilation (optimizer, losses, loss_weights) is typically handled in the training script
    return model 