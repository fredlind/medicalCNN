import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

def test():
    print('networks are in!')
## Functions for constructing layer blocks in the CNN:
def input_concatenation(height, width, channels, n_data):
    # Input layer:
    inputs = []
    name_base = 'input_'
    for i_in in range(n_data):
        name = name_base + str(i_in)
        inputs.append(Input(shape=(height, width, channels), name=name))

    input = Concatenate(axis=-1)(inputs)

    return input, inputs

def initial_convolution(out_channels, inputs, activation='relu', kernel_initializer='he_normal'):
    
    # First, special convolutional layer, same as convolution block TODO: Remove
    conv1 = Conv2D(out_channels, 3, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(out_channels, 3, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv1)
    conv1 = BatchNormalization()(conv1)
    
    return conv1

def convolution_block(out_channels, prev_layer, activation='relu', kernel_initializer='he_normal', 
                      dropout=0.0):
     # First standard convolutional layer
    conv = Conv2D(out_channels, 3, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(prev_layer)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout)(conv)
    conv = Conv2D(out_channels, 3, activation=activation, padding='same',
                   kernel_initializer=kernel_initializer)(conv)
    conv = BatchNormalization()(conv)
    
    return conv


def build_model(height, width, channels, n_data=1,
                                         filtin = 2,  
                                         filt = 2,
                                         short_connect=True, 
                                         depth = 4, 
                                         activation='relu',
                                         kernel_initializer='he_normal',
                                         dropout=0.0):
    """Constructs the model. TODO: Change name"""
    print(filt, short_connect, depth, activation, kernel_initializer, dropout)
    no_up_blocks = depth - 1
    # The width of the convolution blocks are constructed
    # of the form filt * (2, 4, 8, ...):
    down_block_size = []
    for i_block in range(depth):
        down_block_size.append(filt * 2 ** (i_block + 1))
    
    long_connections = []
 
    input, inputs  = input_concatenation(height, width, channels, n_data)
    long_connections.append(input)
    short_connection = input
    conv1 = initial_convolution(filtin, input)
    
    # Concatentation
    if short_connect:
        in_net = Concatenate(axis=-1)([short_connection, conv1])
    else: in_net = conv1
    # Encoder: 
    for i_block in range(depth):
        #Downsampling:
        #pool = MaxPooling2D(pool_size=(2, 2))(in_net)
        pool = Conv2D((1+ i_block) * 8 - 1, 3 , strides=(2,2), padding='same')(in_net)
        #Convolutional block
        conv = convolution_block(down_block_size[i_block], pool, activation=activation,
                                 kernel_initializer=kernel_initializer, dropout=dropout)
    
        # Concatentation
        if short_connect: #If short connect is true we concatenate the input of
            #the convolutional block with its output:
            in_net = Concatenate(axis=-1)([pool, conv])
        else: in_net = conv
        
        if i_block < depth - 1: # Storing output for constructing skip connections to decoder
            long_connections.append(conv)


    ## Decoder:
    for i_block in range(no_up_blocks):
        # Upsampling:
        up = UpSampling2D(size=(2,2))(in_net) 
        # Creating skip connections:
        in_net = Concatenate(axis=-1)([up, long_connections[depth - i_block - 1]])
        # Convolutional block:
        conv = convolution_block(down_block_size[i_block], in_net, 
                                 activation=activation, 
                                 kernel_initializer=kernel_initializer,
                                 dropout=dropout)
        if short_connect: #If short connect is true we concatenate the input of
            #the convolutional block with its output:
            in_net = Concatenate(axis=-1)([up, conv])
        else: in_net = conv
    
    # Final upsampling/concatenation:
    up = UpSampling2D(size=(2,2))(in_net)
    final = Concatenate(axis=-1)([up, long_connections[0]])
                                       
    output = Conv2D(1, 1, activation='sigmoid')(final) # Sigmoid activation will have output values [0, 1]
    
    
    return Model(inputs=inputs, outputs=[output])




# Build your model.
def do_the_building(gen_train, depth, conv_channels, activation, kernel_initializer,
                    short_connect=False, dropout=0.0, filtin=2):
    """ Function that constructs model. TODO: remove"""
    height, width, channels = gen_train.in_dims[0][1:]
    model = build_model(height=height, width=width, channels=channels, short_connect=short_connect,
                        depth = depth, filt=conv_channels, activation=activation,
                        kernel_initializer=kernel_initializer, dropout=dropout, filtin=filtin)
    #model.summary()
    return model
