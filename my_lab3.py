#!/usr/bin/env python
# coding: utf-8

# # Assingment 3 - Generative Models
# ### Course: Convolutional Neural Networks with Applications in Medical Image Analysis
# 
# Office hours: Minh - Thursdays (odd weeks) 13.15--16.00, Attila - Wednesdays (even weeks) 08.15--12.00, Tommy - Monday May 31, 13.15-15.00
# 
# Hello again! The third assignment will have the same task as the second one, except now you have to reach the requirements using a generative model. Below is an example of a generative adversarial network (GAN) for the segmentation task to help you get started, but feel free to re-use your code from Assignment 2 or to build a variational autoencoder instead.
# 
# The third assignment is also based on the BraTS Challenge (http://braintumorsegmentation.org/), containing MRI slices of the brain, of different contrasts (sometimes referred to as modalities): T1-weighted (T1w), T1-weighted with contrast agent (T1w-CE), T2-weighted (T2w), FLAIR, and also a manually segmented binary map of a tumor, if visible on the slice. You are called to address one of the two sub-tasks from before:
# 
# - Image segmentation: to produce segmentation labels of the tumor and non-tumor regions of the brains based on one or more input images. For this task, you will implement and use the Dice score to evaluate (aim for a DICE score higher than $0.8$ on the validation set).
# 
# OR
# 
# - Modality transfer: to generate images (selected freely) from another modality (selected freely) based on the input image's modality --- but not between T1w and T1w-CE. For this task, we will use Mean Squared Error (MSE) to evaluate. You should aim for an error below $0.15$ on the validation set.
# 
# Note: If you normalise your data, the scale of the MSE will change. If the scaling factor you used is $s$ (_i.e._, you work with data that is _e.g._ $(x-m)/s$, where $m$ is _e.g._ a mean), then aim for an error below $0.15/s^2$.
# 
# Your task is to look through the highly customizable code below, which contains all the main steps for image segmentation of the data. However you are also encouraged to start out from your notebook for Assignment 2.
# 
# Your tasks, to include in the report, are:
# - Which task you selected and why.
# - How you reached the required performances (for segmentation, a DICE score above 0.8 and for Modality Transfer a MSE error below 0.15.)
# - Plot the training/validating losses and accuracies. Describe when to stop training, and why that is a good choice.
# - Once you have reached the required loss on the validation data, only then evaluate your model on the testing data as well.
# - Describe the thought process behind building your model and choosing the model hyper-parameters.
# - Describe what you think are the biggest issues with the current setup, and how to solve them.
# 
# Upload your notebook to Canvas (with ready-to-run code), together with your report (in PDF format) as a zip-file. The deadline for the assignment is June $4^{th}$, 17:00.
# 
# Good luck and have fun!

# In[15]:


# Parameters
view_data = False #True #
full_data_set = True
batch_size = 32
original_generator = False
original_discriminator = True
# Optimal generator parameters: 


# In[16]:


# Import necessary packages for loading the dataset
from numpy import random
from data_folder_path import get_data_folder_path
import numpy as np  # Package for matrix operations, handling data
np.random.seed(2021)
import os
import matplotlib.pyplot as plt  # Package for plotting
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras
import tensorflow
import copy
import pickle
from hyper_parameters import *

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tensorflow.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU(s) available. Training will be lightning fast!")
else:
    print("No GPU(s) available. Training will be slow!")


# In[17]:


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 data_path,
                 inputs,
                 outputs,
                 batch_size=32
                 ):

        self.data_path = data_path
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size

        if data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(data_path):
            raise ValueError('The data path is incorrectly defined.')

        self.file_idx = 0
        self.file_list = [self.data_path + '/' + s for s in
                          os.listdir(self.data_path)]
        
        self.on_epoch_end()
        with np.load(self.file_list[0]) as npzfile:
            self.out_dims = []
            self.in_dims = []
            self.n_channels = 1
            for i in range(len(self.inputs)):
                im = npzfile[self.inputs[i]]
                self.in_dims.append((self.batch_size,
                                    *np.shape(im),
                                    self.n_channels))
            for i in range(len(self.outputs)):
                im = npzfile[self.outputs[i]]
                self.out_dims.append((self.batch_size,
                                        *np.shape(im),
                                        self.n_channels))
            npzfile.close()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.file_list)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.file_list[k] for k in indexes]

        # Generate data
        i, o = self.__data_generation(list_IDs_temp)
        return i, o

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))
        np.random.shuffle(self.indexes)
    
    #@threadsafe_generator
    def __data_generation(self, temp_list):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        inputs = []
        outputs = []

        for i in range(self.inputs.__len__()):
            inputs.append(np.empty(self.in_dims[i]).astype(np.single))

        for i in range(self.outputs.__len__()):
            outputs.append(np.empty(self.out_dims[i]).astype(np.single))

        for i, ID in enumerate(temp_list):
            with np.load(ID) as npzfile:
                for idx in range(len(self.inputs)):
                    x = npzfile[self.inputs[idx]]                         .astype(np.single)
                    x = np.expand_dims(x, axis=2)
                    inputs[idx][i, ] = x

                for idx in range(len(self.outputs)):
                    x = npzfile[self.outputs[idx]]                         .astype(np.single)
                    x = np.expand_dims(x, axis=2)
                    outputs[idx][i, ] = x
                npzfile.close()

        return inputs, outputs


# #### Again, you can decide between lab computers and your home computer.

# In[18]:


lab_computer = False
if lab_computer:
    gen_dir = "/import/software/3ra023vt21/brats/data/"
else:
    if not os.path.isdir("/CNN2021"):
        print('not in path')
        #! git clone https://github.com/attilasimko/CNN2021.git
    gen_dir = get_data_folder_path()#"/home/fredrik/Documents/Kurser/Teknik/CNNmedical/Lab3/CNN2021/LAB2/" #/CNN2021/LAB2/"
    """
    %cd /CNN2021/LAB2/
    ! git pull
    %cd ../..
    %pwd
    """


# In[19]:


# Available arrays in data: 'flair', 't1', 't2', 't1ce', 'mask'
# See the lab instructions for more info about the arrays
if full_data_set:
    input_arrays = ['flair', 't1', 't2', 't1ce']
else:
    input_arrays = ['t1ce']  # ['flair', 't1', 't1ce']

output_arrays = ['mask']

# Note: This example code is thus setup for the segmentation task. If you want to do image
#       transfer instead, change the output to one of the MRI contrasts (and change the
#       input(s) if you want).

batch_size = batch_size
train_path =  gen_dir + 'training/' 
print(train_path)
gen_train = DataGenerator(data_path = train_path,
                          inputs=input_arrays,
                          outputs=output_arrays,
                          batch_size=batch_size)

gen_val = DataGenerator(data_path = gen_dir + 'validating',
                        inputs=input_arrays,
                        outputs=output_arrays,
                        batch_size=batch_size)

gen_test = DataGenerator(data_path = gen_dir + 'testing',
                         inputs=input_arrays,
                         outputs=output_arrays,
                         batch_size=batch_size)


# In[20]:


# Look at some sample images
if view_data:
    img_in, img_out = gen_train[0]
    for inp in range(np.shape(img_in)[0]):
        plt.figure(figsize=(12, 5))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.imshow(img_in[inp][i, :, :, 0])
            plt.title("Image size: " + str(np.shape(img_in[inp][i, :, :, 0])))
            plt.tight_layout()
        plt.suptitle("Input for array: " + gen_train.inputs[inp], fontsize=20)
        plt.show()

    plt.figure(figsize=(12, 4))
    for outp in range(np.shape(img_out)[0]):
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.imshow(img_out[outp][i, :, :, 0])
            plt.title("Image size: " + str(np.shape(img_out[outp][i, :, :, 0])))
            plt.tight_layout()

        plt.suptitle("Output for array: " + gen_train.outputs[outp], fontsize=20)
        plt.show()

# NOTE: The images are of different size. Also they are RGB images.


# ### The dataset preprocessing so far has been to help you, you should not change anything. However, from now on, take nothing for granted.

# In[21]:


# A quick summary of the data:
if view_data:
    print(f"Training set size   : {len(gen_train.file_list)}")
    print(f"Validation set size : {len(gen_val.file_list)}")
    print(f"Test set size       : {len(gen_test.file_list)}")
    print("")
    print(f"Training size       : {gen_train.in_dims}")
    print(f"Validation size     : {gen_val.in_dims}")
    print(f"Test size           : {gen_test.in_dims}")


# In[22]:


# Import packages important for building and training your model.

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

if not original_generator:
    print('using networks!')
    import networks
    networks.test()


# In[23]:


if original_generator:
    def build_generator(height, width, channels):
        input_1 = Input(shape=(height, width, channels), name='input_1')
        
        x = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_1)
        x = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(1, 1, activation = 'sigmoid')(x)

        return Model(inputs=[input_1], outputs=[x])
else:
    n_data = len(input_arrays)
    def build_generator(height, width, channels):
        model = networks.build_model(height, width, channels, 
                                         n_data=n_data,
                                         filtin = filtin,  
                                         filt = filt,
                                         short_connect=True, 
                                         depth = depth, 
                                         activation=gen_activation,
                                         kernel_initializer=gen_kernel_init,
                                         dropout=gen_dropout)
        
        return model


# In[24]:


if original_discriminator:
    def build_discriminator(height, width, channels, n_images):
        inp =[]
        for i_inp in range(n_images):
            inp.append(Input(shape=(height, width, channels), name='input_image' + str(i_inp)))
        
        inp_mask = Input(shape=(height, width, channels), name='input_mask')
        inp.append(inp_mask)
        inputs = Concatenate(axis=-1)(inp)
        
        x = Conv2D(4, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
        x = Conv2D(4, 5, strides=(5, 5))(x) #MaxPooling2D(pool_size=(5, 5))(x)
        x = Conv2D(8, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = Conv2D(8, 5, strides=(5, 5))(x) #x = MaxPooling2D(pool_size=(5, 5))(x)
        x = Flatten()(x)
        x = Dense(32)(x)
        x = Dense(1)(x)
        x = Activation('tanh')(x)
        
        
        return Model(inputs= inp, outputs=[x])


# In[25]:


# Build your generator.

height, width, channels = gen_train.in_dims[0][1:]
print(height, width, channels)
generator = build_generator(height=height, width=width, channels=channels)
generator.summary()

# NOTE: Are the input sizes correct?
# NOTE: Are the output sizes correct?
# NOTE: Try to imagine the model layer-by-layer and think it through. Is it doing something reasonable?
# NOTE: Are the model parameters split "evenly" between the layers? Or is there one huge layer?
# NOTE: Will the model fit into memory? Is the model too small? Is the model too large?


# In[26]:


# Build your discriminator.

discriminator = build_discriminator(height=height, width=width,
                                    channels=channels, n_images=n_data)
discriminator.summary()

# NOTE: Does the generator have similar number of parameters as the discriminator?

# NOTE: Are the input sizes correct?
# NOTE: Are the output sizes correct?
# NOTE: Try to imagine the model layer-by-layer and think it through. Is it doing something reasonable?
# NOTE: Are the model parameters split "evenly" between the layers? Or is there one huge layer?
# NOTE: Will the model fit into memory? Is the model too small? Is the model too large?


# In[27]:


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return -K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

mse = tensorflow.keras.losses.MeanSquaredError()

# Compile the discriminator

optim_D = optimizers.Adam(lr=learning_rate_D)
discriminator.compile(loss=["binary_crossentropy"], optimizer=optim_D, metrics=["accuracy"])
#binary_crossentropy
# NOTE: Are you satisfied with the loss function?
# NOTE: Are you satisfied with the metric?
# NOTE: Are you satisfied with the optimizer and its parameters?


# In[28]:


# Create GAN model
if original_generator and n_data==1:
    input_1 = Input(shape=(height, width, channels), name='input_1')
elif not original_generator:
    input_1 = []
    for i_inp in range(n_data):
        input_1.append(Input(shape=(height, width, channels), name='input_1' + str(i_inp)))
else: print('not implemented')
# Fix the layers in the discriminator
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False
print(channels)
pred = generator(input_1)
pred_d = discriminator([input_1, pred])
GAN = Model(inputs=input_1, outputs=[pred_d])


optim_GAN = optimizers.Adam(lr=learning_rate_GAN)
# The layers of the discriminator inside the GAN will be non-trainable
# The layers of the discriminator in the discriminator are still trainable.
GAN.compile(loss=["binary_crossentropy"], optimizer=optim_GAN, metrics=["accuracy"])
#binary_crossentropy mean_squared_error

# In[35]:


from IPython.display import clear_output

# Fit the model to the dataset
n_epochs = 10

fake_labels = 0.2 * np.random.random_sample((batch_size, 1))   #np.zeros((batch_size, 1))
real_labels = 0.2 * np.random.random_sample((batch_size, 1)) + 0.8

loss_D = []
acc_D = []
loss_GAN = []
acc_GAN = []
DICE = []
print(n_epochs, ' epochs')
for epoch in range(n_epochs):
    print('Started epoch ', epoch)
    loss_D_epoch = []
    acc_D_epoch = []
    loss_GAN_epoch = []
    acc_GAN_epoch = []
    DICE_epoch = []
    print('Number of batches:', len(gen_train))
    for idx in range(len(gen_train)):
        x, y = gen_train[idx]
        #print(x[0].shape)
        #print(len(x))
        y_pred = generator.predict_on_batch([x]) #x[0]
        z = []
        for i_data in range(len(x)):
            z.append(np.concatenate((x[i_data], x[i_data]), axis=0))
            #z[i_data].append(x[i_data])
        """
        print(len(z))
        print(len(y[0]))
        print(len(y_pred))
        print(len(real_labels))
        print(len(fake_labels))
        """
        z.append(np.concatenate((y[0], y_pred), axis=0))
        out_data =  np.concatenate((real_labels, fake_labels), axis=0)
        #print(out_data.shape)
        #print(len(z))
        #for i in range(len(z)):
        #    print(z[i].shape)
        loss_d = discriminator.train_on_batch(z, out_data)
        loss_D_epoch.append(loss_d[0])
        
        loss_gan = GAN.train_on_batch(x, real_labels) #real_labels
        loss_GAN_epoch.append(loss_gan[0])
        #print(idx, ' batches done')

    print('computing losses')
    for idx in range(len(gen_val)):
        
        x, y = gen_val[idx]
        
        y_pred = generator.predict_on_batch(x)
        z = copy.copy(x)
        z.append(y_pred)
        loss_d = discriminator.test_on_batch(z, fake_labels)
        acc_D_epoch.append(loss_d[1])

        loss_gan = GAN.test_on_batch(x, real_labels)
        acc_GAN_epoch.append(loss_gan[1])

        y_pred = generator.predict_on_batch(x)
        DICE_epoch.append(-dice_coef(y[0], y_pred))

    loss_D.append(np.mean(loss_D_epoch))
    acc_D.append(np.mean(acc_D_epoch))
    loss_GAN.append(np.mean(loss_GAN_epoch))
    acc_GAN.append(np.mean(acc_GAN_epoch))
    DICE.append(np.mean(DICE_epoch))
    
    if (epoch > 0):
        x = np.linspace(1, epoch + 1, epoch + 1)
        clear_output(wait=True)
        plt.figure(1)
        plt.subplot(321)
        plt.title("loss_disc")
        plt.plot(x, loss_D)
        plt.subplot(322)
        plt.title("loss_GAN")
        plt.plot(x, loss_GAN)
        plt.subplot(323)
        plt.plot(x, acc_D)
        plt.subplot(324)
        plt.plot(x, acc_GAN)

        plt.subplot(313)
        plt.plot(x, DICE)
        with open('results.pkl', 'wb') as f:
            pickle.dump([loss_D,acc_D, loss_GAN, acc_GAN, DICE], f)
        #plt.show()

#plt.show()
# In[36]:
mses = []
dices = []
for i in range(len(gen_val)):
    x, y = gen_val[i]

    y_pred = generator.predict_on_batch(x)

    mses.append(mse(y[0], y_pred).numpy())
    dices.append(-dice_coef(y[0], y_pred).numpy())

print(f"Validation MSE : {np.mean(mses):.3f}")
print(f"Validation Dice: {np.mean(dices):.3f}")  # Note: Dice is only relevant for the segmentation task!

with open('validation_values.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([mses, dices, np.mean(mses), np.mean(dices)], f)
# What is your evaluation metric? Is it meaningful with regards to the task?
# NOTE: Is this high enough? How about varying model hyper-parameters? Perhaps implement data augmentation?


# Display some true and generated images

img_in, img_out = gen_val[np.random.randint(0, len(gen_val))]
prediction = generator.predict(img_in)

plt.figure(figsize=(9, 128))
for idx in range(16):
    plt.subplot(batch_size, 3, idx * 3 + 1)
    plt.imshow(img_in[0][idx, :, :, 0])
    plt.title("input")
    plt.subplot(batch_size, 5, idx * 5 + 2)
    plt.subplot(batch_size, 3, idx * 3 + 2)
    plt.imshow(img_out[0][idx, :, :, 0])
    plt.title("GT")
    plt.subplot(batch_size, 3, idx * 3 + 3)
    plt.imshow(prediction[idx, :, :, 0])
    # plt.colorbar()
    plt.title("pred")


# In[37]:


# Evaluate the model on the validation data. What values do you expect for the two tasks?
# predictions = generator.evaluate(gen_val, verbose=0)
# print(f"Validation MSE : {predictions[0]:.3f}")
# print(f"Validation Dice: {predictions[1]:.3f}")  # Note: Dice is only relevant for the segmentation task!


# In[38]:


# Final model evaluation on the test data
if False:  # NOTE: Only ever look at the test data AFTER you have chosen your final model!
    # predictions = model.evaluate(gen_test, verbose=0)
    # print(f"Test MSE       : {predictions[0]:.3f}")
    # print(f"Test Dice      : {predictions[1]:.3f}")
    mses = []
    dices = []
    for i in range(len(gen_test)):
        x, y = gen_test[i]

        y_pred = generator.predict_on_batch(x[0])

        mses.append(mse(y[0], y_pred).numpy())
        dices.append(-dice_coef(y[0], y_pred).numpy())

    print(f"Test MSE : {np.mean(mses):.3f}")
    print(f"Test Dice: {np.mean(dices):.3f}")  # Note: Dice is only relevant for the segmentation task!

