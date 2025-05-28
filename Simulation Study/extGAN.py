
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:33:21 2019

Modified on Fri Oct 25 10:58:11 2024

@author: yobo19   modified by lzxvc
"""
# =============================================================================
# CHANGE PRE REG AND STUFF ! ! ! ! ! ! ! ! ! ! ! ! ! ! !  ! ! ! ! ! ! !
# =============================================================================
import os
os.chdir("/Users/lzxvc/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/extGAN_comp/")



import pandas as pd

df = pd.read_csv("flex_data.csv",   )
print(df)
sim_data = df.drop(['Unnamed: 0'],axis=1).to_numpy()

coords = pd.read_csv("coords.csv",   )
print(coords)
coords = coords.drop(['Unnamed: 0'],axis=1).to_numpy()

coords_padding = pd.read_csv("coords_padding.csv",   )
print(coords_padding)
coords_padding = coords_padding.drop(['Unnamed: 0'],axis=1).to_numpy()


n_trai = 500
batch_size = 100
noise_dim = 100
decay_lab = 90
n_sub_ids = 25

LAMBDA = 0
LAMB_epoch = 0

import os, time
import numpy as np
import pandas as pd
from scipy.stats import kstest
from random import sample
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

LAMBDA_float = np.float32(LAMBDA)


def add_slash(string):
    return '/'+string
def add_totheline(string):
    return '\n'+string
def add_underscore(string):
    return '_'+string




MODE = 'dcgan'
# moving_avg = decay!=0
# ma_label = 'moving_avg='+str(moving_avg)
batch_norm = True
bn_label = 'batch_norm='+str(batch_norm)

## ---------------------------------------------------------------------------------------------------------------------------------
## ------------------------------                Gibberish in their original code              -------------------------------------
## ---------------------------------------------------------------------------------------------------------------------------------
# #Load datar
# df_train_test = pd.read_csv(data_dir + add_slash("train_test_y_EU_rain"), sep=',',header=None).iloc[1:]
# df_train_test = df_train_test.values.astype(float).transpose()

# ids_EU = pd.read_csv(data_dir + "/ids_EU", sep=',',header=None).iloc[1:]
# ids_EU = ids_EU.values.astype(int)

# noise_dim = noise_dim
# noise_label = 'noise_dim='+str(noise_dim)

# n_valid = 0
# n_valid_label = 'n_valid='+str(n_valid)

# n_train = n_trai
# n_train_label = 'n_train='+str(n_train)

# n_test = 1999-n_train
# n_test_label = 'n_test='+str(n_test)

# D_train_it = 2
# D_train_it_label = 'D_train_it='+str(D_train_it)

# decay = decay_lab*0.01
# decay_label = 'decay='+str(decay)

# n_EC_gan = n_test
# n_EC_gan_label = 'n_EC_gan='+str(n_EC_gan)
 
# train_epoch = train_epoch
# train_epoch_label = 'n_epoch=' + str(train_epoch)

# arch_label = '_'+arch

# LAMBDA = LAMBDA 
# LAMBDA_label00 = 'LAMBDA='+str(LAMBDA)

# fil = filter(lambda ch: ch not in ".", LAMBDA_label00)
# LAMBDA_label = "".join(list(fil))+"_"+str(LAMB_epoch)

# LAMBDA_label1 = str(LAMBDA)
# fil = filter(lambda ch: ch not in ".", LAMBDA_label1)
# LAMBDA_label1 = "".join(list(fil))+"_"+str(LAMB_epoch)

# LAMB_epoch = LAMB_epoch 
# LAMBDA_epoch_label = 'LAMB_epoch='+str(LAMB_epoch)

# MODE = 'dcgan'
# moving_avg = decay!=0
# ma_label = 'moving_avg='+str(moving_avg)
# batch_norm = True
# bn_label = 'batch_norm='+str(batch_norm)

# # add condition d
# save_data_dir = save_data_dir0 +add_slash(data) + arch_label + add_slash(noise_label) + \
#  add_slash(n_train_label) + add_slash(n_valid_label) + \
#  add_slash(n_test_label) + add_slash(LAMBDA_label) + add_slash(D_train_it_label) + add_slash(ma_label) + \
#  add_slash(decay_label) +  add_slash(bn_label) + add_slash(n_EC_gan_label) + \
#  add_slash(train_epoch_label)
 
#  # add condition
# results_dir = results_dir0 + add_slash(data) + arch_label + add_slash(noise_label) + \
#  add_slash(n_train_label) + add_slash(n_valid_label) + \
#  add_slash(n_test_label) + add_slash(LAMBDA_label) + add_slash(D_train_it_label) + add_slash(ma_label) + \
#  add_slash(decay_label) +  add_slash(bn_label) + add_slash(n_EC_gan_label) + \
#  add_slash(train_epoch_label)
 
# if not os.path.exists(save_data_dir):
#     os.makedirs(save_data_dir)

# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

# save_path_EC_train = os.path.join(save_data_dir, 'EC_train')
# save_path_EC_test = os.path.join(save_data_dir, 'EC_test')
# save_path_1fourth = os.path.join(save_data_dir, 'station_1fourth')
# save_path_midway = os.path.join(save_data_dir, 'station_midway')
# save_path_3fourth = os.path.join(save_data_dir, 'station_3fourth')
# save_path_last = os.path.join(save_data_dir, 'station_last')

# #add condition
# configuration = data_label+add_totheline(bn_label)+\
# add_totheline(noise_label)+add_totheline(ma_label)+\
# add_totheline(decay_label)+add_totheline(D_train_it_label)+\
# add_totheline(n_train_label)+add_totheline(n_valid_label)+\
# add_totheline(n_test_label)+add_totheline(n_EC_gan_label)+\
# add_totheline(train_epoch_label)+add_totheline(LAMBDA_label00)+\
# add_totheline(str(LAMBDA_epoch_label))

# print(configuration)

# # add condition
# config = "/config"+add_underscore(tit)+add_underscore(std_method)+\
# add_underscore(arch)+add_underscore("ntr"+str(n_trai))+\
# add_underscore("dnoi"+str(noise_dim))+add_underscore("epo"+str(train_epoch))+\
# add_underscore("dec"+str(decay_lab))+add_underscore("LAMBDA"+LAMBDA_label1)+".txt"

# text_file = open(save_data_dir+"/Configuration.txt","w")
# text_file.write("Configuration:\n\n%s" % configuration)
# text_file.close()

 

# # G(z)
# def generator(z, isTrain=True):
#     # First fully connected layer, 1x1x25600 -> 5x5x1024
#     fc = tf.keras.layers.Dense(5*5*1024, use_bias=False)(z)
#     fc = tf.reshape(fc, (-1, 5, 5, 1024))
#     if isTrain:  # Assuming batch_norm is true when training
#         fc = tf.keras.layers.BatchNormalization()(fc, training=isTrain)
#     lrelu0 = lrelu(fc, 0.2)
#     drop0 = tf.keras.layers.Dropout(0.5)(lrelu0, training=isTrain)
    
#     # Deconvolution, 7x7x512
#     conv1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=1, padding='valid', use_bias=False)(drop0)
#     if isTrain:
#         conv1 = tf.keras.layers.BatchNormalization()(conv1, training=isTrain)
#     lrelu1 = lrelu(conv1, 0.2)
#     drop1 = tf.keras.layers.Dropout(0.5)(lrelu1, training=isTrain)
    
#     # Deconvolution, 9x10x256
#     conv2 = tf.keras.layers.Conv2DTranspose(256, (3, 4), strides=1, padding='valid', use_bias=False)(drop1)
#     if isTrain:
#         conv2 = tf.keras.layers.BatchNormalization()(conv2, training=isTrain)
#     lrelu2 = lrelu(conv2, 0.2)
#     drop2 = tf.keras.layers.Dropout(0.5)(lrelu2, training=isTrain)
    
#     # Output layer, 20x24x1
#     logits = tf.keras.layers.Conv2DTranspose(1, (4, 6), strides=2, padding='valid')(drop2)
    
#     return logits




# # D(x)
# def discriminator(x, isTrain=True):
#     # 1st hidden layer 9x10x64
#     conv1 = tf.keras.layers.Conv2D(64, (4, 5), strides=2, padding='valid', 
#                                    kernel_initializer=tf.keras.initializers.GlorotNormal())(x)
#     lrelu1 = lrelu(conv1, 0.2)
#     drop1 = tf.keras.layers.Dropout(0.5)(lrelu1, training=isTrain)
    
#     # 2nd hidden layer 7x7x128
#     conv2 = tf.keras.layers.Conv2D(128, (3, 4), strides=1, padding='valid', use_bias=False)(drop1)
#     if isTrain:
#         conv2 = tf.keras.layers.BatchNormalization()(conv2, training=isTrain)
#     lrelu2 = lrelu(conv2, 0.2)
#     drop2 = tf.keras.layers.Dropout(0.5)(lrelu2, training=isTrain)
    
#     # 2nd hidden layer 5x5x256
#     conv3 = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding='valid', use_bias=False)(drop2)
#     if isTrain:
#         conv3 = tf.keras.layers.BatchNormalization()(conv3, training=isTrain)
#     lrelu3 = lrelu(conv3, 0.2)
#     drop3 = tf.keras.layers.Dropout(0.5)(lrelu3, training=isTrain)
    
#     # Fully connected 1x1
#     flat = tf.reshape(drop3, (-1, 5*5*256))
#     logits = tf.keras.layers.Dense(1)(flat)
#     o = tf.sigmoid(logits)
    
#     return o, logits












## ---------------------------------------------------------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------------------------
# training parameters
lr = 0.0002
tunning_period = 0
awhile0 = 0
best_validation_EC_train = 9999
best_validation_EC_test = 9999
keep_proba = 0.5
smooth_factor = 0.1
alpha = 0.2

# To empirically transform to a uniform
def rank(dataset):
    ranked = np.empty(np.shape(dataset))
    for j in range(np.shape(ranked)[1]):
        if np.sum(dataset[:,j])==0:
            ranked[:,j] = np.zeros(np.shape(ranked[:,j])[0])
        else:  
            indices = list(range(len(dataset[:,j])))
            indices.sort(key=lambda x: dataset[:,j][x])
            output = [0] * len(indices)
            for i, x in enumerate(indices):
                output[x] = i
            ranked[:,j] = output+np.ones(np.shape(ranked[:,j])[0])
    return ranked

def rank2(dataset):
    ranked = np.empty(np.shape(dataset))
    for j in range(np.shape(ranked)[1]):
        if all(i == dataset[0,j] for i in dataset[:,j]):
            ranked[:,j] = len(ranked[:,j])/2 + 0.5
        else:
            array = dataset[:,j]
            temp = array.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(array))
            ranked[:,j] = ranks + 1
    return ranked

def ecdf(dataset):
    return rank2(dataset)/(len(dataset)+1)

def margina(dataset):
    a = ecdf(dataset)
    J = np.shape(a)[1]
    n = np.shape(a)[0]
    z = n*(-1)
    for j in range(J):
        if np.sum(a[:,j])==z:
            a[:,j] = np.zeros(np.shape(a[:,j])[0])
    return a

def tf_converter(arg,typ="float"):
    if typ=="float":
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    elif typ=="int":
        arg = tf.convert_to_tensor(arg, dtype=tf.int32)
    return arg

# Define the leaky ReLU function
def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)

# Define the Generator Model
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = tf.keras.layers.Dense(5*5*1024, use_bias=False)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.conv2d_transpose1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=1, padding='valid', use_bias=False)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.conv2d_transpose2 = tf.keras.layers.Conv2DTranspose(256, (3, 4), strides=1, padding='valid', use_bias=False)
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.conv2d_transpose3 = tf.keras.layers.Conv2DTranspose(1, (4, 6), strides=2, padding='valid')
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, z, isTrain=True):
        x = self.dense(z)
        x = tf.reshape(x, (-1, 5, 5, 1024))
        if isTrain:
            x = self.batch_norm1(x, training=isTrain)
        x = lrelu(x, 0.2)
        x = self.dropout(x, training=isTrain)
        
        x = self.conv2d_transpose1(x)
        if isTrain:
            x = self.batch_norm2(x, training=isTrain)
        x = lrelu(x, 0.2)
        x = self.dropout(x, training=isTrain)
        
        x = self.conv2d_transpose2(x)
        if isTrain:
            x = self.batch_norm3(x, training=isTrain)
        x = lrelu(x, 0.2)
        x = self.dropout(x, training=isTrain)
        
        logits = self.conv2d_transpose3(x)
        output = tf.tanh(logits)
        
        return output


# Define the Discriminator Model
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2d1 = tf.keras.layers.Conv2D(64, (4, 5), strides=2, padding='valid', 
                                              kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.conv2d2 = tf.keras.layers.Conv2D(128, (3, 4), strides=1, padding='valid', use_bias=False)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.conv2d3 = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding='valid', use_bias=False)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, x, isTrain=True):
        x = self.conv2d1(x)
        x = lrelu(x, 0.2)
        x = self.dropout1(x, training=isTrain)
        
        x = self.conv2d2(x)
        if isTrain:
            x = self.batch_norm1(x, training=isTrain)
        x = lrelu(x, 0.2)
        
        x = self.conv2d3(x)
        if isTrain:
            x = self.batch_norm2(x, training=isTrain)
        x = lrelu(x, 0.2)
        x = self.dropout2(x, training=isTrain)
        
        x = tf.reshape(x, (-1, 5*5*256))
        logits = self.dense(x)
        # output = tf.sigmoid(logits)
        
        return logits


# Set up training parameters
num_epochs = 10000
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

# Training step function
@tf.function
def train_step(real_images):
    noise = tf.random.normal([batch_size, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        fake_images = generator(noise, isTrain=True)
        
        # Discriminator outputs
        real_output = discriminator(real_images, isTrain=True)
        fake_output = discriminator(fake_images, isTrain=True)
        
        # Calculate losses
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                    cross_entropy(tf.zeros_like(fake_output), fake_output)

    # Calculate gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients to the optimizer
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    
    





# Create instances of the generator and discriminator
generator = Generator()
discriminator = Discriminator()








# load and prep DATA
# lon_range = np.array([np.amin(ids_EU[:,0]),np.amax(ids_EU[:,0])])
# lat_range = np.array([np.amin(ids_EU[:,1]),np.amax(ids_EU[:,1])])

n_lon = 22 #lon_range[1]-lon_range[0]+1
n_lat = 18 #lat_range[1]-lat_range[0]+1

m = int(n_lat); y=int(n_lon); z=int(1)
X_dim = [m,y,z]
dim = X_dim

#train set: need to standardise it separately (as df is ... but then split)
n_train = sim_data.shape[0]
train_set = sim_data[:n_train,:]
train_set = margina(train_set)
np.shape(train_set)



import matplotlib.pyplot as plt
plt.scatter(coords[:,0],coords[:,1],c=train_set[:,0], marker=',',s=500)
plt.colorbar()
plt.show()


train_set = train_set.transpose()
train_set = np.reshape(train_set,[-1,*X_dim])
tmp = np.reshape(train_set[:,:,:,0], newshape = [500,396], order='A')

plt.scatter(coords[:,0],coords[:,1],c=tmp[0,:], marker=',',s=500)
plt.colorbar()
plt.show()




#pad
pad = 1
train_set = np.pad(train_set,((0,0),(1,1),(1,1),(0,0)),'constant',constant_values=0)
tmp1 = np.reshape(train_set[:,:,:,0], newshape = [500,480], order='C')

plt.scatter(coords_padding[:,0],coords_padding[:,1],c=tmp1[0,:], marker=',',s=300)
plt.colorbar()
plt.show()

train_set_tf = tf_converter(train_set)
train_set = train_set*2-1  # Normalize the images to [-1, 1]




# #valid set: need to standardise it separately (as df is ... but then split)
# test_set = df_train_test[n_train:,:]
# test_set = margina(test_set)
# np.shape(test_set)


# test_set = np.reshape(test_set,[-1,*X_dim])
# test_set = np.pad(test_set,((0,0),(1,1),(1,1),(0,0)),'constant',constant_values=0)
# test_set_tf = tf_converter(test_set)





BATCH_SIZE = batch_size

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_set).shuffle(train_set.shape[0]).batch(BATCH_SIZE)



# Training loop
for epoch in range(num_epochs):
    for image_batch in train_dataset:
      train_step(image_batch)
    # train_step(train_set)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Generator and Discriminator trained.')

    




# Generate fake images after training
noise = tf.random.normal([500, noise_dim])  # Generate 5 random noise vectors
generated_images = generator(noise, isTrain=False)  # Generate fake images

# Show the shapes of generated images
print("Generated Images Shape:", generated_images.shape)



import matplotlib.pyplot as plt



fig = plt.figure(figsize=(6, 6))

for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.imshow((generated_images[i, :, :, 0]+1)/2)
    plt.axis('off')
plt.show()


for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.imshow(train_set[i, :, :, 0])
    plt.axis('off')
plt.show()






fig = plt.figure(figsize=(3,3))

for i in range(9):
    plt.subplot(3, 3, i+1)
    if i<3: plt.imshow((generated_images[i, :, :, 0]+1)/2) 
    else: plt.imshow(train_set[i-3, :, :, 0])
    plt.axis('off')
plt.show()





wh = 2
plt.imshow(train_set[wh,:,:,0])
plt.show()
plt.imshow((generated_images[wh,:,:,0]+1)/2)
plt.show()

X_sim = (generated_images[:,:,:,0]+1)/2
X_sim = np.reshape(X_sim.numpy(), [500,480], order='C')
np.savetxt("./X_sim.csv", X_sim, delimiter=",")



plt.scatter(coords_padding[:,0],coords_padding[:,1],c=np.log(X_sim[0,:]), marker=',',s=500)
plt.colorbar()




































# Generate noise to train the generator
def generate_noise(batch_size, noise_dim):
    return np.random.normal(size=(batch_size, noise_dim))

def generate_and_plot_fake_images(generator, noise_dim, num_images=5):
    noise = generate_noise(num_images, noise_dim)
    generated_images = generator(noise, isTrain=False)
    
    # Rescale pixel values (assuming generator produces images in range [-1, 1])
    generated_images = (generated_images + 1) / 2.0

    # Plot the images
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(generated_images[i, :, :, 0])  # Adjust for RGB if needed
        plt.axis('off')
    plt.show()



# Generate fake images after training
generate_and_plot_fake_images(generator, noise_dim=noise_dim, num_images= batch_size)




    
# Get indices of EU/FR from padded matrix
def get_ii_pad(pad,ids_FR_in_FRext):
    return ids_FR_in_FRext+pad
    
# These indices can directly be used on the matrix to return the non-zero values in a vector
def non_zero_indices(x):
    g = np.where(x!=0)
    return g
# Different format than non_zero_indices. Reassembles coordinates of the non-zero points in a matrix
def non_zero_coords(g):
    g = np.array(g)
    h = [g[:,i] for i in range(np.shape(g)[1])]
    return h
# Based on the indices returned by non_zero_indices, returns corresponding values in a vector
def get_relevant_points(non_zero_indices,mytensor):
    g = non_zero_indices
    b = mytensor[g]
    return b

def get_all_relevant_points(non_zero_indices,mytensor):
    n = np.shape(mytensor)[0]
    all_ = []
    all_ = [get_relevant_points(non_zero_indices,mytensor[i,:,:,:]) for i in range(n)]
    all_ = np.array(all_)
    return all_

def get_all_relevant_points2(mytensor,ids_):
    all_ = np.squeeze(mytensor[:,ids_[:,1],ids_[:,0]],axis=2)
    return all_


def inv_unit_frechet(uniform):
    frechet_distributed = -1/np.log(uniform)
    return frechet_distributed

def unit_frechet(x):
    frechet_pdf = (x**(-2))*np.exp(-1/x)
    return frechet_pdf

def pos_coords(frechet):
    J = np.shape(frechet)[1]
    vec=[]
    for i in np.linspace(0,J-1,J):
        i = int(i)
        for j in np.linspace(i+1,J-1,J-1-i):
            vec += [np.array([i,j])]
    return vec

def ind_ind(ind,pos_coordinates):
    vec = []
    ind = np.sort(ind)
    for i in ind: 
        for j in ind:
            if i<j: vec += [np.array([i,j])]
    inds = []
    l = np.array(pos_coordinates).tolist()
    l_vec = np.array(vec).tolist()
    for m in l_vec:
        inds += [l.index(m)]
    return inds
 
def com_pos_samps(pos_coordinates,pos_coordinate):
    here = np.all(pos_coordinates==pos_coordinate,axis=1)
    index = np.where(here)
    return index

def inv_max(frechet,pos_coordinates,pos_coordinate):
    n = np.shape(frechet)[0]
    index = com_pos_samps(pos_coordinates,pos_coordinate)[0][0]
    locs = pos_coordinates[index]
    loc1 = int(locs[0]);loc2=int(locs[1])
    X1 = np.reshape(frechet[:,loc1],(n,1));X2=np.reshape(frechet[:,loc2],(n,1))
    X = np.concatenate((X1,X2),axis=1)
    maxim = np.array([np.max(X[k,:]) for k in range(n)])
    inv_maxim = 1/maxim
    return inv_maxim

def get_EC(pos_coordinates,pos_coordinate,ECs):
    index = com_pos_samps(pos_coordinates,pos_coordinate)[0][0]
    this_one = ECs[index]
    return this_one

def exp_pdf(x,theta):
    return theta*np.exp(-theta*x)

def min_max(array):
    r = np.array([np.min(array),np.max(array)])
    return r
    
def EC_est(vec,frechet,n,i):
    pos = vec[i]
    loc1 = int(pos[0])
    loc2 = int(pos[1])
    samp1 = np.reshape(1/frechet[:,loc1],(n,1))
    samp2 = np.reshape(1/frechet[:,loc2],(n,1))
    samps = np.concatenate((samp1,samp2),axis=1)
    minim = [np.min(samps[k,:]) for k in range(n)]
    EC = n/np.sum(minim)
    return EC

def ECs_estim(frechet,vec):
    n = np.shape(frechet)[0]
    shape = np.shape(vec)[0]
    ECs = np.empty(shape)
    for i in range(shape):
        ECs[i] = EC_est(vec,frechet,n,i)
    return ECs

def ECs_estimates2(data,ids_,ind="not",inspect=False):
    all_ = get_all_relevant_points2(data,ids_)
    if ind!="not":
        ind = np.array(ind).tolist()
        all_ = all_[:,ind]
    frechet = inv_unit_frechet(all_)
    pos_coordinates_here = pos_coords(frechet)
    ECs = ECs_estim(frechet,pos_coordinates_here)

    return ECs

def av_kstest(data,n,df):
    # ! to unif(-1,1)
    df_non_zero_indices = non_zero_indices(df[1,:,:,:])
    all_data = get_all_relevant_points(df_non_zero_indices,data)[:(n),:]
    J = np.shape(all_data)[1]
    ks_data_ = [kstest(all_data[:,j],'uniform',(-1,2))[0] for j in range(J)]
    ks_data = np.mean(ks_data_)
    return ks_data

def av_kstest2(data,ids_,n,samp=False):
    # ! to unif(-1,1)
    all_data = get_all_relevant_points2(data,ids_)[:(n),:]
    if samp: 
        sha = np.shape(all_data)[1]
        ind = sample(range(sha),51)
        all_data = all_data[:,ind]
    J = np.shape(all_data)[1]
    ks_data_ = [kstest(all_data[:,j],'uniform',(-1,2))[0] for j in range(J)]
    ks_data = np.mean(ks_data_)
    return ks_data   

#####EC estimation in tf


def pos_coords4tf(n_sub_ids):
    J = n_sub_ids
    vec=[]
    for i in np.linspace(0,J-1,J):
        i = int(i)
        for j in np.linspace(i+1,J-1,J-1-i):
            vec += [np.array([i,j])]
    return vec
pos_coordinates = np.array(pos_coords4tf(n_sub_ids))
pos_coordinates_tf = tf_converter(pos_coordinates,typ="int")

def tf_shape(obj):
    return obj.get_shape().as_list()

def tf_repeat_each(vec,times=396):
    vec = tf.reshape(vec,[-1,1])
    dd = tf.tile(vec, [1, times])
    dd = tf.reshape(dd, [-1,1]) 
    return dd

def tf_cbind(t1,t2):
    return tf.concat([t1, t2],1)

def tf_get_ind(ids_tf,n,times=396):
    #times to repeat each element of seq(1,rang)
    #c is the upper limit of rang, batch_size for G, n_train for train_set
    c = tf.constant([n,1])
    ids_rep = tf.tile(ids_tf,c)
    seq = tf.range(n)
    seq_rep = tf_repeat_each(seq,times)
    return tf_cbind(seq_rep,ids_rep)

def tf_get_all_relevant_points2(mytensor,ids_tf,n):
    #squeezes 300,20,24,1 to 300,20,24
    res = tf.squeeze(mytensor, axis=3)
    #gets indices of 396 location in tf
    ind_tf = tf_get_ind(ids_tf,n)
    shape = tf.shape(ids_tf)[0]
    #gathers the 396 locations for res -> dim: 300*18*22
    fin = tf.gather_nd(res,ind_tf)
    #dim: 300*18*22,1 -> dim:300,396
    all_ = tf.reshape(fin,shape=[-1,shape])
    return all_

def tf_get_sub_relevant_points2(mytensor,ind_loss,n,times=n_sub_ids):
    #gets indices of 25 random location in tf of tensor dim:300x396
    ind_loss_tf = tf_get_ind(ind_loss,n,times=times)
    shape = tf.shape(ind_loss)[0]
    #dim:300x396 -> 300x25
    all_new = tf.gather_nd(mytensor,ind_loss_tf)
    all_new = tf.reshape(all_new,shape=[-1,shape])
    return all_new

#automatically sets n=batch_size
def tf_get_all_relevant_points2_G(G_z,ids_tf,n=batch_size):
    return tf_get_all_relevant_points2(G_z,ids_tf,n)

#automatically sets n=n_train
def tf_get_all_relevant_points2_train(train_set_tf,ids_tf,n=n_train):
    return tf_get_all_relevant_points2(train_set_tf,ids_tf,n)

def tf_inv_unit_frechet(uniform):
    frechet_distributed = -1/tf.log(uniform)
    return frechet_distributed

def tf_inv_exp(uniform):
    exp_distributed = -tf.log(1-uniform)
    return exp_distributed

#Estimates the EC for a pair of locations
def tf_EC_est(vec_tf,frechet_tf,n,i):
    pos1 = vec_tf[i,:][0]
    pos2 = vec_tf[i,:][1]
#    frechet_train_pos1 = 1/frechet_tf[:,pos1]
#    frechet_train_pos2 = 1/frechet_tf[:,pos2]
    frechet_train_pos1 = frechet_tf[:,pos1]
    frechet_train_pos2 = frechet_tf[:,pos2]
    frechet_train_pos1 = tf.reshape(frechet_train_pos1,shape=[-1,1])
    frechet_train_pos2 = tf.reshape(frechet_train_pos2,shape=[-1,1])
    minima_tf = tf.where(tf.less(frechet_train_pos1,frechet_train_pos2),frechet_train_pos1,frechet_train_pos2)
    ECs_tf = n/tf.reduce_sum(minima_tf)
    return ECs_tf

def tf_ECs_estimates2(data,ids_tf,is_Gen,ind_loss,is_valid=False):
    if is_Gen:
        if is_valid:
            all_ = tf_get_all_relevant_points2_G(data,ids_tf,n=n_EC_gan)
            n = n_EC_gan
        else:
            all_ = tf_get_all_relevant_points2_G(data,ids_tf,n=batch_size)
            n = batch_size
    else:
        if is_valid:
            all_ = tf_get_all_relevant_points2_train(data,ids_tf,n=n_valid)
            n = n_valid
        else:
            all_ = tf_get_all_relevant_points2_train(data,ids_tf,n=n_train)
            n = n_train
    all_ = tf_get_sub_relevant_points2(all_,ind_loss,n)
    n_ = tf.shape(all_)[0]
    n_ = tf.cast(n_, dtype=tf.float32)
#    uniform = tf.map_fn(lambda x: tfd.Empirical(all_[:,x]).cdf(all_[:,x])*(n_/(n_+1)), tf.range(tf.shape(all_)[1]),dtype=tf.float32)
#    uniform = tf.map_fn(lambda x: dist.cdf(all_[:,x]), tf.range(tf.shape(all_)[1]),dtype=tf.float32)
#    uniform = tf.transpose(uniform)
#    frechet = tf_inv_unit_frechet(all_)
    frechet = tf_inv_exp(all_)
    ECs = tf.map_fn(lambda x: tf_EC_est(pos_coordinates_tf,frechet,n,x), tf.range(tf.shape(pos_coordinates_tf)[0]),dtype=tf.float32)
    return ECs, tf.math.reduce_max(all_), tf.math.reduce_min(all_), tf_inv_unit_frechet(tf.math.reduce_max(all_)), tf_inv_unit_frechet(tf.math.reduce_min(all_))      

def tf_ECs_estimates3(data,ids_tf,is_Gen,ind_loss,is_valid=True):
    if is_Gen:
        if is_valid:
            all_ = tf_get_all_relevant_points2_G(data,ids_tf,n=n_train)
            n = n_train
        else:
            all_ = tf_get_all_relevant_points2_G(data,ids_tf,n=batch_size)
            n = batch_size
    else:
        if is_valid:
            all_ = tf_get_all_relevant_points2_train(data,ids_tf,n=n_test)
            n = n_test
        else:
            all_ = tf_get_all_relevant_points2_train(data,ids_tf,n=n_train)
            n = n_train
    all_ = tf_get_sub_relevant_points2(all_,ind_loss,n)
    n_ = tf.shape(all_)[0]
    n_ = tf.cast(n_, dtype=tf.float32)
    uniform = tf.map_fn(lambda x: tfd.Empirical(all_[:,x]).cdf(all_[:,x])*(n_/(n_+1)), tf.range(tf.shape(all_)[1]),dtype=tf.float32)
    uniform = tf.transpose(uniform)
    frechet = tf_inv_exp(uniform)
    ECs = tf.map_fn(lambda x: tf_EC_est(pos_coordinates_tf,frechet,n,x), tf.range(tf.shape(pos_coordinates_tf)[0]),dtype=tf.float32)
    return ECs, tf.math.reduce_max(all_), tf.math.reduce_min(all_), tf_inv_unit_frechet(tf.math.reduce_max(all_)), tf_inv_unit_frechet(tf.math.reduce_min(all_))      

def get_all_vars(var):
    k = np.shape(var)[0]
    all_vars = []
    for i in range(k):
        all_vars += [np.reshape(var[i],[-1,])]
    return np.concatenate(all_vars)




ids_EU_pad = get_ii_pad(pad,ids_EU)
ids_ = ids_EU_pad-1
ids_ = np.int32(ids_)
ids_rev = np.column_stack((ids_[:,1],ids_[:,0]))
ids_tf = tf.convert_to_tensor(ids_rev)

new_m = np.shape(train_set)[1]; new_y = np.shape(train_set)[2]
dim = np.array([new_m,new_y])
df_non_zero_indices = non_zero_indices(train_set[1,:,:,:])

# variables : input
x = tf.keras.Input(shape=(None, new_m, new_y, z), dtype=tf.float32, name='x')
z = tf.keras.Input(shape=(None, 1, 1, noise_dim), dtype=tf.float32, name='z')
LAMB = tf.keras.Input(shape=(), dtype=tf.float32, name="LAMB")
ind_loss = tf.keras.Input(shape=(n_sub_ids, 1), dtype=tf.int32, name="ind_loss")
isTrain = tf.keras.Input(shape=(), dtype=tf.bool, name="isTrain")

# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

res_g = tf_ECs_estimates2(G_z,ids_tf,True,ind_loss)

gEC = res_g[0]
addinfo = res_g[1:]
trEC = tf_ECs_estimates2(train_set_tf,ids_tf,False,ind_loss)[0]
coef = tf.convert_to_tensor(len(pos_coordinates),tf.float32)
pre_reg_loss = (1/tf.sqrt(coef))*tf.norm(tf.subtract(gEC,trEC))
reg_loss = LAMB*pre_reg_loss




# loss for each network for vanilla GANs
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real) * (1 - 0.1)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss_no_reg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))
G_loss = G_loss_no_reg + reg_loss

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    Optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)
    
    D_grads = Optimizer.compute_gradients(D_loss, var_list=D_vars)
    D_optim = Optimizer.apply_gradients(D_grads)
    
    G_grads_noreg = Optimizer.compute_gradients(G_loss_no_reg,var_list=G_vars)
    G_optim_noreg = Optimizer.apply_gradients(G_grads_noreg)
    
    G_grads_pre_regonly = Optimizer.compute_gradients(pre_reg_loss,var_list=G_vars)
    
    G_grads_regonly = Optimizer.compute_gradients(reg_loss,var_list=G_vars)
    
    G_grads = Optimizer.compute_gradients(G_loss,var_list=G_vars)
    G_optim_reg = Optimizer.apply_gradients(G_grads)
    
# Adding the moving average adjustment to optimization
ema = tf.train.ExponentialMovingAverage(decay=decay)
        
with tf.control_dependencies([D_optim]):
    m_D = ema.apply(D_vars)

with tf.control_dependencies([G_optim_noreg]):
    m_G_noreg = ema.apply(G_vars)
    
with tf.control_dependencies([G_optim_reg]):
    m_G_reg = ema.apply(G_vars)
    
av_D = [ema.average(D_vars[i]) for i in range(len(D_vars))]
av_G = [ema.average(G_vars[i]) for i in range(len(G_vars))]

# Validation score
res_g_val = tf_ECs_estimates3(G_z,ids_tf,True,ind_loss,is_valid=True)
gEC_val = res_g_val[0]
add_info_val = res_g_val[1:]
trEC_test = tf_ECs_estimates3(test_set_tf,ids_tf,False,ind_loss,is_valid=True)[0]
trEC_train = tf_ECs_estimates3(train_set_tf,ids_tf,False,ind_loss,is_valid=False)[0]
valid_score_test = (1/tf.sqrt(coef))*tf.norm(tf.subtract(gEC_val,trEC_test))
valid_score_train = (1/tf.sqrt(coef))*tf.norm(tf.subtract(gEC_val,trEC_train))
# Treshhold
threshold_score = (1/tf.sqrt(coef))*tf.norm(tf.subtract(trEC_train,trEC_test))

print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_hist = {}

#train_hist['accuracy_valid_list_EC'] = []
train_hist['last_improv_EC_train'] = []
train_hist['last_improv_EC_test'] = []

#train_hist['per_iterat_ptimes'] = []
train_hist['per_score_calc_ptimes'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

train_hist['accuracy_G_train_EC'] = []
train_hist['accuracy_G_test_EC'] = []
train_hist['accuracy_train_test_EC'] = []

train_hist['best_validation_EC_train'] = []
train_hist['best_validation_EC_test'] = []

# training-loop
saver = tf.train.Saver()
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    print('###################### EPOCH START, LAMB='+str(LAMBDA_float)+' LAMB_epoch='+str(LAMB_epoch))
    LAMB_=LAMBDA_float
#    accuracy_valid_list_EC = []
    epoch_start_time = time.time()
    for iterat in range(3):
        iterat_start_time = time.time()
        # update discriminator
        x_ = train_set
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, noise_dim))
        
        _, av_D_ = sess.run([m_D, av_D], {x: x_, z: z_, isTrain: True})
        [D_vars[k].load(av_D_[k],sess) for k in range(len(D_vars))] 

        if iterat%D_train_it==0 and iterat!=0:
            #noise
            z_ = np.random.normal(0, 1, (batch_size, 1, 1, noise_dim))
            # selection of location to compute ECs in loss
            ind_loss_ = sample(range(396),n_sub_ids)
            ind_loss_ = np.reshape(ind_loss_,(n_sub_ids,1))
            # update generator            
            _, av_G_ = sess.run([m_G_noreg, av_G], {z: z_, x:x_, isTrain: True})
            [G_vars[k].load(av_G_[k],sess) for k in range(len(G_vars))] 

            
            print('###################### GRADIENT UPDATE HAS OCCURED')

        # calculate iteration time
        per_iterat_time = time.time() - iterat_start_time
#        
#        train_hist['per_iterat_ptimes'].append(per_iterat_time)
   
    #calculate EC scores between train test and GANs
    if(epoch%100==0):
        score_calc_start_time = time.time()
        z_ = np.random.normal(0, 1, (n_train, 1, 1, noise_dim))
        accuracy_G_train_EC = sess.run(valid_score_train,{z: z_, ind_loss:ind_loss_, isTrain: False})
        accuracy_G_test_EC = sess.run(valid_score_test,{z: z_, ind_loss:ind_loss_, isTrain: False})
        accuracy_train_test_EC = sess.run(threshold_score,{z: z_, ind_loss:ind_loss_, isTrain: False})
        train_hist['accuracy_G_train_EC'].append(accuracy_G_train_EC)
        train_hist['accuracy_G_test_EC'].append(accuracy_G_test_EC)
        train_hist['accuracy_train_test_EC'].append(accuracy_train_test_EC)
        print("Accuracy_G_test_EC: ", accuracy_G_test_EC)
        print("Accuracy_train_test_EC: ",accuracy_train_test_EC)
        score_calc_time = time.time()-score_calc_start_time
    
        train_hist['per_score_calc_ptimes'].append(score_calc_time)
        
        accuracy_valid_EC_train = accuracy_G_train_EC
        accuracy_valid_EC_test = accuracy_G_test_EC
    
    # Every epoch after awhile0 + tunning period epochs, or at last epoch calculate accuracy for train and valid
    if epoch>tunning_period+awhile0:
        this_accuracy_EC_train = accuracy_valid_EC_train
        if this_accuracy_EC_train < best_validation_EC_train:
            # update best accuracy
            best_validation_EC_train = this_accuracy_EC_train
            # update at what epoch it was found
            last_improv_EC_train = epoch
            # save the model since it has best accuracy
            saver.save(sess, save_path_EC_train)
            # add a little star to the printed message to show us your love
            improved_str_EC_train = '***'
        else:
            # ...
            improved_str_EC_train = ''            

        # print message with accuracy values and indication if best
        msg = "Accuracy EC train: {}{}"
        print(msg.format(round(accuracy_valid_EC_train,3), improved_str_EC_train))

    if epoch>tunning_period+awhile0:
        this_accuracy_EC_test = accuracy_valid_EC_test
        if accuracy_valid_EC_test < best_validation_EC_test:
            # update best accuracy
            best_validation_EC_test = this_accuracy_EC_test
            # update at what epoch it was found
            last_improv_EC_test = epoch
            # save the model since it has best accuracy
            saver.save(sess, save_path_EC_test)
            # add a little star to the printed message
            improved_str_EC_test = '***'
        else:
            # ...
            improved_str_EC_test = ''            

        # print message with accuracy values and indication if best
        msg = "Accuracy EC test: {}{}"
        print(msg.format(round(accuracy_valid_EC_test,3), improved_str_EC_test))
            
    # Stations to retrieve generated data at different times in training, namely at 2, it//3 and it*2//3...
    if epoch == train_epoch//4 - 1:
        saver.save(sess, save_path_1fourth)
    if epoch == train_epoch*2//4 - 1:
        saver.save(sess, save_path_midway)
    if epoch == train_epoch*3//4 - 1:
        saver.save(sess, save_path_3fourth)
    if epoch == train_epoch - 1:
        saver.save(sess,save_path_last)
    
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    print('[%d/%d] - epoch_time: %.2f, mean_iterat_time: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(per_iterat_time))) 
    print('###################### EPOCH END')
    print("\n")
    print("\n")
    print("\n")

end_time = time.time()
total_ptime = end_time - start_time

train_hist['total_ptime'].append(total_ptime/60)

train_hist['last_improv_EC_train'].append(last_improv_EC_train)
train_hist['last_improv_EC_test'].append(last_improv_EC_test)

df = pd.DataFrame(data=np.array(train_hist['accuracy_G_train_EC']))
df.to_csv(results_dir+"/accuracy_G_train_EC.csv", header=None)

df = pd.DataFrame(data=np.array(train_hist['accuracy_G_test_EC']))
df.to_csv(results_dir+"/accuracy_G_test_EC.csv", header=None)

df = pd.DataFrame(data=np.array(train_hist['accuracy_train_test_EC']))
df.to_csv(results_dir+"/accuracy_train_test_EC.csv", header=None)

train_hist['best_validation_EC_train'].append(best_validation_EC_train)
train_hist['best_validation_EC_test'].append(best_validation_EC_test)


summary = 'per epoch time='+str(round(np.mean(train_hist['per_epoch_ptimes']),ndigits=3))+\
add_totheline('EC scores calc time='+str(round(np.mean(train_hist['per_score_calc_ptimes']),ndigits=3)))+\
add_totheline('Total time='+str(round(train_hist['total_ptime'][0],ndigits=3)))+\
add_totheline('best validation EC train='+str(round(train_hist['best_validation_EC_train'][0],ndigits=3)))+\
add_totheline('best validation EC test='+str(round(train_hist['best_validation_EC_test'][0],ndigits=3)))+\
add_totheline('last improv EC train='+str(train_hist['last_improv_EC_train'][0]))+\
add_totheline('last improv EC test='+str(train_hist['last_improv_EC_test'][0]))+\
add_totheline('total epochs='+str(train_epoch))


print(summary)

text_file = open(results_dir+"/summary.txt", "w")
text_file.write("summary: %s" % summary)
text_file.close()

sess.close()






















