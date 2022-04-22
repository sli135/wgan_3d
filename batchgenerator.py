import numpy as np 
import pandas as pd 
import tensorflow as tf
import os
import shutil
'''
from skimage.transform import resize
from PIL import Image as im 
tf.compat.v1.experimental.output_all_intermediates(True)
tf.compat.v1.disable_eager_execution()
tf.__version__

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, SeparableConv1D,MaxPooling2D, Dropout
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Add, Convolution2D, Conv2D, Conv2DTranspose, multiply, LeakyReLU, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from functools import partial
from tensorflow.keras.layers import Lambda
'''
class Batch_Generator(tf.keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            np.load(file_name)['arr_0'].reshape(1, 14, 7, 350)
               for file_name in batch_x]), np.array(batch_y)
class Batch_weight_Generator(tf.keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, weights_e,weights_p, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.weights_e = weights_e
    self.weights_p = weights_p
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_ew = self.weights_e[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_pw = self.weights_p[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            np.load(file_name)['arr_0'].reshape(74, 350, 1)
               for file_name in batch_x]), np.array(batch_y,),np.array(batch_ew,),np.array(batch_pw,)
def read_data(filename,images_dir,name):
    nLabels = 4
    nFeatures = 74*350
    count = 0
    labels = []
    filenames = []
    #images_dir = '/gpfs/slac/staas/fs1/g/g.exo-userdata/users/shaolei/data_reweight/all_images'
    with open(filename) as infile:
        for line in infile:
            if count % 1000 == 0:
                print("%i processed."%count)
            file = 'image_%i.npz'%count
            full_path = os.path.join(images_dir, file)
            x = np.fromstring(line, dtype=float, sep=',')
            arr = np.array(x[0:nFeatures])
            arr = arr.reshape(74,350,1)
            np.savez_compressed(full_path, arr)

            filenames.append(full_path )
            labels.append(x[nFeatures:nFeatures+4])
            count += 1
    labels = np.array(labels)
    print(len(filenames))
    print(labels.shape)
    np.save('train_files/filenames_%s.npy'%name, filenames)
    np.save('train_files/labels_%s.npy'%name,labels)
    np.save('train_files/labels_p_%s.npy'%name,labels[:,:3])
    np.save('train_files/labels_e_%s.npy'%name,labels[:,3])
    print(filename + " has %i events" % count)
def make_constrainer(train_energy):
    print(train_energy)
    input_shape = (74, 350, 1)
    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding = 'same',input_shape = input_shape,
        kernel_regularizer=l2(5e-3)))
    model.add(MaxPooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(32, (5, 5), padding = 'same',
        kernel_regularizer=l2(5e-3)))
    model.add(MaxPooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(64, (5, 5), padding = 'same',
        kernel_regularizer=l2(5e-3)))
    model.add(MaxPooling2D(pool_size=(2, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(128, (5, 5), padding = 'same',
        kernel_regularizer=l2(5e-3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None))
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Dense(1024))
    model.add(Dense(256))
    if train_energy == True:
        model.add(Dense(1))
    else:
        model.add(Dense(3))
    opt_c = Adam(2e-6,beta_1 = 0.5,beta_2 = 0.9, decay= 0.0)
    model.compile(optimizer = opt_c,loss = 'mean_squared_error')
    model.summary()
    return model
def save_data(train_files,test_files):
    data_path = "/gpfs/slac/staas/fs1/g/g.exo-userdata/users/shaolei/data_reweight/"
    train_dir = '/gpfs/slac/staas/fs1/g/g.exo-userdata/users/shaolei/data_reweight/all_images'
    test_dir = '/gpfs/slac/staas/fs1/g/g.exo-userdata/users/shaolei/data_reweight/test_images'
    for file in train_files:
        read_data(data_path + file,train_dir,"train")
    for file in test_files:
        read_data(data_path + file,test_dir,"test")
def load_data(files,label_files):
    filenames = np.load(files)
    labels = np.load(label_files)
    return filenames, labels
    #shutil.make_archive("all_images", "zip", "all_images")
def load_weight_data(files,label_files,weight_e_files,weight_p_files):
    filenames = np.load(files)
    labels = np.load(label_files)
    weights_e = np.load(weight_e_files)
    weights_p = np.load(weight_p_files)
    return filenames, labels, weights_e,weights_p
def train(X_train_filenames, y_train,X_val_filenames, y_val):
    batch_size = 20
    my_training_batch_generator = Batch_Generator(X_train_filenames, y_train, batch_size)
    my_validation_batch_generator = Batch_Generator(X_val_filenames, y_val, batch_size)
    model = make_constrainer(False)
    
    model.fit_generator(generator=my_training_batch_generator,
                    epochs = 3,
                   verbose = 1,
                   validation_data = my_validation_batch_generator)
    '''
    batch = my_training_batch_generator.__getitem__(0)
    model.train_on_batch(batch)
    '''
if __name__ == "__main__":
    train_set = ['wTh_WF_Train.csv']
    test_files = ['wTh_WF_Test.csv']
    #save_data(train_set,test_files)
    X_train,Y_train = load_data("train_files/filenames_train.npy","train_files/labels_p_train.npy")
    X_test,Y_test = load_data("train_files/filenames_test.npy","train_files/labels_p_test.npy")
    train(X_train,Y_train,X_test,Y_test)




