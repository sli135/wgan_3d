#!/usr/bin/python
import numpy as np
import csv,os.path,os
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import batchgenerator as bg

tf.compat.v1.experimental.output_all_intermediates(True)
tf.compat.v1.disable_eager_execution()
tf.__version__

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Embedding, SeparableConv1D,MaxPooling2D, Dropout
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Add, Convolution2D, Conv2D, Conv2DTranspose, multiply, LeakyReLU, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from functools import partial
from tensorflow.keras.layers import Lambda

BATCH_SIZE = 32
DECAY_RATE = 1e-5
# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
time = 350
KE,KP = 0.01,0.01 # highKE 0.1 scintE 0.01 lowKE 0.005 phase2 0.01
train_file,train_label = 'train_files/phase2/train_small_set_file_se_2.npy','train_files/phase2/train_small_set_label_se_2.npy'
#train_file,train_label = 'train_files/phase2/train_small_set_file_se.npy','train_files/phase2/train_small_set_label_se.npy'
val_file,val_label = 'train_files/phase1/val_file_se.npy','train_files/phase1/val_label_se.npy'

################################# Loss and Save Functions ###################################################
class RandomWeightedAverage(Add):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def save_loss(csv_path,critic,dis,epoch):
    #csv_path = '/gpfs/slac/staas/fs1/g/g.exo/shaolei/GAN_loss/'
    mode = 'w'
    write_header = True
    if os.path.isfile(csv_path + 'critic_loss_constrain_'+str(epoch)+'.csv'):
        mode = 'a'
        write_header = False
    with open(csv_path + 'critic_loss_constrain_'+str(epoch)+'.csv', mode=mode) as csv_file:
        fieldnames = ['loss', 'wasserstein1', 'wasserstein2','gradient_penalty']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in critic:
            writer.writerow({'loss': row[0], 'wasserstein1': row[1], 'wasserstein2': row[2], 'gradient_penalty':row[3]})
    with open(csv_path + 'generator_loss_constrain_'+str(epoch)+'.csv', mode=mode) as csv_file:
        fieldnames = ['loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in dis:
            writer.writerow({'loss': row})
def save_weights(save_model_path,name,generator_model,discriminator_model):
    symbolic_weights = getattr(generator_model.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open(save_model_path + 'model_generator_weights_'+name+'.pkl', 'wb') as f:
        pickle.dump(weight_values, f,pickle.HIGHEST_PROTOCOL)
    symbolic_weights = getattr(discriminator_model.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open(save_model_path + 'model_discriminator_weights_'+name+'.pkl', 'wb') as f:
        pickle.dump(weight_values, f,pickle.HIGHEST_PROTOCOL)

def save_models(save_model_path,name,generator,discriminator,generator_model,discriminator_model):
    generator.save(save_model_path + 'model_generator_'+name+'.h5')
    discriminator.save(save_model_path + 'model_discriminator_'+name+'.h5')
    generator_model.save(save_model_path + 'model_train_generator_'+name+'.h5')
    discriminator_model.save(save_model_path + 'model_train_discriminator_'+name+'.h5')
    generator_model.save_weights(save_model_path + 'weights_train_generator_'+name+'.h5')
    discriminator_model.save_weights(save_model_path + 'weights_train_discriminator_'+name+'.h5')

def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight,sample_weight=None):
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors 
    ## tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead.
    print('averaged_samples',averaged_samples)
    gradients = K.gradients(y_pred, averaged_samples)[0]
    print('gradients',gradients)
    #with tf.GradientTape() as g:
    #    g.watch(averaged_samples)
    #    gradients = g.gradient(y_pred, averaged_samples)
    print('g.gradient',gradients)
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)
def constrainer_loss(y_true, y_pred):
    k = 1
    return tf.reduce_mean((y_true - y_pred) ** 2) 

def DenselyConnectedSepConv(z, nfilter, **kwargs):
    ''' Densely Connected SeparableConvolution2D Layer'''
    c = SeparableConv1D(nfilter, 3, padding = 'same', depth_multiplier=1, **kwargs)(z)
    return concatenate([z, c], axis=-1)
########################################   ReLoad functions ##############################
def load_models(save_model_path,name):
    print("************************** Load Models. *******************************")
    discriminator = load_model(save_model_path + 'model_discriminator_'+name+'.h5')
    generator = load_model(save_model_path + 'model_generator_'+name+'.h5')

    generator_input_label = Input(shape=(4,))

    real_samples = Input(shape=(74,350,1))
    critic_input_label = Input(shape=(4,))
    print('real_samples: ',real_samples)
    generator_input_noise_for_discriminator = Input(shape=(100,))
    generator_input_for_discriminator = [critic_input_label,generator_input_noise_for_discriminator]
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    averaged_samples = RandomWeightedAverage()([real_samples,
                                                generated_samples_for_discriminator])
    print(averaged_samples)
    generator_model = load_model(save_model_path + 'model_train_generator_'+name+'.h5',
                        compile=False,
                        custom_objects={'wasserstein_loss':wasserstein_loss,})
    #generator_model.summary()
    discriminator_model = load_model(save_model_path + 'model_train_discriminator_'+name+'.h5',
                        compile=False,
                        custom_objects={'wasserstein_loss':wasserstein_loss,
                                        'RandomWeightedAverage':RandomWeightedAverage,
                                        'gradient_penalty':partial(gradient_penalty_loss,
                                                            gradient_penalty_weight = GRADIENT_PENALTY_WEIGHT,
                                                            averaged_samples = averaged_samples)})
    #discriminator_model.summary()
    return generator_model,discriminator_model

def load_opt_weights(save_model_path,name):
    with open(save_model_path + 'model_generator_weights_'+name+'.pkl', 'rb') as f:
        opt_w_g = pickle.load(f)
    with open(save_model_path + 'model_discriminator_weights_'+name+'.pkl', 'rb') as f:
        opt_w_d = pickle.load(f)
    return opt_w_g,opt_w_d

###################################### Make Neural Networks Functions ################################################


def make_discriminator():
    name = 'discriminator'
    input_features = Input(shape=(4,),name = 'features')
    
    feature_reshape = Reshape((1,1,4))(input_features)
    feature_tile = K.tile(feature_reshape,[1,74,350,1])

    images = Input(shape=(74,350,1),name = 'image')

    sequential_input = concatenate([feature_tile,images],axis = 3)
    print(sequential_input.shape)
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding = 'same', input_shape = (74,350,5))) # , input_shape = (74,100,1)
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    #####################################################
    model.add(Reshape((64,)))
    #model.add(Dense(1,kernel_initializer='he_normal',name = 'discriminator_output'))
    #model.summary()
    model_output_layer = model(sequential_input)
    output_concatenate = concatenate([input_features,model_output_layer],axis = 1)
    output_dense = Dense(128)(output_concatenate)
    output_relu = LeakyReLU()(output_dense)
    output_layer = Dense(1,name = 'discriminator_output')(output_relu)
    output_model = Model(inputs=[input_features,images],
                            outputs=[output_layer],name = name)
    return output_model

def make_NoPooling_discriminator():
    name = 'discriminator'
    input_features = Input(shape=(4,),name = 'features')
    
    feature_reshape = Reshape((1,1,4))(input_features)
    feature_tile = K.tile(feature_reshape,[1,74,350,1])

    images = Input(shape=(74,350,1),name = 'image')

    sequential_input = concatenate([feature_tile,images],axis = 3)
    print(sequential_input.shape)
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding = 'same', input_shape = (74,350,5))) # , input_shape = (74,100,1)
    model.add(LeakyReLU())
    #model.add(AveragePooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    #model.add(AveragePooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    #model.add(AveragePooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    #model.add(AveragePooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(1, (1,1), padding='same'))
    model.add(LeakyReLU())
    #####################################################
    model.add(Flatten())

    #model.add(Dense(1,kernel_initializer='he_normal',name = 'discriminator_output'))
    #model.summary()
    model_output_layer = model(sequential_input)
    output_concatenate = concatenate([input_features,model_output_layer],axis = 1)
    output_dense = Dense(128)(output_concatenate)
    output_relu = LeakyReLU()(output_dense)
    output_layer = Dense(1,name = 'discriminator_output')(output_relu)
    output_model = Model(inputs=[input_features,images],
                            outputs=[output_layer],name = name)
    return output_model
def regression_loss(y_true,y_pred,
                          label,k,sample_weight=None):
    #k = 0.01
    print('y_pred',y_pred.shape)
    print('label',label.shape)
    print(y_pred[:BATCH_SIZE][1].shape)
    print(y_pred[BATCH_SIZE:][1].shape)
    e_real = tf.reduce_mean((label - y_pred[:BATCH_SIZE]) ** 2)
    e_gen = tf.reduce_mean((label - y_pred[BATCH_SIZE:]) ** 2)
    return k * K.abs(e_real - e_gen)
def make_generator():
    name = 'generator'
    model = Sequential()
    model.add(Dense(5*11*16, input_dim=104))
    model.add(LeakyReLU())
    model.add(Reshape((5,11,16), input_shape=(5*11*16,)))
    model.add(Conv2DTranspose(16,(3,3), strides = (2,4) ,padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(32,(3,3), strides = 2 ,padding='same'))
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64,(3,3), strides = 2 ,padding='same'))
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(128,(3,3), strides = 2 ,padding='same'))
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128,(7,3), padding = 'valid'))
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (3,3), padding='same')) #change in Conv2D and no strides needed
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (3,3), padding='same'))
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (3,3), padding='same')) #change in Conv2D and no strides needed
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(1, (1,1), padding='same'))
    #model.summary()
    noise = Input(shape=(100,),name = 'noise')
    label = Input(shape=(4,), name = 'label')
    model_input = concatenate([label,noise],axis = 1)
    
    image = model(model_input)
    return Model(inputs = [label,noise], outputs = [image],name = name)
def train(Epochs,save_name,continue_flag = False):
    print('Start training. Epochs:',Epochs)
    data_path = "/gpfs/slac/staas/fs1/g/g.exo-userdata/users/shaolei/"
    save_model_path = '/gpfs/slac/staas/fs1/g/g.exo/shaolei/GAN_MPD_models/'
    save_loss_path = '/gpfs/slac/staas/fs1/g/g.exo/shaolei/GAN_MPD_loss/'
    try:
        os.mkdir(save_model_path)
    except:
        print(save_model_path,'exists')
        #os.system('rm '+save_loss_path+'/*')
    try:
        os.mkdir(save_loss_path)
    except:
        print(save_loss_path,'exists')
        #os.system('rm '+save_loss_path+'/*')
    ################ batch generator ##############################
    X_train,Y_train = bg.load_data(train_file,train_label) #_500-3000
    train_batch_generator = bg.Batch_Generator(X_train,Y_train, BATCH_SIZE)

    X_test,Y_test = bg.load_data(val_file,val_label)
    val_batch_generator = bg.Batch_Generator(X_test,Y_test, BATCH_SIZE)

    ############################################################
    if continue_flag == False:
        discriminator = make_NoPooling_discriminator()
        #discriminator.summary()
        generator = make_generator()
    else:
        generator_model,discriminator_model = load_models(save_model_path,save_name)
        opt_w_g,opt_w_d = load_opt_weights(save_model_path,save_name)
        print('D() weights',len(opt_w_d))
        print('G() weights',len(opt_w_g))
        discriminator = discriminator_model.get_layer('discriminator')
        generator = generator_model.get_layer('generator')
    opt_d = Adam(1e-8,beta_1 = 0.5,beta_2 = 0.9, decay= 0) # MPD-NoPooling-phase2-flatten 1e-8
    opt_g = Adam(5e-8, beta_1 = 0.5,beta_2 = 0.9, decay= 0) # old 5e-8
    
    ############ Make D() and G() for training ###################

    generator_input_noise = Input(shape=(100,))
    generator_input_label = Input(shape=(4,))
    generator_input = [generator_input_label,generator_input_noise]
    generator_layers = generator(generator_input)
    discriminator_input = generator_layers
    name_layer = tf.keras.layers.Lambda(lambda x: x, name='discriminator_output')
    discriminator_layers_for_generator = name_layer(discriminator([generator_input_label,discriminator_input]))

    if continue_flag == False:
        discriminator.trainable = False 
    else:
        generator.trainable = True
        discriminator.trainable = False
    generator_model = Model(inputs=[generator_input_label,
                                    generator_input_noise],
                            outputs=[discriminator_layers_for_generator])
    generator_model.compile(optimizer = opt_g, loss = [wasserstein_loss])
    if continue_flag == False:
        #generator_model.summary()
        generator_model.metrics_names
        generator_model._make_train_function()
        print("G() weights before reload",len(generator_model.optimizer.get_weights()))
    else:
        #generator_model.summary()
        generator_model.load_weights(save_model_path + 'weights_train_generator_'+save_name+'.h5')
        generator_model._make_train_function()
        print("G() weights",len(generator_model.optimizer.get_weights()))
        generator_model.optimizer.set_weights(opt_w_g)

    ################################ Discriminator model ############################
    
    discriminator.trainable = True
    generator.trainable = False
    real_samples = Input(shape=(74, 350, 1))
    critic_input_label = Input(shape=(4,))

    print('real_samples: ',real_samples)
    generator_input_noise_for_discriminator = Input(shape=(100,))
    generator_input_for_discriminator = [critic_input_label,generator_input_noise_for_discriminator]
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    print('generated_samples_for_discriminator: ',generated_samples_for_discriminator)
    discriminator_output_from_generator = tf.keras.layers.Lambda(lambda x: x, name='gen_pred_output')(discriminator([critic_input_label,generated_samples_for_discriminator]))
    print('discriminator_output_from_generator: ',generated_samples_for_discriminator.shape)
    discriminator_output_from_real_samples = tf.keras.layers.Lambda(lambda x: x, name='real_samples_output')(discriminator([critic_input_label,real_samples]))

    averaged_samples = tf.keras.layers.Lambda(lambda x: x, name='averaged_samples')(RandomWeightedAverage()([real_samples,
                                                generated_samples_for_discriminator]))
    averaged_samples_out = tf.keras.layers.Lambda(lambda x: x, name='averaged_output')(discriminator([critic_input_label,averaged_samples]))
    print('averaged_samples_out',averaged_samples_out)
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)

    #constrainer_output_from_generator = tf.keras.layers.Lambda(lambda x: x, name='constrainer_output')(constrainer_e(generated_samples_for_discriminator))
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = 'gradient_penalty'

    discriminator_model = Model(inputs=[real_samples,
                                        critic_input_label,
                                        generator_input_noise_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    discriminator_model.compile(optimizer=opt_d, loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
     
    if continue_flag == True:
        discriminator_model.load_weights(save_model_path + 'weights_train_discriminator_'+save_name+'.h5')
        discriminator_model._make_train_function()
        print('D() weights',len(discriminator_model.optimizer.get_weights()))
        discriminator_model.optimizer.set_weights(opt_w_d)
    discriminator_model.metrics_names
    ################################### Training #####################################
    start = 0
    iterations = train_batch_generator.__len__()
    indices = np.array([i for i in range(iterations - 1)])
    
    val_ids = val_batch_generator.__len__() - 1
    for epoch in range(start,start+Epochs):
        #iterations = 20
        print("Epoch: ", epoch)
        print("Number of batches: ", iterations)
        minibatches_size = BATCH_SIZE * TRAINING_RATIO
        np.random.shuffle(indices)
        #print(indices)
        discriminator_loss = []
        generator_loss = []
        for i in range(iterations // TRAINING_RATIO):
            if i % 100 == 0: print('Minibatch %i processed.' %i,)
            minibatches = indices[i:i+TRAINING_RATIO]
            for j in range(TRAINING_RATIO):
                image_batch,label_batch = train_batch_generator.__getitem__(minibatches[j])
                batch_size = image_batch.shape[0]
                noise = np.random.rand(batch_size, 100).astype(np.float32)
                positive_y = np.ones((batch_size, 1), dtype=np.float32)
                negative_y = -positive_y
                dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
                discriminator_loss.append(discriminator_model.train_on_batch(
                    [image_batch,label_batch, noise],
                    [negative_y, positive_y, dummy_y]))
            generator_loss.append(generator_model.train_on_batch([label_batch,
                                                                  np.random.rand(batch_size,100),
                                                                  image_batch],
                                                                [negative_y]))
        save_loss(save_loss_path,discriminator_loss,generator_loss,save_name)
        save_models(save_model_path,save_name,generator,discriminator,generator_model,discriminator_model)
        save_weights(save_model_path,save_name,generator_model,discriminator_model)
    print("G() opt",len(generator_model.optimizer.get_weights()))
    print("D() opt",len(discriminator_model.optimizer.get_weights()))
    print('Done.')
if __name__ == "__main__":
    print("tensorflow version is", tf.__version__)
    print("keras version is", tf.keras.__version__)
    save_name = 'critic-test'#'test-1'#'MPD-NoPooling-phase2-flatten' # 'MPD-NoPooling-phase2'# 
    print("#---------------------Training: ",save_name,"-----------------------#")
    print("-------- KE = %f, KP = %f -----------" % (KE,KP))
    #train(10,save_name,False)
    train(10,save_name,True)