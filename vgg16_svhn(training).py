import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import os
import pickle
import tensorflow as tf
from PIL import Image

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)


from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout,Conv3D,MaxPooling3D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
#from vis.utils import utils
from tensorflow.keras import activations
#from vis.visualization import visualize_activation, get_num_filters
#from vis.input_modifiers import Jitter



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

tf.config.experimental_run_functions_eagerly(True)
print("test")

# def fix_gpu():
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)


# fix_gpu()


def VGG16_cifar10():
    model = VGG16(include_top=False,input_shape=(224,224,3))
    # for layer in model.layers:
        # layer.trainable = False
    x = tf.keras.models.Sequential()
    x.add(model)
    x.add(Flatten())
    x.add(Dense(4096, activation='relu'))
    x.add(Dense(4096, activation='relu'))
    x.add(Dense(10, activation='softmax'))
    x.summary()
    return x
def VGG16_cifar10_pretrain():
    model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
    for layer in model.layers:
        layer.trainable = False
    x = tf.keras.models.Sequential()
    x.add(model)
    x.add(Flatten())
    x.add(Dense(4096, activation='relu'))
    x.add(Dense(4096, activation='relu'))
    x.add(Dense(10, activation='softmax'))
    x.summary()
    return x
def train_datagen(train_dir, val_dir, batch_size, traindf, val_df):
    
    datagen = ImageDataGenerator(
        rescale=1./255.
        )

 
    train_generator = datagen.flow_from_dataframe(
        train_df,
        directory=train_dir,
        x_col='image',
        y_col='class',
        weight_col='sample_weight',
        batch_size= batch_size,
        class_mode='categorical',
        target_size=(224, 224)
        

        )
    val_datagen = ImageDataGenerator(
        rescale=1./255.)

    valid_generator = val_datagen.flow_from_dataframe(
            val_df,
            directory = val_dir,
            x_col='image',
            y_col='class',
            batch_size= batch_size,
            class_mode='categorical',
            target_size=(224, 224)
            )
    return train_generator,valid_generator

def data_csv_fn(data_csv_dir,data_csv ):
    data_csv_dir = data_csv_dir
    data_csv = data_csv
    df=pd.read_csv(data_csv_dir+"\\"+data_csv)
    df['class'] = df['class'].astype(str)
    return df

def training(train_generator,valid_generator,model_avg, model_name):
    if os.path.exists(model_avg) == False:
        os.mkdir(model_avg)
    for i in range(10):
        model = tf.keras.models.Sequential()
        model = VGG16_cifar10()
        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor = 'val_loss'),
        ]
        opt = tf.keras.optimizers.Adam(learning_rate=0.000005)
        
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        model_avg = model_avg
        history = model.fit(
            train_generator,                    
            validation_data = valid_generator,
            epochs=20,
            callbacks=my_callbacks
            )
        model.save(model_avg+"\\"+model_name+'_'+str(i))

data_csv_dir = 
train_csv ="SVHN_balance_train.csv"
train_df = data_csv_fn(data_csv_dir,train_csv)

val_csv ="SVHN_balance_val.csv"
val_df = data_csv_fn(data_csv_dir,val_csv)

train_dir = 
val_dir = 
batch_size = 64
train_generator, valid_generator = train_datagen(train_dir, val_dir, batch_size, train_df, val_df)

model_avg = 
model_name = '0505_svhnbalance_stop2'
print("=======ok=========")
training(train_generator,valid_generator, model_avg, model_name)





    