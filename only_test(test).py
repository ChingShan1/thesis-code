
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
# from vis.utils import utils
from tensorflow.keras import activations
#from vis.visualization import visualize_activation, get_num_filters
# from vis.input_modifiers import Jitter



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

tf.config.experimental_run_functions_eagerly(True)


def data_csv_fn(data_csv_dir,data_csv ):
    data_csv_dir = data_csv_dir
    data_csv = data_csv
    df=pd.read_csv(data_csv_dir+"\\"+data_csv)
    df['class'] = df['class'].astype(str)
    return df
    
test_datagen = ImageDataGenerator(
        rescale=1./255)


data_dir = 
test_generator = test_datagen.flow_from_directory(
        directory = data_dir,
        batch_size= 64,
        class_mode='categorical',
        target_size=(224, 224),
        shuffle=False,
        )

model_dir = 
accuracy_all=[]
macro_f1score_all =[]
print(model_dir)
for i in os.listdir(model_dir):
    test_generator.reset()
    model = tf.keras.models.load_model(model_dir+"/"+i)
    # loss, accuracy = model.evaluate(test_generator, verbose=1)
    predictions = model.predict(test_generator, steps=10000/64)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.array(test_generator.classes)
    accuracy = accuracy_score(true_classes, predicted_classes)
    print("Step "+str(i))
    print("accuracy: "+ str(accuracy))
    macro_f1score = f1_score(true_classes, predicted_classes, average='macro')
    print("macro_f1score: "+ str(macro_f1score))
    accuracy_all.append(accuracy)
    macro_f1score_all.append(macro_f1score)





 