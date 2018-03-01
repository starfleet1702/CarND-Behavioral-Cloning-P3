# Importing dependencies
import numpy as np; # for math functions
import csv; # for reading data from csv file
import os; # for using os specific fucntion for reading files in folder etc.
from sklearn.model_selection import train_test_split; # for splitting training and validation data
from sklearn.utils import shuffle;
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D,MaxPooling2D,Dropout,Cropping2D,Lambda,ELU; # for defining DNN model using keras and tensorflow as backend
from keras.callbacks import ModelCheckpoint;
import matplotlib.pyplot as plt;
import cv2;
from random import getrandbits;
import random;
import math;

# Defining Constants
MODEL_FILE_NAME='model.h5'
DATA_FOLDER_PATH = './data/';
#DATA_FOLDER_PATH = 'E:/Nk_Carnd/Term1/Project_3/SharpTurn1/';
#DATA_FOLDER_PATH = 'E:/Nk_Carnd/Term1/Project_3/data/';
CSV_FILE_NAME = 'driving_log.csv';
Y_CROP_START=65;
Y_CROP_END=135;
COLUMN_INDEX_CENTER_IMG=0;
COLUMN_INDEX_LEFT_IMG=1
COLUMN_INDEX_RIGHT_IMG=2;
COLUMN_INDEX_STEERING_ANGLE=3;
TARGET_SIZE = (64,64);
PRINT_MODEL_ARCH=1;
STEERING_ANGLE_OFFSET=0.20;
IMAGE_COUNT_PER_BIN = 80

#-----Hyper Parameters-----
BATCH_SIZE = 64;
EPOCHS = 15;

def normalize_dataset(dataset):
    num_bins = 1000;
    step=1.0/num_bins;
    freq=[0]*(num_bins+1);
    output_dataset=[];
    for data in dataset:
        yi=float(data[COLUMN_INDEX_STEERING_ANGLE]);
        index=math.floor(abs(yi/step));
        if(freq[index]<IMAGE_COUNT_PER_BIN):
            freq[index]+=1;
            output_dataset.append(data);
    steer_angles = [];
    for i in range(0,len(output_dataset)):
        steer_angles.append(float(output_dataset[i][COLUMN_INDEX_STEERING_ANGLE]));

    # For Visulizing Histogram
    # print("length of normalize datset : ", len(output_dataset));
    # plt.figure('normalizede dataset histogram');
    # plt.hist(steer_angles,bins=100);
    # plt.show();

    return output_dataset;

def load_csv(csv_file_path=os.path.join(DATA_FOLDER_PATH,CSV_FILE_NAME)):
    entries=[];
    with open(csv_file_path) as csv_file:
        reader = csv.reader(csv_file);
        # Check for headers in first row
        try :
            entry = next(reader);
            temp_angle = float(entry[COLUMN_INDEX_STEERING_ANGLE]);
            # if header not found seek to 0
            csv_file.seek(0);
        except:
            pass

        for entry in reader:
            #print('entry : ',entry)
            entries.append(entry);
    return entries;

def read_image_by_path(img_path):
    '''
     param: img_path - path of image
     read image as RGB image from the given path
     In Drive.py Images are read as RGB Image , but opencv uses BGR to load an Image , 
     so converting it to BGR2RGB after reading
    '''
    # CV2 Implementation
    # img=cv2.imread(img_path);
    # return cv2.cvtColor(img,cv2.COLOR_BGR2RGB);
    return plt.imread(img_path);

def get_preprocessed_image(img):
    # Cropping image and resizing
    return cv2.resize(img[Y_CROP_START:Y_CROP_END,:,:],TARGET_SIZE);

def augment_row(row):
    steering_angle=float(row[COLUMN_INDEX_STEERING_ANGLE]);
    # randomly choosing the image from center , left and right camera
    img_index_to_choose = np.random.choice([COLUMN_INDEX_CENTER_IMG,COLUMN_INDEX_LEFT_IMG,COLUMN_INDEX_RIGHT_IMG]);
    img=read_image_by_path(DATA_FOLDER_PATH+str.strip(row[img_index_to_choose]));
    # For Visualizing Image
    # plt.figure('Original Image')
    # plt.imshow(img);
    # plt.show();
    # img=read_image_by_path(str.strip(row[img_index_to_choose]));
    if(img_index_to_choose==COLUMN_INDEX_LEFT_IMG):
        steering_angle+=STEERING_ANGLE_OFFSET;
    elif (img_index_to_choose==COLUMN_INDEX_RIGHT_IMG):
        steering_angle-=STEERING_ANGLE_OFFSET;

    img = get_preprocessed_image(img);

    #randomly flip the image vertically
    if(random.getrandbits(1)): # Faster than choice
        # Flipping the image vertically to generate more data on fly, it also remove the left bias from the data
        #if(random()>=0.5): # Faster than getrandbits
        img=cv2.flip(img,flipCode=1); # Flippingthe image vertically, flagCode=0 Horizatal flip , -1 for both
        steering_angle=-steering_angle;

    return img,steering_angle;

def get_generator(samples,batch_size=BATCH_SIZE):
    num_samples=len(samples)
    while 1:
        for i in range(0,num_samples,batch_size):
            batch_samples=samples[i:i+batch_size];  # batch_samples=samples[i:min(i+batch_size,num_samples)];, min is not requirred python handles it internally,array will be cliped till len(array)
            steering_angles=np.zeros(shape=(batch_size,),dtype=np.float32);  # these should be nparrays
            imgs=np.zeros(shape=(batch_size,64,64,3),dtype=np.float32);  # these should be nparrays
            for j in range(0,len(batch_samples)):
                batch_sample = batch_samples[j];
                imgs[j],steering_angles[j] = augment_row(batch_sample);
            imgs,steering_angles=shuffle(imgs,steering_angles);
            yield (imgs,steering_angles); # these should be nparrays

def resize_img(img):
    img=np.asarray(img,dtype=np.float32)
    return cv2.resize(img,TARGET_SIZE)

def get_model(Verbose=PRINT_MODEL_ARCH):
    '''
    function defines the Deep Neural Netwrok Model built using karas
    '''
    model=Sequential();
    # Cropping layer : added in model to do cropping on GPU which will be faster
    # model.add(Cropping2D(cropping=((Y_CROP_START,Y_CROP_END),(0,0)),input_shape=(160,320,3)));
    # Resizing the image to (64,64) as requirement of model
    model.add(Lambda(lambda x : (x/255.0)-0.5,input_shape=(64,64,3)));
    # 1st Convolution layer output_shape = 64x64x32 
    model.add(Convolution2D(32,5,5,subsample=(1,1),border_mode='same'));#,activation='ELU'));
    model.add(ELU());
    # 1st MaxPooling layer output_shape = 32x32x32
    model.add(MaxPooling2D(pool_size=(2,2),strides=None));
    # 2nd Convolution layer output_shape = 32x32x16
    model.add(Convolution2D(16,3,3,subsample=(1,1),border_mode='same'));#,activation='ELU'));
    model.add(ELU());
    #model.add(Dropout(0.4));
    # 2nd MaxPooling layer output_shape = 16x16x16
    model.add(MaxPooling2D(pool_size=(2,2),strides=None));
    # 3rd Convolution layer output_shape = 14x14x16
    model.add(Convolution2D(16,3,3,subsample=(1,1),border_mode='valid'));#,activation='ELU'));
    model.add(ELU());
    model.add(Dropout(0.3));

    # Flatten the output
    model.add(Flatten())
    # 4th layer : Dense output_shape = 1024
    #model.add(Dense(1024))
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dropout(0.3))
    # 5th layer: Dense output_shape = 512
    #model.add(Dense(512))
    model.add(Dense(256))
    model.add(ELU())
    # Finally a single output, since this is a regression problem
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    if(Verbose==1):
        model.summary();
    return model;

def train_model(samples):
    # Splitting data
    train_data,valid_data = train_test_split(samples,test_size=0.2);
    # generators
    train_generator = get_generator(train_data,BATCH_SIZE);
    valid_generator = get_generator(valid_data,BATCH_SIZE);

    model=get_model();

    # loading weights - from previous best weights
    if(os.path.exists(MODEL_FILE_NAME)):
        print("Loading Model....")
        model.load_weights(MODEL_FILE_NAME);
        print("Model Loaded....")
    
    # checkpoint to save best weights after each epoch based on the improvement in val_loss
    checkpoint = ModelCheckpoint(MODEL_FILE_NAME, monitor='val_loss', verbose=1,\
    save_best_only=True, mode='min',save_weights_only=False);
    callbacks_list = [checkpoint];#,callback_each_epoch];

    print('training started....')
    history = model.fit_generator(train_generator,samples_per_epoch=len(train_data),\
    nb_epoch=EPOCHS,validation_data=valid_generator,nb_val_samples=len(valid_data),callbacks=callbacks_list)

    # Plotting Losses
    plt.figure();
    plt.plot(history.history['loss']);
    plt.plot(history.history['val_loss']);
    plt.legend(['training loss','val_loss']);
    plt.show();

    with open('model.json','w') as model_json:
        model_json.write(model.to_json());

if __name__=="__main__":
    print("loading CSV file.....")
    samples = load_csv();
    print("CSV file loaded successfully..");
    print("Number of samples in CSV : ",len(samples));
    print(samples[0][0]);

    norm_samples=normalize_dataset(samples);
    train_model(norm_samples);

    #normalize_dataset(samples);#[COLUMN_INDEX_CENTER_IMG],samples[COLUMN_INDEX_STEERING_ANGLE])

    # print(samples[0][0])
    # img=np.array(plt.imread((samples[0][0])));
    # gen = get_generator(norm_samples);
    # x,y=next(gen)
    # rand_index=np.random.randint(0,len(x))
    # print("shape of input image : " , (x[rand_index].shape));
    # plt.figure("Random preprocessed Image");
    # plt.imshow((x[rand_index]*255/np.amax(x[rand_index]).astype('uint8')));
    # plt.show();

    # For Visualizing Original and preprocessed Image
    # org_img=read_image_by_path('.//data//IMG//center_2016_12_01_13_30_48_287.jpg')
    # plt.figure("Original Image");
    # plt.imshow(org_img);
    # pre_img = get_preprocessed_image(org_img);
    # plt.figure("Preprocessed Image");
    # plt.imshow(pre_img);
    # plt.show();

