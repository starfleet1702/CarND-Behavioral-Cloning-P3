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
DATA_FOLDER_PATH = 'D:/Term1/Project 3 Behaviour cloning/CarND-Behavioral-Cloning-P3-master/data/';
CSV_FILE_NAME = 'driving_log.csv';
Y_CROP_START=65;
Y_CROP_END=135;
COLUMN_INDEX_CENTER_IMG=0;
COLUMN_INDEX_LEFT_IMG=1
COLUMN_INDEX_RIGHT_IMG=2;
COLUMN_INDEX_STEERING_ANGLE=3;
PRINT_MODEL_ARCH=1;
STEERING_ANGLE_OFFSET=0.20;
INPUT_IMAGE_WIDTH = 320
INPUT_IMAGE_HEIGHT = 160

#-----Hyper Parameters-----
BATCH_SIZE = 64;
EPOCHS = 15;

def balance_dataset(dataset,IMAGE_COUNT_PER_BIN = 80):
    """
    Returns Balanced Dataset
    Divides the dataset into num_bins, append only IMAGE_COUNT_PER_BIN images into output_dataset
    resulting in balanced dataset
    It is requirred here as the provided sample data is having most of the steering angle samples around 0

    # Arguments
    dataset : list - unbalanced dataset
    IMAGE_COUNT_PER_BIN : int - number of Images to keep per bean
    
    # Returns
    output_dataset : list - balanced dataset

    """

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
    """
    Returns the list of entries in the csv file expect headers

    # Arguments
    csv_file_path : String - path of csv file to load

    # Returns
    entries : list - list of all entries in csv file provided except headers
    """

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
    """
    read image as RGB image from the path provided
    
    In Drive.py Images are read as RGB Image , but opencv uses BGR to load an Image
    so Conversion to RGB after reading is requirred for CV2 Implementation
    
    Matplotlib reads an image as RGB Image only

    # Arguments 
    img_path : String - path of the image to read
    
    # Returns
    output_img : numpy.array - RGB Image

    """

    # CV2 Implementation
    # img=cv2.imread(img_path);
    # return cv2.cvtColor(img,cv2.COLOR_BGR2RGB);
    return plt.imread(img_path);

def get_preprocessed_image(img):
    """
    Returns Preprocessed Image
    Preprocessing includes cropping the image vertically

    # Argument
        img : numpy.array - input image

    # Returns:
        output_img : numpy.array - preprocessed output image

    """

    # Cropping image and resizing
    # output_img = cv2.resize(img[Y_CROP_START:Y_CROP_END,:,:],TARGET_SIZE);
    output_img = img[Y_CROP_START:Y_CROP_END,:,:];
    # plt.figure('Original Image')
    # plt.imshow(output_img);
    # plt.show();
    return output_img;

def augment_row(row):
    """
    Returns an augmented row 
    augmantation includes randomly choosing the camera image from center left and right camera
    adding the offset to steering angle based on the image choosen

    preprocessing the image which includes cropping the image to just select the drivable portion from the frame

    randomly flipping the image along vertical axis and changing the sign of steering angle

    # Argument 
        row : array - consisting of 3 camera images , steering angle and throttle and break values
    # Return 
        img : augmented image
        steering_angle : augmented steering angle value
    """

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
    """
    loads the batch as per the batch_size and returns imgages and steering angle
    generator is used to save memory and loading data for single batch only
    # Argument
        samples : list - containing imgages and steering angle
        batch_size : int - size of a batch
    
    # Return 
        Yields imgages and steering angle for a batch
    """
    num_samples=len(samples)
    while 1:
        for i in range(0,num_samples,batch_size):
            batch_samples=samples[i:i+batch_size];  # batch_samples=samples[i:min(i+batch_size,num_samples)];, min is not requirred python handles it internally,array will be cliped till len(array)
            steering_angles=np.zeros(shape=(batch_size,),dtype=np.float32);  # these should be nparrays
            imgs=np.zeros(shape=(batch_size,Y_CROP_END-Y_CROP_START,INPUT_IMAGE_WIDTH,3),dtype=np.float32);  # these should be nparrays
            for j in range(0,len(batch_samples)):
                batch_sample = batch_samples[j];
                imgs[j],steering_angles[j] = augment_row(batch_sample);
            imgs,steering_angles=shuffle(imgs,steering_angles);
            yield (imgs,steering_angles); # these should be nparrays

def get_model(Verbose=PRINT_MODEL_ARCH):
    """
    Returns the Deep Neural Sequential Netwrok Model built using karas

    # Arguments
    Verbose : int if 1 then prints the summary of model on console else not 
    
    # Returns
    model : Sequential Model

    """
    model=Sequential();
    # Cropping layer : added in model to do cropping on GPU which will be faster
    # model.add(Cropping2D(cropping=((Y_CROP_START,Y_CROP_END),(0,0)),input_shape=(160,320,3)));
    model.add(Lambda(lambda x : (x/255.0)-0.5,input_shape=((Y_CROP_END-Y_CROP_START),INPUT_IMAGE_WIDTH,3)));
    # 1st Convolution layer output_shape = 35x160x32 
    model.add(Convolution2D(32,5,5,subsample=(2,2),border_mode='same'));#,activation='ELU'));
    model.add(ELU());
    # 1st MaxPooling layer output_shape = 17x80x32
    model.add(MaxPooling2D(pool_size=(2,2),strides=None));
    # 2nd Convolution layer output_shape = 17x80x16
    model.add(Convolution2D(16,3,3,subsample=(1,1),border_mode='same'));#,activation='ELU'));
    model.add(ELU());
    # model.add(Dropout(0.4));
    # 2nd MaxPooling layer output_shape = 8x40x16
    model.add(MaxPooling2D(pool_size=(2,2),strides=None));
    # 3rd Convolution layer output_shape = 6x38x16
    model.add(Convolution2D(16,3,3,subsample=(1,1),border_mode='valid'));#,activation='ELU'));
    model.add(ELU());
    model.add(Dropout(0.3));

    # Flatten the output
    model.add(Flatten())
    # 4th layer : Dense output_shape = 512
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dropout(0.3))
    # 5th layer: Dense output_shape = 256
    model.add(Dense(256))
    model.add(ELU())
    # Finally a single output, since this is a regression problem
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    if(Verbose==1):
        model.summary();
    return model;

def train_model(samples):

    """
    Splits the provided samples into training and validation data
    validation data size is chosen 20% of training data

    loads the best weights if model.h5 file exists, trains the model 
    and saves the best model after each epoch if validation loss improves
    also plots training and validation loss graph after the training completion

    # Argument
        samples : list - containing imgages and steering angle
    """

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
    #print(samples[0][0]);

    # balancing the dataset
    norm_samples=balance_dataset(samples);
    # training the model
    train_model(norm_samples);

    #balance_dataset(samples);#[COLUMN_INDEX_CENTER_IMG],samples[COLUMN_INDEX_STEERING_ANGLE])

    # print(samples[0][0])
    # img=np.array(plt.imread((samples[0][0])));
    # gen = get_generator(norm_samples);
    # x,y=next(gen)
    # rand_index=np.random.randint(0,len(x))
    # print("shape of input image : " , (x[rand_index].shape));
    # plt.figure("Random preprocessed Image");
    # plt.imshow((x[rand_index]).astype('uint8'));
    # plt.show();

    # For Visualizing Original and preprocessed Image
    # org_img=read_image_by_path('D://Term1//Project 3 Behaviour cloning//CarND-Behavioral-Cloning-P3-master//data//IMG//center_2016_12_01_13_30_48_287.jpg')
    # plt.figure("Original Image");
    # plt.imshow(org_img);
    # pre_img = get_preprocessed_image(org_img);
    # plt.figure("Preprocessed Image");
    # plt.imshow(pre_img);
    # plt.show();

