# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

![Unprocessed_image_from_center][/output_images/center_image.png]
![flipped_center_image][/output_images/flipped_center_image.png]


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md  summarizing the results
* video.mp4  containing video for car runnning on track 1
* run_challenge.mp4 containing video for car running on track 2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around both the tracks by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it also contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used NVIDIA architecture for my model.
My model consists of a 3 convolution neural network with 5x5 filter sizes and 2 convolutional neural nerowrk with 3x3 filter sizes and depths between 32 and 64 . 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. 


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting . 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 80% of the data is used for training and 20% of the data is used for validation purpose.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I took udacity dataset for training purpose. For that i considered images from all the three cameras i.e. left_camera, right_camera and center camera. Also I flipped the images to increase the dataset(data augmentation).



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Initially i tried to use Lenet Architecture. But the model was failing where the car was gettin sharp turns. Then i implemented NVDIA architecture. To check how my model was performing I split my image and steering angle data into a training and validation set.

To combat the overfitting, I used the dropout technique This helped the car at sharp turns. At the end the car was able to drive autonoously around the track after applying the above techniques.

#### 2. Final Model Architecture

The final model architecture  consisted of a convolution neural network with the following layers and layer sizes :

--> Convolution Layer : 5x5 kernels with depth = 24
--> Convolution Layer : 5x5 kernels with depth = 36
--> Convolution Layer : 5x5 kernels with depth = 48
--> Convolution Layer : 3x3 kernels with depth = 64
--> Convolution Layer : 3x3 kernels with depth = 64
--> Fully Connected Layer : output size = 1000 
--> Fully Connected Layer : output size = 100 
--> Fully Connected Layer : output size = 50
--> Fully Connected Layer : output size = 10
--> Fully Connected Layer : output size = 1


Below is tha image showing summary of my model:


![model_summary][/output_images/model_summary.png]

#### 3. Creation of the Training Set & Training Process

I have used Udacity datset to train my model. Also i have used the images from left and right camera and then flipped all the 3 types of images in order to get more dataset.

Below is one of the image from center camera

![center_image][/output_images/center_image.png]

Below are 2 images from left and right camera:

![left_image][/output_images/left_image.png]
![right_image][/output_images/right_image.png]


Then I flipped all theses 3 types of images in order to augment the data:

![center_flipped_image][/output_images/flipped_center_image.png]
![left_flipped_image][/output_images/left_flipped_image.png]
![right_flipped_image][/output_images/flipped_right_image.png]

After the data collection, i normalized the data and then I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used number of epochs as 2 for my model.

The car was running fine on both the tracks after the above methodology was used.
