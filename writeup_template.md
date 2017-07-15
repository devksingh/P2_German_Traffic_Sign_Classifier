#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/devksingh/udacity_traffic_sign_classifier/blob/master/Traffic_Sign_Classifier_proj_final.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text](https://github.com/devksingh/udacity_traffic_sign_classifier/blob/master/trafficsign_chart.png)

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At first I decided to run the cnn model with all the 3 colour channel which gave accuracy of the model close to 90%. But my rationale behind converting the images to gray scale is:
1. We dont have have images which differes only on colour basis, like we dont have to differenciate between red guitar or black guitar. The images are distinct in feature and having RGB colour channel would not make any difference. I tried this as well and it did not make any diffrence. Therefore I went ahead with gray images so that model will have less parameters.
Secondly I normalised the image data so that it ranges between -1 to +1, which is considered as best practice for training a model. So I tried (X_train/127.5-1), although I tried training the model without normalization and results were pretty bad.
Thirdly I tried to augment the data inserting flip of the image by using numpy.fliplr function and rotated the image as wel., again it did not make any difference.
My aim was to go with lighter model so I tried many things multiple time and if did not make any difference then I removed it.
In the mean time I have also asked my mentor for good data augmentation tips and I would try to impletement those in future projects. I also watched stanford CS231n lectures for tips.
Here is an example of a traffic sign image before and after grayscaling.
![rgb scale](https://github.com/devksingh/udacity_traffic_sign_classifier/blob/master/rgb_image1.png)
![gray scale](https://github.com/devksingh/udacity_traffic_sign_classifier/blob/master/gray_image1.png)

As a last step, I normalized the image data to make it in -1 to +1 range.



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	     	| 1x1 stride, same padding, outputs 10x10x16 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 5x5x16 = 400        									|
| Fully connected		| 400 to 122        									|
| Fully connected dropour		| prob=.5        									|
| Fully connected		| 84 to 43        									|
| Fully connected dropour		| prob=.5        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I trialed by varying all the parameters and used an epoch of 30 and batch size of 124 with learning rate .00097. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of .995
* validation set accuracy of .956 
* test set accuracy of .933

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? I chose lenet to start with as I had tried it on mnist and it worked really well. 
* What were some problems with the initial architecture? I started with lenet only and after few calibrations itr worked well.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I started first without dropout and then added dropout. I also tried to add one more layer between 2 and 3 but it did not improve the result. I also tried to augment the data set with flip images on train dataset but it did not impro the result so I went ahead with lightweight model.
* Which parameters were tuned? How were they adjusted and why?
I tuned number of layers, tried to add labels, then I calibrated learning rate, Epoch and batch size. I also played with number of filters at each label but again went with 6, 16 for layer one and two.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Adding dropout layer improved the lenet model result. Adding/deleting layers and filters did not do any wonder.

If a well known architecture was chosen:
* What architecture was chosen? 
  I chose LeNet model, which we used on mnist data as part of exercise before.
* Why did you believe it would be relevant to the traffic sign application? 
  It worked well on mnist so i decided to use again.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
I always got accuracy between 90 to 97%
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text](https://github.com/devksingh/udacity_traffic_sign_classifier/blob/master/sign1.png) 
![alt text](https://github.com/devksingh/udacity_traffic_sign_classifier/blob/master/sign2.png) 
![alt text](https://github.com/devksingh/udacity_traffic_sign_classifier/blob/master/sign3.png) 
![alt text](https://github.com/devksingh/udacity_traffic_sign_classifier/blob/master/sign4.png) 
![alt text](https://github.com/devksingh/udacity_traffic_sign_classifier/blob/master/sign5.png)

The speed limit 80 image might be difficult to classify because half of digit 80 looks like 30 that's why cnn was not confident for this image.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 80 km/ph      		| 30 km/ph   									| 
| No Passing     			| No Passing 										|
| Turn Left Ahead					| Turn Left Ahead											|
| Ahead Only	      		| Ahead Only					 				|
| 100 km/ph			| 100 km/ph      							|




The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.3%...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

The top five soft max probabilities for the 80 km/ph image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| 30 km/ph   									| 
| .06     				| 50 km/ph 										|
| .00					| 70 km/ph											|
| .00	      			| Stop					 				|
| .00				    | Roundabout      							|


For the No Passing image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No Passing   									| 
| .00     				| No passing for vehicles over 3.5 metric tons 										|
| .00					| Dangerous curve to the left											|
| .00	      			| Dangerous curve to the right					 				|
| .00				    | End of all speed and passing limits      							|

For the Turn left ahead image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn left ahead   									| 
| .00     				| Keep right										|
| .00					| Turn right ahead											|
| .00	      			| Stop					 				|
| .00				    | Ahead only      							|

For the 100 km/h image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .41         			| 100 km/h   									| 
| .27     				| 30 km/h 										|
| .14					| Roundabout											|
| .06	      			| 80 km/h					 				|
| .03				    | 50 km/h      							|

For the Ahead only image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead only   									| 
| .00     				| Road work 										|
| .00					| 60 km/h											|
| .00	      			| Yield					 				|
| .00				    | Go straight or right      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


