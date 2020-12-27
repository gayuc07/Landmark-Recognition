# Popular Attraction/Landmark Recognition Using Google Landmark Dataset

With a rapid increase in the use of smartphones and other social apps, Image Recognition, Image Classification and Image Processing are the latest concepts that interest data engineers in computer vision tasks. A major challenge with image classification is the lack of a large, annotated dataset to train better and robust models. 

## Problem Statement

Recognizing and training the model to identify any landmark is a challenging task as the appearance of the landmark varies with geometry, illumination and a different aspect ratio of the image presented. To overcome this issue, a collection of images is used to capture typical appearance of the location. This project will focus to build a model that recognizes a given popular attraction or landmark using Google landmark dataset. This landmark recognition model will be handy to identify the name of a landmark in the image. This will also helpful for photo organization in smartphones and fields like aviation, maps, crime - solving, etc.

## Dataset

In order to capture the typical appearance of an image via a collection of images, we need a large annotated landmark dataset. Google has released its latest landmark dataset named, GoogleLandmarks-v2 (September 2019) which makes it our ideal choice for landmark recognition and retrieval purposes. This dataset includes over 5 million images with more than 200,000 diverse landmark classes. Google has published this dataset in 3 sets – train, index and test. The train and test files are used for landmark recognition and index file is used for retrieval purposes. Train dataset consists of image details of various landmarks, while test dataset consists of images that include no landmark, one landmark or multiple landmark. The major challenge while using this dataset is that of a highly imbalanced training dataset. This is because since there are large number of categories, also many classes with single digit training data which makes it difficult to classify and train the model for such classes.

- Train dataset – 4132,914 location data with 203,094 unique classes
- Test dataset – 117,577 data points

Since the dataset is highly imbalanced, performing data pre-processing needs to be considered before training the model. The dataset also needs to be cleaned to find any broken url (analyzing the image). The dataset is created by crowdsourcing the landmark available online. Each image might have different pixel size; hence these images need to be resized to one uniform pixel size for analysis and training. 

![Freq plot](https://github.com/gayuc07/Landmark-Recognition/blob/main/Images/Freq_Plot.JPG)

## Algorithm Used

In this project, we have used HOG classifier for feature extraction and created comparative study how dataset reacts to various classifier like Logistic Regression, SVM, Naïve Bayes, KNN, Random Forest, Decision Tree and ensemble - Voting Classifier.

## Experimental Setup

### Data Load & Preprocessing:

Train Dataset from Google landmark dataset is loaded, and top 10 sampled records are identified and stored. This data is used as source data for our project. Dataset is divided two parts – Train & Test sets. From image link given in URL, images are downloaded and saved in two folder – Train_image and Test_image. If image link in dataset is inaccessible or broken, id’s associated with data is added to errored is list. As downloaded image are of different dimension, to maintain uniformity, images are resized to aspect ratio – (256,256).

- Image_Download.py → This file describes image download and resize process. “download_prep” function is called from main function for every datapoint in train and test data. Once Image is processed, we have used HOG classifier for feature extraction from loaded images. 

- Feature_Extraction.py – This file contains “hog” function – which calculates the gradients and orientation for each pixel values in image and histogram is derived for each cell. The feature details are then stacked as Numpy array and final Numpy array contains feature of the image is returned.

This process is repeated for all images in the dataset, resulting array is saved train feature and test feature list set respectively. Associated labels are saved to test labels and train labels. These are used as input and target variables. To save computational time, as data download and feature extraction for 30k dataset is huge, we have preloaded the data and csv files containing feature and label details are used for analysis purposes.

## Modelling

As we have one input feature variable, we couldn’t able to hyper tune the parameters with respect to variables. We experimented with various model parameters that best fit for our dataset.

- Model_Function.py – This File contains model functions used for this project. It takes the train set feature and labels, fit the model and returns the predicted label set.

### Model Comparison

![Accuracy](https://github.com/gayuc07/Landmark-Recognition/blob/main/Images/acc.JPG)

From Accuracy Score and Kappa Score, we could say Random forest gives better accuracy rate and kappa value also falls under Fair agreement region, followed by Ensemble and Logistic Regression. SVM model has lowest accuracy and kappa score, hence, it doesn’t suit for given dataset.

![Evaluation](https://github.com/gayuc07/Landmark-Recognition/blob/main/Images/model_eval.JPG)

### Cross Validation Score

We reconfirm our result, we performed 10-fold cross-validation on trained set. Please find below result for the models.

![cross validation](https://github.com/gayuc07/Landmark-Recognition/blob/main/Images/cv.JPG)

The cross-validation score is similar, we have Random forest, logistic regression with better score and SVM models has least value.

## Conclusion

Landmark recognition model is built to classify top 10 sampled landmark id of google dataset. This project explored the possibility of building model with various machine learning algorithm. From comparative study, we could see Random Forest algorithm works best for given dataset followed by logistic Regression. In terms of ensemble model, Random forest with nonlinear SVM gives better classification Model. SVM model doesn’t suit for our dataset. As dataset is highly imbalance, its hard to find optimum boundary using SVM. Thus, random forest well suited for our landmark recognition data. However, Accuracy achieved is 68%, which is not great. As dataset is huge and imbalance, if we increase class scalability, these algorithms may not work best for recognizing landmark. In such cases we can use neural network may works better. Also, many classes have least datapoints, if we get more annotated images, prediction percentage may increase further.

## References

- Announcing Google-Landmarks-v2: An Improved Dataset for Landmark Recognition & Retrieval (2019, September),
Retrieved from: https://ai.googleblog.com/2019/05/announcing-google-landmarks-v2-improved.html

- The Common Visual Data Foundation(2019, September), Google Landmarks Dataset v2,
Retrieved from: https://www.kaggle.com/c/landmark-recognition-2019

- Y. Li, D. J. Crandal and D. P. Huttenlocher, Landmark Classification in Large-scale Image Collections,
Retrieved from: https://www.cs.cornell.edu/~yuli/papers/landmark.pdf

- A. Crudge, W. Thomas and K. Zhu, Landmark Recognition Using Machine Learning,
Retrieved from: http://cs229.stanford .edu/proj2014/Andrew%20Crudge, %20Will%20Thomas,%20Kaiyuan%20Zhu,%20Landmark%20Recognition%20Using%20Machine%20Learning.pdf

- Y. Takeuchi, P. Gros, M. Hebert and K. Ikeuchi, Visual Learning for Landmark Recognition,
Retrieved from: https://www.cs.cmu.edu/~takeuchi/iuw97/iuw97.html https://www.Analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/

- HOG Classifier Feature Engineering for Images: A Valuable Introduction to the HOG Feature Descriptor
Retrieved from:https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/





