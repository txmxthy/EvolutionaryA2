# Genetic Programming for Image Classification [20 marks]

Image classification is an important and fundamental task of assigning images to one of the pre-defined groups. Image classification has a wide range of applications in many domains, including bioinformatics, facial recognition, remote sensing, and healthcare. 

To achieve highly accurate image classification, a range of global and local features must be extracted from any image to be classified. The extracted features are further utilised to build a classifier that is expected to assign the correct class labels to every image.

In this question, you are provided with two image datasets, namely FEI 1 and FEI 2 1. 

* The two datasets contain many benchmark images for facial expression classification. 
* Example images from the two datasets are given in Figure 1. 
* All images contain human faces with varied facial expressions. 
* They are organised into two separate sets of images, one for training and one for testing. 
* Your task is to build an image classifier for each dataset that can accurately classify any image into two different classes, i.e., “Smile” and “Neutral ”. 
* There are two steps that you need to perform to achieve this goal, as described in subsequent subsections.


## 4.1 Automatic Feature Extraction through GP

In this subsection, we use the GP algorithm (i.e., FLGP) introduced in the lectures to design image feature extractors automatically. You will use the provided strongly-typed GP code in Python to automatically learn suitable images features respectively for the FEI 1 and FEI 2 datasets, identify the best feature extractors evolved by GP for both datasets and interpret why the evolved feature extractors can extract useful features for facial expression classification. Based on the evolved feature extractors, create two pattern files: one contains training examples and one contains test (unseen) examples, for both the FEI 1 and FEI 2 datasets.

Every image example is associated with one instance vector in the pattern files. Each instance vector has two parts: the input part which contains the value of the extracted features for the image; and the output part which is the class label (“Smile” or “Neutral ”). The class label can be simply a number (e.g., 0 or 1) in the pattern files. Choose an appropriate format (ARFF, Data/Name, CSV, etc) for you pattern files. Comma Separated Values (CSV) format is a good choice; it can easily be converted to other formats. Include a compressed version of your generated data sets in your submission.

## 4.2 Image Classification Using Features Extracted by GP

Train an image classifier of your choice (e.g., Linear SVM or Na ̈ıve Bayes classifier) using the training data and test its performance on the unseen test data that are obtained from the previous step (Subsection 4.1). Choose appropriate evaluation criteria (such as classification accuracy) to measure the performance of the trained classifier on both training and test data. Present and discuss the evaluation results.

Study the best GP trees obtained by you from the previous step (Subsection 4.1), with respect to both the FEI 1 and FEI 2 datasets. Identify and briefly describe all the global and local image features that can be extract by the GP trees from the images to be classified. Explain why the extracted global and local image features can enable the image classifier to achieve good classification accuracy.