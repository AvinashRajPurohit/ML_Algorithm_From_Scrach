For More Post of Data science Click here[Data_Science_Blogs](https://datascienceblogs.herokuapp.com/).
# ML_Algorithm_From_Scrach
This repository will contain all the basic algorithm from scrach in python with blog which contain explaination of algorithm
## Description Of All The Algos
**K-Nearest Neighbors Algorithm**
In wiki "In pattern recognition, the k-nearest neighbor algorithm is a non-parametric method proposed by Thomas Cover used for classification and regression."

In this post suppose we are doing sentiment classification of  Amazon fine-food reviews using KNN.
So we have a dataset of features and ground truth where ground truth is binary(0,1)

## "How KNN works?"
---> In this algorithm like the name suggest it find the nearest neighbors of a data point let here k = 4.
        1. -: Geometrically KNN first, find the 4-Nearest Neighbors and then check their corresponding class label/ground truth/Yi's 
        2. -: After that KNN takes the majority vote of the class labels if our 4 Nearest Neighbors have "1" class label  more than 3 times than our 
               predicting class label will be "1" eg. {x1="1", x2="1", x3= "1", x4= "0"} than we have "1" three times, our predicting class label will be 1.
       

K-NN fails when the data point is an outlier because geometrically K-NN works with distance if the outlier is the far away from out other data points in this situation our algorithm should say "I don't know"

Mathematically K-NN works with distance and distance measuring can be done with -:
1. Euclidian distance(L2-Norm) Norm refers to the total length of all the vectors in the space.
2. Manhatten distance(L1-Norm)
3. Co-sine distance
4. Hamming distance(text processing)

KNN will always give same output for a given set of input repeatedly tested. cause K-NN is an deterministic algorithm.

# KNN Limitations 
1. -: Takes too much space to store data in memory(RAM) O(nd)
2. -: takes too much time to process (O(nd))

In KNN "K" is a hyperparameter we can tune it via "K-fold cross-validation" means to choose the appropriate "K" for our algorithm, I will make a post on cross-validation"

