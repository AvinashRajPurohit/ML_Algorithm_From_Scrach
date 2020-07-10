For More Post of Data science Click here [**Data_Science_Blogs**](https://datascienceblogs.herokuapp.com/).
# 1.ML_Algorithm_From_Scrach
This repository will contain all the basic algorithm from scrach in python with blog which contain explaination of algorithm
## Description Of All The Algos
### K-Nearest Neighbors Algorithm
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




# 2. Proportional Sampling From Scratch


### Proportional Sampling is a very important concept of probability and Machine learning.

#### Why should we learn Proportional Sampling -:
--> According to a 'methods.sagepub.com' "Numerous researchers recommend proportional sampling because it is relatively more efficient than the uniform sampling or simple random sampling on the whole study region."

#### * What is the purpose of proportional sampling?
--> Let suppose we have an Array/List of 'n' elements.
--> If we pick an element from the list randomly than each element has equal probable for picking and we don't want to pick randomly.
--> Now pick an element from the list of 'n' elements such that the probability of picking an element is proportional to the value at the corresponding index. so we pick the element randomly but the value which is high in the list has more probability for picking up and the probability of picking a small value is significantly small.

Algorithm of Proportional Sampling -:
Let suppose we have a list of n elements = d[] = [1.2, 3.3, 5.3, 1.2, 23]
## 1. Step-1 =:
            * Compute the sum of all the values in the list called 'S' -:
                                                      ------> S = Σdi's = 34                   
            * divide each value of list by 'S' and store in d_delta -:
                                                      ------> d_delta[] = di's/S = [0.035, 0.097, 0.155, 0.035, 0.676 ] = all value lies between 0 to1.
            * compute the cumulative normalized sum of each value in 'd_delta' and store in 'd_tilda-:
                                                      ------> d_tilda[0] = d_delta[0]  ||  d_tilda[1] = d_tilda[0] + d_detla[1]  || d_tilda[2] = d_tilda[1] + d_detla[2]...... and so on
              ------> d_tilda[] = [0.035, 0.132, 0.286, 0.322, 0.1]
              we all know what is the meaning of cumulative sum. 
              From the Step-1 we have finally a list d_tilda which have all the comulative sum of d_delta.


## 2. Step-2 =: Sample one value uniformly and store in 'unif'
                       -------> python->unif  = np.random.uniform(0.0,1.0,1)

## 3.Step-3 => now compare 'unif' value with each value in d_tilda and if unif is less than d_tilda value than return the value of d_tilda.
                             for i in d_tilda:
                                  if unif <= i:
                                      return i

## GitHub code for proportional sampling-:
----> https://github.com/AvinashRajPurohit/ML_Algorithm_From_Scrach/blob/master/2.Perportional_Sampling_Algo.ipynb
