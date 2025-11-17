# arXiv:1708.07747v2 [cs.LG] 15 Sep 2017

## Fashion-MNIST: a Novel Image Dataset for

## Benchmarking Machine Learning Algorithms

```
Han Xiao
Zalando Research
Mühlenstraße 25, 10243 Berlin
han.xiao@zalando.de
```
```
Kashif Rasul
Zalando Research
Mühlenstraße 25, 10243 Berlin
kashif.rasul@zalando.de
```
```
Roland Vollgraf
Zalando Research
Mühlenstraße 25, 10243 Berlin
roland.vollgraf@zalando.de
```
Abstract

```
We present Fashion-MNIST, a new dataset comprising of 28 × 28 grayscale
images of 70 , 000 fashion products from 10 categories, with 7 , 000 images
per category. The training set has 60 , 000 images and the test set has
10 , 000 images. Fashion-MNIST is intended to serve as a direct drop-
in replacement for the original MNIST dataset for benchmarking machine
learning algorithms, as it shares the same image size, data format and the
structure of training and testing splits. The dataset is freely available at
https://github.com/zalandoresearch/fashion-mnist.
```
1 Introduction

```
The MNIST dataset comprising of 10-class handwritten digits, was first introduced by LeCun et al.
[1998] in 1998. At that time one could not have foreseen the stellar rise of deep learning tech-
niques and their performance. Despite the fact that today deep learning can do so much the sim-
ple MNIST dataset has become the most widely used testbed in deep learning, surpassing CIFAR-
10 [Krizhevsky and Hinton, 2009] and ImageNet [Deng et al., 2009] in its popularity via Google
trends^1. Despite its simplicity its usage does not seem to be decreasing despite calls for it in the
deep learning community.
```
```
The reason MNIST is so popular has to do with its size, allowing deep learning researchers to quickly
check and prototype their algorithms. This is also complemented by the fact that all machine learning
libraries (e.g. scikit-learn) and deep learning frameworks (e.g. Tensorflow, Pytorch) provide helper
functions and convenient examples that use MNIST out of the box.
Our aim with this work is to create a good benchmark dataset which has all the accessibility of
MNIST, namely its small size, straightforward encoding andpermissive license. We took the ap-
proach of sticking to the 10 classes 70 , 000 grayscale images in the size of 28 × 28 as in the original
MNIST. In fact, the only change one needs to use this dataset is to change the URL from where the
MNIST dataset is fetched. Moreover, Fashion-MNIST poses a more challenging classification task
than the simple MNIST digits data, whereas the latter has been trained to accuracies above 99.7%
as reported in Wan et al. [2013], Ciregan et al. [2012].
```
```
We also looked at the EMNIST dataset provided by Cohen et al. [2017], an extended version of
MNIST that extends the number of classes by introducing uppercase and lowercase characters. How-
```
(^1) https://trends.google.com/trends/explore?date=all&q=mnist,CIFAR,ImageNet


ever, to be able to use it seamlessly one needs to not only extend the deep learning framework’s
MNIST helpers, but also change the underlying deep neural network to classify these extra classes.

2 Fashion-MNIST Dataset

Fashion-MNIST is based on the assortment on Zalando’s website^2. Every fashion product on Za-
lando has a set of pictures shot by professional photographers, demonstrating different aspects of
the product, i.e. front and back looks, details, looks with model and in an outfit. The original picture
has a light-gray background (hexadecimal color:#fdfdfd) and stored in 762 × 1000 JPEG format.
For efficiently serving different frontend components, theoriginal picture is resampled with multiple
resolutions, e.g. large, medium, small, thumbnail and tiny.

We use the front look thumbnail images of 70 , 000 unique products to build Fashion-MNIST. Those
products come from different gender groups: men, women, kids and neutral. In particular, white-
color products are not included in the dataset as they have low contrast to the background. The
thumbnails ( 51 × 73 ) are then fed into the following conversion pipeline, whichis visualized in
Figure 1.

1. Converting the input to a PNG image.
2. Trimming any edges that are close to the color of the cornerpixels. The “closeness” is
    defined by the distance within5%of the maximum possible intensity in RGB space.
3. Resizing the longest edge of the image to 28 by subsampling the pixels, i.e. some rows and
    columns are skipped over.
4. Sharpening pixels using a Gaussian operator of the radiusand standard deviation of 1. 0 ,
    with increasing effect near outlines.
5. Extending the shortest edge to 28 and put the image to the center of the canvas.
6. Negating the intensities of the image.
7. Converting the image to 8-bit grayscale pixels.

Figure 1: Diagram of the conversion process used to generateFashion-MNIST dataset. Two exam-
ples from dress and sandals categories are depicted, respectively. Each column represents a step
described in section 2.

```
Table 1: Files contained in the Fashion-MNIST dataset.
```
```
Name Description # Examples Size
train-images-idx3-ubyte.gz Training set images 60 ,000 25MBytes
train-labels-idx1-ubyte.gz Training set labels 60 ,000 140Bytes
t10k-images-idx3-ubyte.gz Test set images 10 ,000 4. 2 MBytes
t10k-labels-idx1-ubyte.gz Test set labels 10 , 000 92 Bytes
```
For the class labels, we use the silhouette code of the product. The silhouette code is manually
labeled by the in-house fashion experts and reviewed by a separate team at Zalando. Each product

(^2) Zalando is the Europe’s largest online fashion platform.http://www.zalando.com


contains only one silhouette code. Table 2 gives a summary ofall class labels in Fashion-MNIST
with examples for each class.

Finally, the dataset is divided into a training and a test set. The training set receives a randomly-
selected 6 , 000 examples from each class. Images and labels are stored in thesame file format as the
MNIST data set, which is designed for storing vectors and multidimensional matrices. The result
files are listed in Table 1. We sort examples by their labels while storing, resulting in smaller label
files after compression comparing to the MNIST. It is also easier to retrieve examples with a certain
class label. The data shuffling job is therefore left to the algorithm developer.

```
Table 2: Class names and example images in Fashion-MNIST dataset.
```
```
Label Description Examples
```
```
0 T-Shirt/Top
```
```
1 Trouser
```
```
2 Pullover
```
```
3 Dress
```
```
4 Coat
```
```
5 Sandals
```
```
6 Shirt
```
```
7 Sneaker
```
```
8 Bag
```
```
9 Ankle boots
```
3 Experiments

We provide some classification results in Table 3 to form a benchmark on this data set. All al-
gorithms are repeated 5 times by shuffling the training data and the average accuracyon the
test set is reported. The benchmark on the MNIST dataset is also included for a side-by-side
comparison. A more comprehensive table with explanations on the algorithms can be found on
https://github.com/zalandoresearch/fashion-mnist.

```
Table 3: Benchmark on Fashion-MNIST (Fashion) and MNIST.
```
```
Test Accuracy
Classifier Parameter Fashion MNIST
DecisionTreeClassifier criterion=entropymax_depth= 10 splitter=best 0. 798 0. 873
criterion=entropymax_depth= 10 splitter=random 0. 792 0. 861
criterion=entropymax_depth= 50 splitter=best 0. 789 0. 886
Continued on next page
```

## Table 3 – continued from previous page

```
Test Accuracy
```

Table 3 – continued from previous page

Test Accuracy

   - criterion=entropymax_depth= 100 splitter=best Classifier Parameter Fashion MNIST
   - criterion=ginimax_depth= 10 splitter=best
   - criterion=entropymax_depth= 50 splitter=random
   - criterion=entropymax_depth= 100 splitter=random
   - criterion=ginimax_depth= 100 splitter=best
   - criterion=ginimax_depth= 50 splitter=best
   - criterion=ginimax_depth= 10 splitter=random
   - criterion=ginimax_depth= 50 splitter=random
   - criterion=ginimax_depth= 100 splitter=random
- ExtraTreeClassifier criterion=ginimax_depth= 10 splitter=best
   - criterion=entropymax_depth= 100 splitter=best
   - criterion=entropymax_depth= 10 splitter=best
   - criterion=entropymax_depth= 50 splitter=best
   - criterion=ginimax_depth= 100 splitter=best
   - criterion=ginimax_depth= 50 splitter=best
   - criterion=entropymax_depth= 50 splitter=random
   - criterion=entropymax_depth= 100 splitter=random
   - criterion=ginimax_depth= 50 splitter=random
   - criterion=ginimax_depth= 100 splitter=random
   - criterion=ginimax_depth= 10 splitter=random
   - criterion=entropymax_depth= 10 splitter=random
- GaussianNB priors=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
- GradientBoostingClassifier n_estimators= 100 loss=deviancemax_depth=
   - n_estimators= 50 loss=deviancemax_depth=
   - n_estimators= 100 loss=deviancemax_depth=
   - n_estimators= 10 loss=deviancemax_depth=
   - n_estimators= 50 loss=deviancemax_depth=
   - n_estimators= 10 loss=deviancemax_depth=
   - n_estimators= 10 loss=deviancemax_depth=
- KNeighborsClassifier weights=distancen_neighbors= 5 p=
   - weights=distancen_neighbors= 9 p=
   - weights=uniformn_neighbors= 9 p=
   - weights=uniformn_neighbors= 5 p=
   - weights=distancen_neighbors= 5 p=
   - weights=distancen_neighbors= 9 p=
   - weights=uniformn_neighbors= 5 p=
   - weights=uniformn_neighbors= 9 p=
   - weights=distancen_neighbors= 1 p=
   - weights=uniformn_neighbors= 1 p=
   - weights=uniformn_neighbors= 1 p=
   - weights=distancen_neighbors= 1 p=
- LinearSVC loss=hingeC= 1 multi_class=ovrpenalty=l2
   - loss=hingeC= 1 multi_class=crammer_singerpenalty=l2
   - loss=squared_hingeC= 1 multi_class=crammer_singerpenalty=l2
   - loss=squared_hingeC= 1 multi_class=crammer_singerpenalty=l1
   - loss=hingeC= 1 multi_class=crammer_singerpenalty=l1
   - loss=squared_hingeC= 1 multi_class=ovrpenalty=l2
   - loss=squared_hingeC= 10 multi_class=ovrpenalty=l2
   - loss=squared_hingeC= 100 multi_class=ovrpenalty=l2
   - loss=hingeC= 10 multi_class=ovrpenalty=l2
   - loss=hingeC= 100 multi_class=ovrpenalty=l2
   - loss=hingeC= 10 multi_class=crammer_singerpenalty=l1 Classifier Parameter Fashion MNIST
   - loss=hingeC= 10 multi_class=crammer_singerpenalty=l2
   - loss=squared_hingeC= 10 multi_class=crammer_singerpenalty=l2
   - loss=squared_hingeC= 10 multi_class=crammer_singerpenalty=l1
   - loss=hingeC= 100 multi_class=crammer_singerpenalty=l1
   - loss=hingeC= 100 multi_class=crammer_singerpenalty=l2
   - loss=squared_hingeC= 100 multi_class=crammer_singerpenalty=l1
   - loss=squared_hingeC= 100 multi_class=crammer_singerpenalty=l2
- LogisticRegression C= 1 multi_class=ovrpenalty=l1
   - C= 1 multi_class=ovrpenalty=l2
   - C= 10 multi_class=ovrpenalty=l2
   - C= 10 multi_class=ovrpenalty=l1
   - C= 100 multi_class=ovrpenalty=l2
- MLPClassifier activation=reluhidden_layer_sizes=[100]
   - activation=reluhidden_layer_sizes=[100, 10]
   - activation=tanhhidden_layer_sizes=[100]
   - activation=tanhhidden_layer_sizes=[100, 10]
   - activation=reluhidden_layer_sizes=[10, 10]
   - activation=reluhidden_layer_sizes=[10]
   - activation=tanhhidden_layer_sizes=[10, 10]
   - activation=tanhhidden_layer_sizes=[10]
- PassiveAggressiveClassifier C=
   - C=
   - C=
- Perceptron penalty=l1
   - penalty=l2
   - penalty=elasticnet
- RandomForestClassifier n_estimators= 100 criterion=entropymax_depth=
   - n_estimators= 100 criterion=ginimax_depth=
   - n_estimators= 50 criterion=entropymax_depth=
   - n_estimators= 100 criterion=entropymax_depth=
   - n_estimators= 50 criterion=entropymax_depth=
   - n_estimators= 100 criterion=ginimax_depth=
   - n_estimators= 50 criterion=ginimax_depth=
   - n_estimators= 50 criterion=ginimax_depth=
   - n_estimators= 10 criterion=entropymax_depth=
   - n_estimators= 10 criterion=entropymax_depth=
   - n_estimators= 10 criterion=ginimax_depth=
   - n_estimators= 10 criterion=ginimax_depth=
   - n_estimators= 50 criterion=entropymax_depth=
   - n_estimators= 100 criterion=entropymax_depth=
   - n_estimators= 100 criterion=ginimax_depth=
   - n_estimators= 50 criterion=ginimax_depth=
   - n_estimators= 10 criterion=entropymax_depth=
   - n_estimators= 10 criterion=ginimax_depth=
- SGDClassifier loss=hingepenalty=l2
   - loss=perceptronpenalty=l1
   - loss=modified_huberpenalty=l1
   - loss=modified_huberpenalty=l2
   - loss=logpenalty=elasticnet
   - loss=hingepenalty=elasticnet


```
Table 3 – continued from previous page
Test Accuracy
Classifier Parameter Fashion MNIST
loss=squared_hingepenalty=elasticnet 0. 815 0. 914
loss=hingepenalty=l1 0. 815 0. 911
loss=logpenalty=l1 0. 815 0. 910
loss=perceptronpenalty=l2 0. 814 0. 913
loss=perceptronpenalty=elasticnet 0. 814 0. 912
loss=squared_hingepenalty=l2 0. 814 0. 912
loss=modified_huberpenalty=elasticnet 0. 813 0. 914
loss=logpenalty=l2 0. 813 0. 913
loss=squared_hingepenalty=l1 0. 813 0. 911
SVC C= 10 kernel=rbf 0. 897 0. 973
C= 10 kernel=poly 0. 891 0. 976
C= 100 kernel=poly 0. 890 0. 978
C= 100 kernel=rbf 0. 890 0. 972
C= 1 kernel=rbf 0. 879 0. 966
C= 1 kernel=poly 0. 873 0. 957
C= 1 kernel=linear 0. 839 0. 929
C= 10 kernel=linear 0. 829 0. 927
C= 100 kernel=linear 0. 827 0. 926
C= 1 kernel=sigmoid 0. 678 0. 898
C= 10 kernel=sigmoid 0. 671 0. 873
C= 100 kernel=sigmoid 0. 664 0. 868
```
4 Conclusions

This paper introduced Fashion-MNIST, a fashion product images dataset intended to be a drop-
in replacement of MNIST and whilst providing a more challenging alternative for benchmarking
machine learning algorithm. The images in Fashion-MNIST are converted to a format that matches
that of the MNIST dataset, making it immediately compatiblewith any machine learning package
capable of working with the original MNIST dataset.

References

D. Ciregan, U. Meier, and J. Schmidhuber. Multi-column deepneural networks for image classifi-
cation. InComputer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on, pages
3642–3649. IEEE, 2012.

G. Cohen, S. Afshar, J. Tapson, and A. van Schaik. Emnist: an extension of mnist to handwritten
letters.arXiv preprint arXiv:1702.05373, 2017.

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical im-
age database. InComputer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference
on, pages 248–255. IEEE, 2009.

A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. 2009.

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document
recognition.Proceedings of the IEEE, 86(11):2278–2324, 1998.

L. Wan, M. Zeiler, S. Zhang, Y. L. Cun, and R. Fergus. Regularization of neural networks using
dropconnect. InProceedings of the 30th international conference on machine learning (ICML-
13), pages 1058–1066, 2013.


