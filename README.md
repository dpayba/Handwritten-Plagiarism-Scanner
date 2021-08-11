INTRODUCTION
------------
Given two images of handwritten text, this machine learning algorithm 
scans and looks for any plagiarism in the two images

### Built With
* [Tensorflow](https://www.tensorflow.org/)
* [Numpy](https://numpy.org/)
* [Cv2](https://pypi.org/project/opencv-python/)
* [SciKitLearn](https://scikit-learn.org/stable/)

DEMO
-------------
Below, the input images are displayed followed by the recognized text. 
After both images are recognized, they are scanned against each other for similarities.

<img src="data/image4.png" width="300">

```
Recognized: "or work on line level"
Probability: 0.30403316020965576
```

<img src="data/line.png" width="300">

```
Recognized: "or work on line level"
Probability: 0.6674364805221558
```

```
file2.txt and file1.txt are 100.00% similar.
```