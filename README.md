# Fruit_Classification
Fruit Classification using HOG discriptor and SVM classifier (From Scratch)

HOG discriptor and SVM classifier is an old-fashioned way for classification. The result maybe less efficient than using SOTA method (such as CNN,...)

## Dataset
Including 2 classes: Rotten and Fresh



<img src="https://user-images.githubusercontent.com/101311817/232272994-c19394c5-940f-4b99-8701-293785b01728.png" width="200" height="200"> <img src="https://user-images.githubusercontent.com/101311817/232273044-ed343b42-5250-484a-a63a-a4e5a1c2827f.png" width="200" height="200">

## Training

Each image will be computed its Hog features

And then, SVM classifer will classify these Hog features

Run **main.py**

## Result

### Training Loss and Accuracy

![Loss](https://user-images.githubusercontent.com/101311817/232274448-8babdcca-2209-4df2-bbee-371f8c6590db.png)

![Accuracy](https://user-images.githubusercontent.com/101311817/232274449-ff992547-6121-4a41-bf3d-6fb6d4527b64.png)

### Test result

Test accuracy = **89,26%**
