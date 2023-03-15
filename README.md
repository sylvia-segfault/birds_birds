# Presentation
https://drive.google.com/file/d/1-xqSXUD-yNe8QyX3zSYnUVne0IC2YcO9/view?usp=sharing

# Problem Statement
A major application of computer vision is tackling the problem of ‘image classification’, wherein the model classifies images into one of multiple predefined categories that it has been trained upon. In this project, we train a model that learns to classify images of birds into the bird species that is pictured in the image.

# Dataset
The dataset we used for this project can be found [here](https://www.kaggle.com/competitions/birds23wi/data).
Methodology

## Data Processing
First, in our data processing, we loaded the images and checked their sizes. The images were not of consistent dimensions and not square, so we needed to perform some data transformations in order to ensure relative consistency across the dataset so that the algorithm can better compare values. Then, we checked the number of images across each class. Since each label had a different number of training images associated with it, we could either downsample or use image augmentation. The minimum number of images in a class was only 10, which would give us a very poor training dataset size if we used downsampling. We decided to use image augmentation instead with a transformer, resizing each sample into a dimension of 256x256 using random cropping with 8 padding, random image flipping along the horizontal axis, as well as the most optimized values of mean and standard deviation for normalization. This increased the number of images of classes with less data, and meanwhile, gave us some samples with a consistent image size. We also split our dataset in a 9:1 ratio of training to validation data to tune our hyperparameters without using the testing dataset, which could bias our model.

## Model Architecture
### ResNet
We wanted to use a neural network to train our model for image classification. In the field of machine learning, it is generally agreed upon that ‘deeper’ neural networks are better able to learn, and are very good for the problem of image classification. However, deep neural networks suffer from two problems. Firstly, they struggle with ‘vanishing gradients’, which make the network learn slowly or even not at all because the partial derivatives at deeper layers become too small to have any effect. This degradation issue was addressed by He et al. in their paper ‘Deep Residual Learning for Image Recognition’, which introduced a deep residual learning framework which is easier to optimize and experiences accuracy gains from greatly increased depth as compared to “plain” neural networks. Given this, using a residual neural network was how we decided to start with our model architecture for this project. Specifically, we used resnets pre-trained on ImageNet from the PyTorch library. We started with ResNet 50, a residual neural network with 50 layers.

### ResNeXt
Xie et al. proposed a different deep neural network architecture for more accuracy, and less network complexity. ResNeXt inherited traits from ResNet, VGG and Inception, but introduces a new hyperparameter of cardinality, or the size of the set of transformations. This allows us to add in more layers while requiring fewer parameters than ResNet. We decided to see if ResNeXt could potentially give us even better results than the ResNet model we started out with, and saw a substantial improvement in accuracy. We used PyTorch’s pretrained ResNeXt 101 with a cardinality of 32.

### Hyperparameters
We started with resnet50 with its number of output features=555 (number of classes in the dataset) using the first train function mentioned in the implementation section.
```
EPOCHS = 15
IMG_SIZE = 128
BATCH_SIZE = 64
STEP_SIZE = 5
GAMMA = 0.1
DECAY = 0.00047
Training   accuracy: 0.906181
```
<br></br>
For the same neural network, we increased the image size and decreased the step size. This has improved the model performance by a small amount.
```
EPOCHS = 10
IMG_SIZE = 256
BATCH_SIZE = 64
STEP_SIZE = 3
GAMMA = 0.1
DECAY = 0.00047
Training   accuracy: 0.920444
Validation accuracy: 0.793622
```
<br></br>
We also tried adjusting the value of gamma. This also appeared to improve model performance by an amount as well. Gamma is a multiplicative factor that decreases the learning rate as the model learns, so it makes sense that a slight increase in this parameter helped increase training accuracy. However, increasing the gamma value past 0.3 only lead to a decrease in training and validation accuracy.
```
EPOCHS = 10
IMG_SIZE = 256
BATCH_SIZE = 64
STEP_SIZE = 3
GAMMA = 0.3
DECAY = 0.00047
Training   accuracy: 0.964558
Validation accuracy: 0.817993
```
<br></br>
Therefore, we then changed the learning rate scheduler by using the second train function mentioned in the implementation section and made the model train for more epochs. Training for longer epochs and using a fine-tuned learning rate scheduler greatly improved the model performance.
```
EPOCHS = 20
IMG_SIZE = 256
BATCH_SIZE = 64
DECAY = 0.00047
Learning rate schedule={0: 0.01, 3: 0.0075, 5: 0.005, 7: 0.0025, 9: 0.001}
Training   accuracy: 0.994669
Validation accuracy: 0.824994
```
<br></br>
In the final training run, we decreased the batch size and used a more complex pre-trained neural network model, resnext101_32x8d. Increasing the model complexity as well as decreasing the batch size to make the model generalize better improved the validation accuracy by a significant amount.
```
EPOCHS = 20
IMG_SIZE = 256
BATCH_SIZE = 32
DECAY = 0.00047
Learning rate schedule={0: 0.01, 3: 0.0075, 5: 0.005, 7: 0.0025, 9: 0.001}
Training   accuracy: 0.998502
Validation accuracy: 0.860254
```
<br></br>

Training for longer epochs allows the model to learn better, but it can take a very long time for a very complex neural network. The final training run took us over 10 hours. A small value of weight decay can reduce the model complexity and prevent the model from overfitting, and improve the generalization performance of deep neural networks. 

A complex model has multiple layers for learning features at various levels of abstraction, which is especially useful for an image classification task. The first layer learns basic features such as edges, the second layer recognizes shapes which are collections of edges, the third layer then trains to recognize features that are collections of shapes such as birds’ wings and legs, and the subsequent layers can recognize high-order features. This setup is much better at generalizing since all the intermediate features between image raw data and high-level classification were learned by the complex model. However, a deep neural network is very computationally expensive to train. We could use transfer learning to implement a model more quickly and efficiently, and thus we used a pre-trained resnet model to transfer features it already learned from different dataset to perform a similar task, which is bird classification in our case.

We utilized a learning rate scheduler with the help of optimizer’s stepLR function in our first try, to reduce the learning rate by multiplying the previous learning rate by 0.1 every 3 epochs. After fine tuning the learning rate schedule, we noticed an improvement in the model performance. Instead of multiplicatively reducing learning rates, we additively decreased each learning rate by 0.0025 around every 2 epochs for the first 10 epochs, and the learning rate remained constant for the rest of 10 epochs out of 20 training epochs in total. The conclusion is that a large learning rate allows fast convergence, but it can lead to large weight updates that jump over minima. A smaller learning rate makes the training process smoother by gradually guiding the optimizer towards the minima, but can get stuck or take a very long time to finish training. To solve these issues, we can use a learning rate scheduler to allow the model to learn fast and reach a set of weights with good values at the beginning of the training, and fine tune smaller learning rates to reach a higher accuracy after that.

In terms of image size, the given dataset had some images with very big dimensions. A larger input image requires the model to learn many more pixels which increases the training time, but we still want the image samples to be big enough for the model to learn well. As a result, we used an image size 256x256. For batch size, it is better to pick a moderate size that is neither too large nor too small. A large batch size can speed up the training process, but can reduce the model’s capacity to generalize.

Based on our experiments, we concluded that training for longer epochs, using a deep convolutional neural network, adjusting learning rates during training, using a moderate image size, a smaller batch size, and a small weight decay are best for image-classification type of tasks.

### Other Details
Our final model training can be summarized as follows:
- Final pretrained model used: ResNeXt 101 from PyTorch
- Loss function: Cross-entropy loss
- Optimizer: SGD

Final Hyperparameters
- Epochs: 20
- Image size: 256
- Batch size: 32
- Decay: 0.00047
- Learning rate schedule: {0: 0.01, 3: 0.0075, 5: 0.005, 7: 0.0025, 9: 0.001}

# Implementation
We adopted the function for processing the birds dataset, the train function, the accuracy function as well as the predict function from [ImageNet and Transfer Learning](https://colab.research.google.com/drive/1EBz4feoaUvz-o_yeMI27LEQBkvrXNc_4?usp=sharing#scrollTo=9qltoeXQQRJ-) and [Transfer Learning to Birds](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing#scrollTo=C5_LglWCs9Iu). In the following, we will discuss the components we customized on top of what we have adopted. We customized the component for extracting the birds dataset into Google Drive folders using the zipfile library. There was one component used to investigate image sizes in the dataset for the purpose of tuning the right size for image resizing and cropping, as mentioned above in the Data Processing section. We also added logic to split the training set into 90% training data and 10% validation data using random_split with a fixed seed 47 and DataLoader. This made it easier to load those data while computing the training and validation accuracies. We had two different kinds of train functions with the only difference being how we adjusted learning rates at different training epochs. The first train function experimented with using the optimizer’s learning rate scheduler by calling optim_lr_scheduler.StepLR with customized step_size and gamma to specify that: every time when the epoch advances to the next step_size, the learning rate should change by step_size. The second train function was adopted from the resources mentioned above and was also the one we used to get the final results.
Results
Our final model submission had a 99.9% training accuracy and 86% validation accuracy.

# Next Steps
Another model we may want to experiment with in the future is EfficientNet. EfficientNet is a CNN that is both smaller and faster than the models we looked at. For instance, EfficientNet-B3 has 7x fewer parameters than ResNeXt-101, while providing similar accuracy. This could have the potential to be a model we can train even faster, and therefore, potentially for more epochs within the same time period to get even more accurate results, without being too computationally expensive.

# Resources
He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

Xie, Saining, et al. "Aggregated residual transformations for deep neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
