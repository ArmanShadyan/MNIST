# PyTorch MNIST Classifier

This notebook contains a simple neural network model built with PyTorch for classifying handwritten digits from the MNIST dataset.

## Data Preparation

The first part of the notebook handles data preparation. We download the MNIST dataset, which is a large database of handwritten digits commonly used for training image processing systems. The data is then transformed into tensors and normalized.

## Model Definition

Next, we define a simple feed-forward neural network with two hidden layers. The first layer takes in a flattened version of the MNIST images (which are 28x28 pixels, hence 784 inputs), and outputs 128 features. The second hidden layer takes these 128 features and outputs 64 features. The final layer takes these 64 features and outputs 10 values (one for each digit from 0 to 9), which are then passed through a LogSoftmax function for the final output.

## Loss Function

We use Negative Log-Likelihood Loss (NLLLoss), which is suitable for classification problems with C classes.

## Training

Finally, we get a batch of images and labels from the trainloader, flatten the images, pass them through the model to get the logits, and then calculate the loss between the logits and the labels. The `print(loss)` statement at the end will print out the loss of the model on the first batch of the training data. This gives us an idea of how well the model is doing before any backpropagation or gradient descent has occurred.
