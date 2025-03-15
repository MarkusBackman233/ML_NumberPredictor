# Neural Network in C++ with Eigen  

This project implements a basic multi-layer neural network from scratch using C++ and the Eigen library for matrix operations.  

## Features  
-  **Feedforward Neural Network** with 2 hidden layers
-  **Sigmoid Activation Function** for non-linearity  
-  **Binary Cross-Entropy Loss Function** for cost calculation
-  **Gradient Descent Optimization** for weight updates
## How it works
Since im using Binary Cross-Entropy Loss Function the output exceede 1, therfor i am using 10 features(inputs) instead.

For an input of 4 the output layer will be:
```
0.000 0.000 0.001 0.000 0.998 0.000 0.000 0.000 0.000 0.000
```

### Training data
```
1 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0
0 0 1 0 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 1 0 0 0
0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 0 0 0 0 1
```
### Training labels
```
0 0 0 0 0 0 0 0 0 1
1 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0
0 0 1 0 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0
0 0 0 0 1 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 1 0 0 0
0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 1 0
```
## Dependencies  
- C++17 or later  
- [Eigen Library](https://eigen.tuxfamily.org/)  

## Usage  
The neural network is trained on a simple pattern where each number predicts the next. After training, you can input a number (0-9), and the network will attempt to predict the next number with a confidence percentage.  

