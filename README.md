# Axiom ML
## Overview
Axiom ML, an educational project to implement the deep learning methods and framework buildup from ground. It is not a competition to existing frameworks such as PyTorch or Tensorflow.
Indeed, a learning project for gaining deep knowledge behind neural network and it's core mathematics. Each feature and method is explained below with mathematical explanation.
The project is still evolving and under testing with 2 unit tests written in pytest.

To contribute to the project, please refer to below [CONTRIBUTE](#Contribute) section. 
Some of the features that will be implemented in future are numerical stability to handle exploding gradients and make it more memory efficient and testing classification model on minst dataset.

## Features
### **Layers**
* **Layer -** Base class to all the neural network classes such as *activation functions*, *DenseLayer* and *models*. If you're building your own *layer*, *normalisation* or *model*, use Layer as the parent class and override the `forward()` as well as `backward()` methods. And, if you make constructor `__init__()`, call `super()` before everything in it.
  
* **Dense Layer -**  Which is a *linear trasnformation* layer.
    $$Z^{[1]}=W^{[1]}a^{[0]} + b^{[1]}$$
### **Activation Functions**
* **Binary Step Function -** Most simple activation function

* **Linear Function -** simple transformation function where output is directly propotional to the input. Also called <ins>***Identity function***</ins>. 

* **Sigmoid Function -** Transform the values between range 0 and 1.

* **Tanh -** Transforms the values in range from -1 to 1. And, is symmetric around origin.

* **Rectified Linear Unit Function -** It deactivates the neuron if the output of [linear transformation](#Layers) if it is a negetive. Thus, causing zero gradients and kills the neuron.

* **Leaky ReLU -** Which is a modified version of <ins>*ReLU*</ins>. Instead of deactivating neurons when `x < 0`, it multiply by a small decimal number `0.01`.

* **Parameterised ReLU -** Another modified version of <ins>*ReLU*</ins> to solve the [dead neuron](https://arxiv.org/pdf/2302.05818) problem. It solves this by introducing a slope (which is `Hyperparameter`).

* **Exponential Linear Unit -** Varient of <ins>*ReLU*</ins> that modifies the slope of the negetive part of the function. `ELU` uses a log curve for defining the negetive value.

* **Softmax -** Softmax returns the value between 0 and 1. It converts unnormalized logits into a valid probability distribution. $\sigma \in (0, 1)^k$, where $K > 1$.

### **Loss Function**
* **Mean Absolute Error -** commonly used in a regression problem, it is the average of sum of absolute difference between target and predicted values.

* **Mean Squared Error -** improved varient of *MAE*, it is the average squared difference of target and predicted values. It penalizes the large outliers.

* **Binary Cross Entropy -** also called *log loss* is a popular loss function in [Binary classification](https://en.wikipedia.org/wiki/Binary_classification) problems.

* **Categorical Cross-Entropy -** Measures the difference between true label and predicted label. It is widely used in [Multi-label Classification](https://en.wikipedia.org/wiki/Multi-label_classification). `C` is the number of classes


### Backpropogation
In simpler terms, This is derivated of the [Layer](#layers) and [Activations](#activation-functions) combined together in chain rule of derivation to calculate the subsequent pervious layer weight, bias and input gradients. 

Suppose, we pass the input by *DenseLayer* first and then *ReLU*. Let `X` be the input feature of dimension `N x M`.
    $$Z^{[0]}=W^{[0]}X + b^{[0]}$$
    $$\alpha^{[1]} = max \{0, Z^{[0]}\}$$ 
    $$\hat{y} = \alpha^{[1]}$$

Now the output of the model will be the $\alpha^{[1]}$. If we do the backpropogation, it is started by calculating the `Loss gradient` using [Loss functions](#loss-function) and applying chain rule. 
    $$
    i.e. \frac{\partial \mathcal{L}}{\partial Z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \alpha^{[1]}} \cdot \frac{\partial \alpha^{[1]}}{\partial Z^{[0]}}
    $$

### Models
* **Linear Regression**
* **MultiLayer Perceptron** `Testing`

## Installation
clone the github repository by using the following command
```
git clone https://github.com/iprabhsidhu/Axiom-ML
```
Once completed, You've now cloned the repository.
This will clone the repo code into your local directory.
If you are a developer, use this command
```bash
pip install -e .[dev]
```
Otherwise, use this one.
```bash
pip install -e .
```
By running the above pip command, project's dependencies (optional : dev) are installed. You can now run or start coding in it. 

## Test
If you like to run tests and see the project's results of Linear Layer and Linear Regression Model. You must have installed development environment.
Run the following command in the terminal or Vscode.
```bash
pytest -s
```
The following command will run all the unit tests and also display the output such as Loss and graph.

These [features](#Features) are tests and untested marked by a tick.
### Layers
- [x] **DenseLayer**
### Activation functions
- [ ] **Binary Step Function**
- [ ] **Linear Function**
- [ ] **sigmoid**
- [ ] **tanh**
- [x] **ReLU**
- [ ] **Leaky ReLU**
- [ ] **Parameterised ReLU**
- [ ] **ELU**
- [ ] **Softmax**
### Loss functions
- [x] **Mean Absolute Error**
- [x] **Mean Squared Error**
- [ ] **Binary Cross-Entropy**
- [x] **Categorical Cross-Entropy**

## Contribute
If you are considering contributing to the project. Here are the [Contribution guideline for the project](./CONTRIBUTING.md)

## License
This project is licensed under MIT License - see the [LICENSE](./LICENSE) for details. 
