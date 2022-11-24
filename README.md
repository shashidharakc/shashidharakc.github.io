## **Back Propagation In Cross Entropy Loss with Softmax**


In this Blog/ notebook we will discuss how weights of the network are updated/ adjusted in a simple classification model. For this we are considering Iris dataset which has 3 class, 4 features for each data point. We will build our netowrk using JAX both Stax api and custom model, we will use Categorical Cross entropy loss with softmax as final layer, with no hidden layer. No hidden layer cos of simplicity of network, we are not building a model that solves the problem with great accuracy, but we just want to understand how weights are adjusted based on the error. By the end of this blog we should have a better understanding of how the weights are updated in classification model.

Architicture of the model is as shown below diagram.

Below is the same diagram updated with some variables/constants updates with random values.

Apart from the inputs and weights we have shown in the figure above, lets say we have True Value = [1.0, 0.0, 0.0] for the give data points. Keeping these values lets go through step by step on how we update the weights.



![](https://github.com/shashidharakc/mynotes.github.io/blob/main/BlogImgs/CrossEntropyNet.drawio.png)



## Forward Prop

Now Lets go through forward pass. What we have so far is input data, ***X*** nd randomly initiallized weights ***W***, easy of understanding we will ommit the bias and procced. 

Lets Calculate ![](https://latex.codecogs.com/gif.latex?Z_%7B1%7D%5E%7B1%7D), this is summation of product of input and weights, generally ![](https://latex.codecogs.com/gif.latex?Z%20%3D%20%5Csum_%7Bi%7D%20%3D%20x_%7Bi%7D%20*%20w_%7Bi%7D). 

so, ![test equation](https://latex.codecogs.com/gif.latex?Z_%7B1%7D%5E%7B1%7D%20%3D%20x%5E%7B1%7D%20*%20w_%7B11%7D%5E%7B1%7D%20&plus;%20x%5E%7B2%7D%20*%20w_%7B12%7D%5E%7B1%7D%20&plus;%20x%5E%7B3%7D%20*%20w_%7B13%7D%5E%7B1%7D%20&plus;%20x_%7B4%7D%20*%20w_%7B14%7D%5E%7B1%7D)

When we plug in the number which yields 

![](https://latex.codecogs.com/gif.latex?Z_%7B1%7D%5E%7B1%7D%20%3D%20%280.81%20*%200.15%29%20&plus;%20%280.57%20*%200.29%29%20&plus;%20%280.28%20*%200.10%29%20&plus;%20%280.84%20*%200.71%29%20%3D%200.91119)

similarly ![](https://latex.codecogs.com/gif.latex?Z_%7B2%7D%5E%7B1%7D%20%3D%200.6105%2C%20Z_%7B3%7D%5E%7B1%7D%20%3D%20-0.1517)

Now we calculate sofmax of the $Z$ ouput using below equation 

![](https://latex.codecogs.com/gif.latex?softmax%28Y_%7Bi%7D%29%20%3D%20%5Cfrac%20%7Be%5E%7By_%7Bi%7D%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BJ%7D%20e%5E%7By_%7Bj%7D%7D%20%7D)

Plugging the logits i.e $Z$ to the above softamx equation we get 

![](https://latex.codecogs.com/gif.latex?0.4794433%2C%200.35493179%2C%200.16562491)

Categorical cross-entropy loss is give by below equation, lets calculate the loss using all the data we have so far.

![](https://latex.codecogs.com/gif.latex?CrosEntropyLoss%20%3D%20%5Csum_%7Bh%7D%20y_h%20log%28%5Chat%20y_h%29)

![](https://latex.codecogs.com/gif.latex?E%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20L%28y%2C%20%5Chat%20y%29)

![](https://latex.codecogs.com/gif.latex?L_%7BCE%7D%20%3D%20-%20%5Csum_%7Bi%3D1%7D%20T_i%20log%28S_i%29)

Lets plugin the numbers we have and calculate the loss.

![](https://latex.codecogs.com/gif.latex?-%5B1.0%20*%20log_2%280.4794433%29%20&plus;%200.0%20*%20log_2%28%200.35493179%29%20&plus;%200.0%20*%20log_2%280.16562491%29%5D)

![](https://latex.codecogs.com/gif.latex?CrosEntropyLoss%20%3D%20-log_2%280.4794433%29)

<img src=https://latex.codecogs.com/gif.latex?CrosEntropyLoss%20%3D%201.0605678>
