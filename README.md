# shashidharakc.github.io

Test 


Apart from the inputs and weights we have shown in the figure above, lets say we have True Value = [1.0, 0.0, 0.0] for the give data points. Keeping these values lets go through step by step on how we update the weights.

### Forward Prop

Now Lets go through a step of forward pass. What we have so far is input data, $X$ nd randomly initiallized weights $W$. Again for making the calcuation easy and of easy of understanding we will ommit the bias and procced. 

Lets Calculate $Z_{1}^{1}$, this is summation of product of input and weights, generally  

$$Z = \sum_{i} = x_{i}*w{i}$$. 

so, 

$$Z_{1}^{1} = x^{1} * w_{11}^{1} + x^{2} * w_{12}^{1} + x^{3} * w_{13}^{1} + x_{4} * w_{14}^{1}$$

When we plug in the number which yields 

$Z_{1}^{1} = (0.81 * 0.15) + (0.57 * 0.29) + (0.28 * 0.10) + (0.84 * 0.71) = 0.91119$

similarly $Z_{2}^{1} = 0.6105, Z_{3}^{1} = -0.1517$

Now we calculate sofmax of the $Z$ ouput using below equation 

$softmax(Y_{i}) = \frac {e^{y_{i}}}{\sum_{j=1}^{J} e^{y_{j}} }$

Plugging the logits i.e $Z$ to the above softamx equation we get 

Categorical cross-entropy loss is give by below equation $0.4794433, 0.35493179, 0.16562491$

Now Cross entropy loss is given by below equation, lets calculate the loss, 

$ CrosEntropyLoss = \sum_{h} y_h log(\hat y_h)$

$ E(\theta) = \frac{1}{m} \sum_{i=1}^{m} L(y, \hat y)$

$ L_{CE} = - \sum_{i=1} T_i log(S_i)$

Lets plugin the numbers we have and calculate the loss.

$        = -[1.0 * log_2(0.4794433) + 0.0 * log_2( 0.35493179) + 0.0 * log_2(0.16562491)] $

$        = - log_2(0.4794433) $

$        = 1.0605678 $


### Back Prop

Lets see how we can update/adjust the weights based on the information we have in hand. Using chain rule we can come to below equation for updating give weights

$\frac{\partial E_{total}}{\partial W_{j}} = \frac{\partial E_{total}}{\partial yHat} * \frac{\partial yHat}{\partial Z^{1}_{i}} * \frac{\partial Z^{1}_{i}}{\partial W_{j}}$

Based on the derivative of Crossentropy with softmax, we can replace the first two terms of the right hand side as show below.

$\frac{\partial E_{total}}{\partial yHat} * \frac{\partial yHat}{\partial Z^{1}_{i}}  = (\hat y_{i} - y_{i}) $ .

Then, 

$\frac{\partial E_{total}}{\partial yHat} * \frac{\partial yHat}{\partial Z^{1}_{i}} * \frac{\partial Z^{1}_{i}}{\partial W_{1}} = (\hat y_{i} - y_{i}) * \frac{\partial Z^{1}_{i}}{\partial W_{1}}$

Now the final term of the left hand side can be put below.

$\frac{\partial Z^{1}_{i}}{\partial W_{j}} = \frac {\partial W_{1}^{ij} * X_{1} + W_{2}^{ij} * X_{2} + W_{3}^{ij} * X_{3} + W_{4}^{ij} * X_{4}} {\partial W_{j}}$

Apart from term where $j$ is equal, rest of the terms are constants so will be equated to zero. 

Putting it all together, in a consince and generic terms,

$\frac{\partial E_{total}}{\partial W_{j}} = (\hat y_{i} - y_{i}) * X_{j}$

Where : $i = class, j = feature$


So lets put all the values of the variables and constants we have in hand

$Y = [1.0, 0.0, 0.0]$

$\hat y = [0.7, 0.2, 0.1]$ 

$\hat y = [0.479, 0.355, 0.166]$

$X = [0.81, 0.57, 0.28, 0.84]$

$W^{1} = [0.15, 0.29, 0.10, 0.71]$

$CEL = 0.515$





Generally Derivate of Loss w.r.t weights $W_{i}$is given below. 

$\frac{\partial E_{total}}{\partial W_{i}} = \frac{\partial E_{total}}{\partial yHat} * \frac{\partial yHat}{\partial Z_{i}} * \frac{\partial Z_{i}}{\partial W_{i}}$

We have three logits $Z^{i}$, four weights $W^{i}$ and four features of the input data $X^{i}$. 
We have four weights we can adjust $W^{i}$, which is the learable components of the model. By adjusting theses weights we can try decrease the overall loss of the network. 






So as shown above derviatives, final eqatuion for the update can be given as below 

$\frac{\partial E_{total}}{\partial W_{j}} = (\hat y_{i} - y_{i}) * X_{j}$

Where : $i = class, j = feature$

So, 

$\frac{\partial E_{total}}{\partial W^{1}_{11}} = (\hat y_{1} - y_{1}) * X^{1}$ => $ = (0.7 - 1.0) * 0.81$ => $ = -0.2430$

$\frac{\partial E_{total}}{\partial W^{1}_{12}} = (\hat y_{1} - y_{1}) * X^{2}$ => $ = (0.7 - 1.0) * 0.57$ => $ = -0.1710$

$\frac{\partial E_{total}}{\partial W^{1}_{13}} = (\hat y_{1} - y_{1}) * X^{3}$ => $ = (0.7 - 1.0) * 0.28$ => $ = -0.0840$

$\frac{\partial E_{total}}{\partial W^{1}_{14}} = (\hat y_{1} - y_{1}) * X^{4}$ => $ = (0.7 - 1.0) * 0.84$ => $ = -0.2520$


$\frac{\partial E_{total}}{\partial W^{1}_{21}} = (\hat y_{2} - y_{2}) * X^{1}$ => $ = (0.2 - 0.0) * 0.81$ => $ = +0.1620$

$\frac{\partial E_{total}}{\partial W^{1}_{22}} = (\hat y_{2} - y_{2}) * X^{2}$ => $ = (0.2 - 0.0) * 0.57$ => $ = +0.1139$

$\frac{\partial E_{total}}{\partial W^{1}_{23}} = (\hat y_{2} - y_{2}) * X^{3}$ => $ = (0.2 - 0.0) * 0.28$ => $ = +0.0560$

$\frac{\partial E_{total}}{\partial W^{1}_{24}} = (\hat y_{2} - y_{2}) * X^{4}$ => $ = (0.2 - 0.0) * 0.84$ => $ = +0.1680$






$\frac{\partial E_{total}}{\partial W^{1}_{11}} = (\hat y_{1} - y_{1}) * X^{1}$ => $ = (0.479 - 1.0) * 0.81$ => $ = -0.4220$

$\frac{\partial E_{total}}{\partial W^{1}_{12}} = (\hat y_{1} - y_{1}) * X^{2}$ => $ = (0.479 - 1.0) * 0.57$ => $ = -0.2969$

$\frac{\partial E_{total}}{\partial W^{1}_{13}} = (\hat y_{1} - y_{1}) * X^{3}$ => $ = (0.479 - 1.0) * 0.28$ => $ = -0.1458$

$\frac{\partial E_{total}}{\partial W^{1}_{14}} = (\hat y_{1} - y_{1}) * X^{4}$ => $ = (0.479 - 1.0) * 0.84$ => $ = -0.4376$


$\frac{\partial E_{total}}{\partial W^{1}_{21}} = (\hat y_{2} - y_{2}) * X^{1}$ => $ = (0.355 - 0.0) * 0.81$ => $ = +0.2875$

$\frac{\partial E_{total}}{\partial W^{1}_{22}} = (\hat y_{2} - y_{2}) * X^{2}$ => $ = (0.355 - 0.0) * 0.57$ => $ = +0.2023$

$\frac{\partial E_{total}}{\partial W^{1}_{23}} = (\hat y_{2} - y_{2}) * X^{3}$ => $ = (0.355 - 0.0) * 0.28$ => $ = +0.0994$

$\frac{\partial E_{total}}{\partial W^{1}_{24}} = (\hat y_{2} - y_{2}) * X^{4}$ => $ = (0.355 - 0.0) * 0.84$ => $ = +0.2981$






We can update the the individual weights with the help of below equation, where $\alpha$ is a constant called stepsize, which controls the scalling of derivative of loss w.r.t the weight.

$\theta_{ij} : = \theta_{ij} - \alpha * \frac{\partial CE}{\partial \theta_{ij}}$

Now lets compute / update a weight based on the what we have computed so far. Let keep stepsize $\alpha = 0.25$, lets select weight $W_{11}^{1}$ which was initialized at $0.15$. 

Plugging in the above number to the update equation, 

$0.15 - (0.25 * -0.4220) = 0.2555 $ 


