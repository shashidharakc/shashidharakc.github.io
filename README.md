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

$CrosEntropyLoss = \sum_{h} y_h log(\hat y_h)$

$E(\theta) = \frac{1}{m} \sum_{i=1}^{m} L(y, \hat y)$

$L_{CE} = - \sum_{i=1} T_i log(S_i)$



