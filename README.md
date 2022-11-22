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
