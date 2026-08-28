
## Neural Thickets
https://arxiv.org/pdf/2603.12228
- **Large-scale multitask pretraining doesn't merely learn one good generalist. It organizes weight space so that many specialized versions of that generalist are locally accessible.**
	- Small perturbations around $\theta^*$ even perform well in newly sampled instances of a learned task family; they aren't merely recovering exact memorized training examples
- Corroborates with literature showing that LLM fine-tuning often has very low intrinsic dimension, and often performs well even when very light-weight
	- i.e. why LoRA fine-tuning works

## Double Descent
- Traditionally, x-axis is model size (though similar trends are seen with training steps)
- There are 3 "regimes" of model sizes:
	- With small enough model, the model doesn't have enough representational power to memorize the dataset but is able to generalize
	- At the *interpolation point*, it has just enough expressiveness to memorize the dataset, but is extremely unstable
	- With a hugely overparameterized model, there are many possible parameter choices that fit the dataset exactly; regularization (either implicit in optimizers like SGD, or explicit like weight decay or drop-out) $\rightarrow$ the model learns a smooth function which has lowest regularization loss $\rightarrow$ able to generalize again (validation loss often reaches its lowest here)
<center><img src="ReadingNotesSupplements/double_descent.png" alt="" style="width:400px; margin-top: 10px"/></center>	
<center><sub><sup>X-axis can be model size or training steps</sub></sup></center>
- Argument by [Favero et. al.]([[2505.16959] Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models](https://arxiv.org/abs/2505.16959)): diffusion models do not exhibit double descent
	- Overfitting/"memorization" still occurs:
		- Consider partitioning a dataset and training a model on each dataset half.
		- Before memorization, both models predict very similar outputs; they learned the same, general score function.
			- However, these outputs look quite poor
		- After memorization, they diverge, each outputting exact images from their individual training sets.
		- ***GENERALIZATION HAPPENS BEFORE MEMORIZATION***
			- "Frequency Principle" -- NN fit lower-frequency patterns before higher frequency
				- This is simple intuition -- generalization is a lower-frequency signal, memorizing datapoints is high frequency
		<center><img src="ReadingNotesSupplements/generalization_before_memorization.png" alt="" style="width:700px; margin-top: 10px"/></center>	
	- Why double descent doesn't occur for diffusion models specifically:
		- In finite-sample regime, actual data distribution is a sum of a (normalized) dirac delta distribution for every data point
		- The loss-minimizing empirical score function is unique: once you have reached the capacity to learn the exact empirical score function, more training/larger model will not change anything
		- For general NNs, 

## Why Gradient Descent Works
In short: because of high dimensions
- THE CORE: moving in dimension $A$ causes the optimization landscape of dimension $B$ to change
	- Specifically, moving toward lower loss by moving in dimension $A$ lowers the loss wherever you are in dimension $B$
- If you look at any low-dim slice of a high-dim optimization landscape and path, it looks like a worm hole opens beneath your solution 
		<center><img src="ReadingNotesSupplements/gradient_descent_wormhole.png" alt="" style="width:800px; margin-top: 10px"/></center>	
	<center><sup><sup>2D Gradient Descent Example (with 1D slice)</center></sup></sup>
	- Low-dim slices give the illusion of barriers that don't exist in full-dim space.

- **More importantly, to truly get stuck in a *false* local minima, you need to get stuck in *every* dimension**
	- With billions of dimensions all changing at once, this is extremely unlikely
	- So gradient descent, as you run it for longer, continues to find lower and lower losses

- It's true: in the full-dimensional optimization problem, gradient descent will converge to the nearest local minima; but:
	1) with over-parameterized models, the space of optimal parameter sets is high dimensional (i.e. there is a huge null-space); so **most local minima more like connected low-loss basins**; they are the global minima
	2) ***false* (i.e. high-loss) local minima are mathematically improbable**

## Math
Dirac Delta Function (aka "unit impulse"): value is zero everywhere except at zero, and whose integral over the entire real line is equal to one
<center><img src="ReadingNotesSupplements/dirac_delta_func.png" alt="" style="width:400px; margin-top: 10px"/></center>	
$$\delta(x) =
\begin{cases}
0, & x \ne 0 \\
\infty, & x = 0
\end{cases} \;\;,\qquad \int_{-\infty}^\infty \delta(x) dx = 1$$
- Often, we want to represent a $\delta$ distribution where the unit mass is not at the origin; there are two equivalent notations for this (consider the unit mass is at $z$):
	- $\delta_z(x) = \delta(x-z) = \begin{cases} 0, & z \ne 0 \\ \infty, & z = 0\end{cases}$ 
- Key property:
	- $\int \delta_z(x) ~\phi(z) ~dz = \phi(x)$  for any function $\phi(x)$
	- Intuition: $\delta$ takes value $1$ when $z=x$, so only $\phi(x)$ remains (but the integral still has mass because of the $\int_{-\infty}^\infty \delta(x) dx = 1$ property of dirac delta)