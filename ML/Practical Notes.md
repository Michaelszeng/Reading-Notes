
## Normalization
Preferred representation: 6D rotations
DO NOT NORMALIZE ROTATIONS
- Whether using 6D rotations or quaternions -- they are already $\in [-1, 1]$
- Normalizing per-dimension stretches the rotation; certain axes get weighed more; not desirable
- For quaternions specifically, there is also also $q \equiv -q$ ambiguity
	- Forces model to learn that $q = -q$
	- If this is controlled in the data (i.e. only ever uses $+q_w$), this doesn't matter

## Understanding Batch Size
- Larger Batch Size $\rightarrow$ more samples used per optimization step $\rightarrow$
	- More stable gradients during optimization step
	- Slower optimization step (if GPU begins to saturate)
- Smaller batch size can be better
	- Less stable gradients $\rightarrow$ more exploration, avoid settling into narrow/brittle local minima $\rightarrow$ less overfitting + memorization
	- 64-128 in general
- Small datasets $\rightarrow$ use smaller batch size
- Unstable architectures (i.e. transformers) prefer larger batch sizes (256+)
#### Good Minima vs Bad Minima
- Wide valleys imply robustness; model won't fail (or produce high loss) if you vary the parameters slightly or give an OOD input
- Having too stable gradients $\rightarrow$ gradient descent finds the closest minima and never escapes
- Slightly unstable gradients $\rightarrow$ escape narrow minima