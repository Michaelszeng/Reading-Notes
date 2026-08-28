## Mixture of Transformers
- Method for encoding multi-modal data (i.e. video, text, actions)
	- Note: in robotics, the text encoder is usually pre-trained frozen
- Separate transformers/weights for each modality
- Attend to each other at each transformer layer -- i.e. action tokens attend to intermediate representations of video tokens at each layer
- Common trick: only use the first few layers of video encoder
	- Most of temporal reasoning happens in early layers, while later layers are used for encoding/representing fine visual detail

## Gradient Accumulation
KEY IDEA: train with larger batch sizes even when a single batch doesn't fit in VRAM
- Select a *micro-batch* size that fits in VRAM
- Only calculate gradients of each *micro-batch*
- Average gradients over all *micro-batches*, then apply gradient clipping + gradient descent step 
Equivalent to full-batch gradient descent:
- Differentiation is linear; `gradient of the sum` == `sum of gradients`
- Except when using batch-norm -- batch-norm uses batch's statistics to calculate gradient step

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