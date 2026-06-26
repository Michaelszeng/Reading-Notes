## JEPA (aka I-JEPA)
- Encoder 1 takes images with missing patches --> outputs embeddings for present patches
- Predictor takes Encoder 1 embeddings --> predicts embeddings of missing patches
- Encoder 2 takes full image --> outputs full embeddings
- Loss: L2 norm in embedding space between Predictor's prediction of missing patch embeddings and of Encoder 2's missing patch embeddings
- KEY IDEAS:
	- no language
	- "world-prediction" -- just predict missing parts of images/videos
	- non-generative; does not predict in pixel space, but rather in latent space
	- Just a visual representation learner (to be used for downstream vision tasks)
Comparison w/CLIP:
<img src="ReadingNotesSupplements/JEPA_vs_CLIP.png" alt="" style="width:100%; margin-top: 10px"/>

### V-JEPA
Same as I-JEPA but takes in video spatio-temporal patches instead of image patches

### VL-JEPA
Basically JEPA converted into a VLM
- Predictor also conditions on language prompt
- 2nd Encoder receives ground-truth response (and uses different weights)
- Loss is measured in language embedding space
![[JEPA_vs_VL_JEPA.png]]

### LeWorldModel (World-Model version of JEPA)
- 1st Encoder receives full current frame
- 2nd Encoder receives full next frame
- Predictor also conditions on action --> predicts embeddings of next frame based on embeddings of current frame
- Loss is measured in image embedding space


## [R3M]([2203.12601](https://arxiv.org/pdf/2203.12601))
- Main contribution: pre-trained encoder trained using human ego-centric data + time contrastic learning
- Notation:
	- $z = F_\phi(I)$
	- $z$ = encoded image
	- $F_\phi$ = encoder
	- $I$ = image
- Time Contrastic Learning:
	- Training objective: Distance between images closer in time is smaller
	- Sample batch $B$ of 3 frames from the same video: $[I_i, I_j, I_k]^{1:B}$ where $i < j < k$:	$$\mathcal{L}_{tcn} = - \sum_{b \in B} \log \frac{e^{S(z_i^b, z_j^b)}}{e^{S(z_i^b, z_j^b)} + e^{S(z_i^b, z_k^b)} + e^{S(z_i^b, z_i^{\neq b})}} = - \sum_{b \in B} \log(1 + e^{S(z_i^b, z_k^b)-S(z_i^b, z_j^b)} + e^{S(z_i^b, z_i^{\ne b}) - S(z_i^b, z_j^b)})$$
		- $S$ is a "similarity function" (in practice, negative L2 norm in embedding space)
		- $z_i^{\ne b}$ is a negative sample *from a different video*
			- In practice, for each 3-frame sample, they use multiple negative examples
		- $tcn$ = "time contrastive network"
		- $\frac{e^{S(z_i^b, z_j^b)}}{e^{S(z_i^b, z_j^b)} + e^{S(z_i^b, z_k^b)} + e^{S(z_i^b, z_i^{\neq b})}}$ is a softmax operator
			- Maximized ($\rightarrow 1$) when numerator $\gg$ denominator
		- This loss function encourages maximizing the differences $S(z_i^b, z_k^b)-S(z_i^b, z_j^b)$ and $S(z_i^b, z_i^{\ne b}) - S(z_i^b, z_j^b)$
			- i.e. ensure that the 2 frames close in time have much larger similarity than 2 frames far in time, or 2 frames from different videos 
- Language Semantic Alignment
	- Train a separate language predictor $g_\theta$ that takes two image embeddings $z_i$, $z_j$ produced by $F_\phi$, and a language token $l$, produces score if the transition from $z_i$ to $z_j$ completes the task described by the language
	- Encourage $F_\phi$ to learn semantically meaningful representation
	- Sample batch $B$ of 2 frames from the same video: $[I_0, I_f]^{1:B}$ where $I_0$ comes from first 20% of video and $I_f$ comes from last 20% of video, along with language label $[l]^{1:B}$:  $$\mathcal{L}_{{language}} = - \sum_{b \in B} \log \frac{e^{g_\theta(z_0^b, z_f^b, l^b)}}{e^{g_\theta(z_0^b, z_f^b, l^b)} + e^{g_\theta(z_0^b, z_0^b, l^b)} + e^{g_\theta(z_0^{\neq b}, z_f^{\neq b}, l^b)}}$$
		- Identical "contrastive-style" loss function as used in $\mathcal L_{tcn}$
		- $z_{i/j}^{\ne b}$ is a negative sample *from a different video*
		- Incentivize high score for 2 different frames from same video, 
		- Incentivize low score for 2 same frames from the video, and for 2 frames from a different video which when using misaligned wrong language label
- L1 Regularization: 		$$ \mathcal L_{regularize} = \sum_{b \in B} \|z_i^b\|_1 $$
	- For image $i$ in batch, apply L1 norm regularization; this prefers sparse embeddings (i.e. many/most components of $z$ being $0$)
	- Supposed to help with reducing dimension of state-space that is given as input to the decoder (i.e. diffusion UNet), reduce compounding errors (questionable results)
- Complete Objective function:  
	- For a given batch of images $I_{0, i, j, k, f}^{1:B}$:  $$\mathcal{L} = \mathbb{E}_{I^{1:B}_{0,i,j,k,f} \sim \mathcal{D}} \left[ \lambda_1 \mathcal{L}_{tcn} + \lambda_2 \mathcal{L}_{language} + \lambda_3 \left\lVert \mathcal{F}_\phi(I_i) \right\rVert_1 + \lambda_4 \left\lVert \mathcal{F}_\phi(I_i) \right\rVert_2 \right]$$
