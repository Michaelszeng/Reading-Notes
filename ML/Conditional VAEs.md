## Conditional VAE's
**Concepts**
- Autoencoders
	<center><img src="ReadingNotesSupplements/Autoencoder.png" alt="" style="width:600px; margin-top: 10px"/></center>	
- Map inputs into compact latent space (and then back out)
- To generate new images: randomly sample points in latent space and run them through decoder
	- PROBLEM: latest space is messy; most points in latent space don't map to meaningful image
- THEREFORE: learn a latent space that is smoother
- Note: this is a form of *variational inference* because we choose a family of tractable distributions $Q = \{ q_\phi(z | x)\}$ and optimize for the best (i.e. best fitting the actual posterior $p_\theta(z|x)$) member of this family. We explicitly minimize $KL(q_\phi(z|x) ~\|~ p_\theta(z|x))$
**How it Works:**
- Notation
	- $p(x)$: data distribution
	- $p(z)$: latent distribution
	- $p(z | x)$: posterior distribution; map from input to latent
	- $p(x | z)$: likelihood distribution; map from latent to input
- KEY IDEA: sample from $p(z | x)$; i.e. the latent distribution that maps to data distribution
	- However, computing $p(z | x)$ is intractable; requires knowing $p(z, x)$ and $p(x)$
	- KEY IDEA:
		- Train an encoder $q_\phi(z | x)$ to approximate $p(z | x)$
			- Pose $q_\phi(z | x)$ as a Gaussian, where $\phi$ is the mean/variance
			- This encoder doesn't map input data directly to a latent-space point, but rather to a Gaussian centered at latent-space point $\mu$
		- Train a decoder $p_\theta(x | z)$ that predicts images from latents
			- We sample a few latent-space points from the Gaussian predicted by the encoder
			- The decoder then maps these back into images
		- Choose $p(z)$ to be a 0,1 Gaussian (you can really pick anything though); model will learn to map to/from a latent space that is close to a 0,1 Gaussian
		- During learning: we minimize: 		$$L(\theta) = -E_{q_\phi(z|x)}[\log p_\theta(x^{(k)} | z)] + KL(q_\phi(z|x^{(k)}) \| p(z))$$
			- The first term is a negative log likelihood of predicting data sample $k$
				- This ensures the combined encoder + decoder produces images close to the data
			- The second term is KL divergence between prior $p(z)$ and encoder's learned distribution
				- This ensures the latent-space is smooth and nice (a Gaussian)
			- "Reparameterization trick":
				- Because the forward process requires sampling, you can't back-prop through it
				- The "reparameterization trick" is just a way to isolate the sampling from the model, include it as a constant (?), so that back-prop works
- Because we penalize divergence between model's learned posterior $q_\phi(z|x)$ and 0,1 Gaussian, the posterior ends up looking like a 0,1 Gaussian -- smooth, filled-in --> you can smoothly interpolate/sample in the latent space
	- Note that VAE's tend to produce blurry outputs in part because of this -- all $x$ are pushed toward $\mu=0$ in latent-space, so everything averages together a little bit
- Another reason for blurry outputs: If there's multi-modaility (multiple $x$ map to same $z$), MSE loss pushes decoder to output average
- Note that we neither know $p(x)$ or $p(z)$; we just have samples from each
**Sources**
- [YouTube Video](https://www.youtube.com/watch?v=qJeaCHQ1k2w)