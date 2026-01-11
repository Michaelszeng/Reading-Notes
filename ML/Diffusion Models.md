
# Principles of Diffusion Models
[[The Principles of Diffusion Models](https://www.arxiv.org/pdf/2510.21890)]([The Principles of Diffusion Models](https://www.arxiv.org/pdf/2510.21890))
Diffusion as a VAE
- Primer on VAEs:
	- traditional autoencoders transform inputs to unstructured latent space
	- sampling arbitrarily from $\mathbb{R}^{d_{latent}}$ usually picks a meaningless area 
- instead of jointly training encoder and decoder, encoder fixed as forward noising process
- Like VAEs, DDPMs optimize a variational lower bound of the log likelihood

# Unifying Perspectives on Diffusion

**Forward Process**
- Probability path perspective: $p_t(\cdot | z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$ where
	- $p_0(x) = \int p_0(x|z) ~p_{data}(z) ~dz = \int p_{init}(x) ~p_{data}(z) ~dz = p_{init}(x)$
	- $p_1(x) = \int p_1(x|z) ~p_{data}(z) ~dz = \int \delta_z(x) ~p_{data}(z) ~dz = p_{data}(x)$
- DDPM forward process perspective: $q(x_t | x_{t-1}) := \mathcal{N}(x_{t}; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$
	- $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha_t})I)$ for $\alpha_t = 1-\beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
	- Equivalence to probability path:
		- $\alpha_t = \sqrt{\bar{\alpha}_t}$, $\beta_t = 1-\bar{\alpha}_t$
**Sampling/Denoising Process**
- Reverse SDE perspective: $X_{t_{k+1}} = X_{t_k} + \left( u^\theta_k + \frac{\sigma_k^2}{2} s^\theta_k \right) \Delta t_k + \sigma_k \sqrt{\Delta t_k} \, \varepsilon$ where $\varepsilon \sim \mathcal{N}(0,I)$
- DDPM perspective: $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t -  \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \, \epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \varepsilon$ where $\varepsilon \sim \mathcal{N}(0,I)$
- DDPM is a particular time-discretization of the variance-preserving reverse SDE


# DDPM (Denoising Diffusion Probabilistic Models)
### Background
- Data $x_0 \sim q(x_0)$
- Latent sequence $x_1, \dots, x_T$
- Latent model $p_\theta(x_0) := \int p_\theta(x_{0:T}) ~dx_{1:T}$
- Reverse Process:
	- $p_\theta(x_{t-1} | x_t) := \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t I)$
	- $p(x_T) = \mathcal{N}(x_T; 0, I)$
	- $p_\theta(x_{0:T}) := p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$ (Gaussian Markov Chain distribution)
- Forward Process/Diffusion process: 
	- $q(x_{1:T} | x_0) := \prod_{t=1}^T q(x_t | x_{t-1})$ (Markov Chain distribution)
	- $q(x_t | x_{t-1}) := \mathcal{N}(x_{t}; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$  (Gaussian w/ mean $\sqrt{1-\beta_t} x_{t-1}$ and variance $\beta_t I$ evaluated at $x_t$}
	- $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha_t})I)$ for $\alpha_t = 1-\beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
		- very simple closed form, thus, sampling $x_t$ given $x_0$ is easy
	- In their implementation, $\beta_t$ are hyper-parameters/constants
- Training minimizes the VLB of negative log likelihood $\log p_\theta(x_0)$, which simplifies eventually to $\arg\min_\theta \left\| \epsilon_\theta(x_k, \text{obs}, k) - \epsilon \right\|^2$ (see notes on Diffusion Policy's loss function)
	- DDPM takes a fairly different perspective from the Reverse-time SDE proposed in https://diffusion.csail.mit.edu
		- They do not view their approach as trying to learn a vector field and/or score function to match a target vector field/score function that encodes the desired probability path
		- They view their approach as:
			- MLE on the data distribution in order to learn parameters
			- Reversing the Gaussian chain during inference
<center><img src="ReadingNotesSupplements/DDPM.png" alt="" style="width:750px; margin-top: 10px"/></center>	

# Flow-Matching Diffusion
[https://diffusion.csail.mit.edu/]
### Overview - What are we doing?
1. Define $p_{init}(x)$ (usually 0-1 Gaussian). We sample "noisy images" from $p_{init}$.
2. Define $p_{data}(x)$ -- the data distribution of "clean images". We want to "denoise" $x_0 \sim p_{init}$ to $x_1 \sim p_{data}$
3. Goal: learn a vector field $u_t$ (which maps $x_t$ to a direction in $x_t$ space) to guide samples from $p_{init}$ to $p_{data}$.
	1. This guidance process can be described by an ODE $\frac{d}{dt} x_t = u_t(x_t)$ and "solved" (for $x_1 \sim p_{data}$) using integration methods (i.e. Euler, Euler-Maruyama).
4. First, we show how to express the target vector field $u^{target}_t$ mathematically (using conditional/marginal probability paths), though we find $u_t^{target}$ intractable to evaluate. Thus, we must learn $u_t^\theta$ to approximate $u_t^{target}$.
5. Second, we define loss function on our learned $u^\theta_t$: $\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, x \sim p_t} \left[ \left\| u_t^{\theta}(x) - u_t^{\text{target}}(x) \right\|^2 \right]$
	1. We use the fact that the minimizer of $\mathcal{L}_{\text{FM}}(\theta)$ is the same as for $\mathcal{L}_{\text{CFM}}(\theta)$ (using conditional vector fields instead) to make this loss function tractable to compute
6. Finally, we parameterize $u_t^\theta$ as a neural net and train using $\mathcal{L}_{\text{FM}}(\theta)$ in typical supervised-learning fashion. 

At least, this covers the ODE/Flow-Matching case. The SDE/Diffusion case follows similarly, but the guidance process is an SDE $\mathrm{d}X_t = \left[u_t^{\text{target}}(X_t)+ \frac{\sigma_t^2}{2} \nabla \log p_t(X_t)\right] \mathrm{d}t+ \sigma_t \mathrm{d}W_t$, and we learn the score function $\log p_t(X_t)$ instead of the vector field (though this turns out to be equivalent to learning the vector field).

### ODE Fundamentals
- Fundamentals
	- $X_i$ is an "image"
	- trajectory $X$: $t \rightarrow X_t$ where $t \in [0,1], X_t \in \mathbb{R}^d$
	- vector field $u$: $(x,t) \rightarrow v \in \mathbb{R}^d$
		- i.e. for every time step, every point maps to some vector in $\mathbb{R}^d$
- ODE: 
	- $\frac{d}{dt} X_t = u_t(X_t)$         (trajectory follows time-varying vector field)
	- $X_0 = x_0$                   (initial condition; trajectory starts at initial noisy image)
- Flow ($\psi$):
	- $\psi_t(X_0) = X_t$                     is the solution to the ODE
	- $(x_0, t) \rightarrow \psi_t(x_0)$                i.e. given initial condition, it tells you what $X_t$ for any given timestep $t$
	- $\frac{d}{dt} \psi_t(x_0) = u_t(\psi_t(x_0))$      i.e. flow ODE (i.e. the derivative of the flow at time $t$ is equal to the vector field at the $X_t$)
	- $\psi_0(x_0) = x_0$                      (initial flow is equal to initial image)
	- Broadly, flow is a collection of solutions to the ODE (for any given initial condition $x_0$)
- Big picture: 
	- vector field defines the ODE
	- A trajectory is a single solution to the ODE given initial condition $x_0$
	- Flow is a function that defines all solutions to the ODE for all initial conditions $x_0$
- Property: If $u_t(x)$ is Lipschitz, then unique solution to flow/trajectory exists
- Approximately Solving ODEs (i.e. computing $X_t$ for some $X_0$) 
	- In practice, solving for $\psi_t$ closed-form is impossible; Euler Integration is used
	<center><img src="ReadingNotesSupplements/solving_ode.png" alt="" style="width:450px; margin-top: 10px"/></center>	

**Flow Models**
- Consider $p_{init}$ -- distribution of noisy images given as input to the model, $p_{data}$ -- data distribution
- Goal: convert $p_{init} \rightarrow p_{data}$
	- We do this by solving the ODE:
		- Sample $X_0 \sim p_{init}$
		- Simulate $X_0$'s trajectory using the ODE $\frac{d}{dt} X_t = u^\theta_t(X_t)$ (and Euler integration) where $u^\theta_t$ is represented as Neural Network
		- We want $\psi_1^\theta(X_0) = X_1 \sim p_{data}$
- NOTE: $\theta$ parameterizes the vector field, which we then simulate to get the flow
	- There exist models where $\theta$ directly parameterizes the flow, though these don't work well

### SDE Fundamentals
- Fundamentals
	- Random variable $X_t, \; t \in [0,1]$
	- i.e. the trajectory is a set of random variables
	- Diffusion coefficient: $\sigma_t: t \rightarrow \mathbb{R}_{\ge0}$
		- describes how much noise injected
- SDEs:
	- $X_0 = x_0$
	- $d X_t = u_t(X_t) ~ dt + \sigma_t ~ d W_t$
		- $\sigma_t ~d W_t$ injects noise into the equation
		- Derivation:
			- Traditional ODE: $\frac{d}{dt} X_t = u_t(X_t)$
			- Using defn. of derivative: $\lim_{h \rightarrow 0} \frac{X_{t+h} - X_t}{h} = u_t(X_t) \quad \implies \quad \frac{X_{t+h} - X_t}{h} = u_t(X_t) + R_t(h)$  where $R_t(h)$ is "residual" term from not having $\lim_{h \rightarrow 0}$; we consider this small and ignore it
														$\implies \quad X_{t+h} = X_t + h ~ u_t(X_t) + h ~ R_t(h)$
														$\implies \quad X_{t+h} \approx X_t + h ~ u_t(X_t)$
														$\implies \quad X_{t+h} \approx X_t + h ~ u_t(X_t) + \sigma_t(W_{t+h} - W_t)$
														$\implies \quad dX_{t} \approx u_t(X_t) ~ dt + \sigma_t ~ dW_t$
- $W_t$ is generated by Brownian Motion (aka "Wiener process"):
	- Random Walks
				<center><img src="ReadingNotesSupplements/brownian_motion.png" alt="" style="width:400px; margin-top: 10px"/></center>	
		- Simulating such a Brownian motion: $W_{t+h} = W_t + \sqrt{h}\,\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d) \quad (t = 0, h, 2h, \ldots, 1 - h)$
			- This achieves $W_t - W_s \sim N(0, (t-s)I_d)$ because: $Var(W_t) = Var(\sum_{k=1}^n \sqrt{h} ~\epsilon_k) = \sum_{k=1}^n Var(\sqrt{h} ~\epsilon_k)= \sum_{k=1}^n h ~Var(\epsilon_k) = \sum_{k=1}^n h \cdot 1 = nh = t$ (using the fact that $\epsilon_t$ are zero-mean and independent)
	- Conditions:
		- $W_0 = 0$
		- $W_t - W_s \sim N(0, (t-s)I_d)$ where $0 \le s < t$ 
			- variance of $W_t$ increases linearly in time
		- $W_{t_1} - W_{t_0} \perp \dots \perp W_{t_n} - W_{t_{n-1}}$ i.e. all increments are independent
	- Notes:
		- NOT differentiable
		- is a type of Gaussian process (Gaussian noise injected every step)
	- NOTE: Brownian motions are quite universal in SDEs! As universal as Gaussians are for distributions, Brownian motion is the universal stochastic process.
		- All Gaussian processes satisfy that $W_t - W_s \sim N(\mu, \sigma^2)$ for some $\mu, \sigma$, though most others do not depend only on $t-s$ for example, or do not satisfy independent increments, or may depend on $X_s$
- Approximately Solving SDEs (i.e. computing $X_t$ for some $X_0$)
	- If $u_t(x)$ is Lipschitz, then unique solution to SDE exists
				<center><img src="ReadingNotesSupplements/solving_sde.png" alt="" style="width:450px; margin-top: 10px"/></center>	
**Diffusion Models**
- Identical to a flow model, but with added noise every Euler step, scaled by a hand-picked $\sigma_t$ schedule
- diffusion model w/ $\sigma_t = 0$ is a flow model

### Training Objective
- Need to learn vector field $u_t^\theta$ by minimizing $\mathcal{L}(\theta) = \| u_t^\theta(x) - u_t^{\text{target}}(x) \|^2$ for some $u_t^{\text{target}}$ that will take samples from $p_{init}$ and move then to samples from $p_{data}$
- The challenge is expressing, mathematically, what $u_t^{\text{target}}$ should be, since this is not obvious just given a dataset
**Probability Paths**
- A conditional probability path is a set of distributions $p_t(x|z)$ s.t.:
	- $p_0(\cdot | z) = p_{init}(\cdot)$
	- $p_1(\cdot | z) = \delta_z(\cdot)$
	- Intuition: Conditional probability path interpolates from $p_{init}$ (usually 0-1 Gaussian) to the dirac delta at $z$
- Marginal probability path $p_t(x)$ induced by conditional:
	- $p_t(x) = \int p_t(x | z) ~ p_{data}(z) ~ dz$
	- Intuition: Marginal probability path interpolates from $p_{init}$ to $p_{data}$:
		- $p_0(x) = \int p_0(x|z) ~p_{data}(z) ~dz = \int p_{init}(x) ~p_{data}(z) ~dz = p_{init}(x)$
		- $p_1(x) = \int p_1(x|z) ~p_{data}(z) ~dz = \int \delta_z(x) ~p_{data}(z) ~dz = p_{data}(x)$   (recall the property of dirac delta distributions: $\int \delta_z(x) ~\phi(z) ~dz = \phi(x)$ for any $\phi(x)$)
	- Note that, in practice, the marginal probability densities are intractable to compute. Rather, what we care about is being able to sample from $p_t(x)$, which only requires the conditional probability path:
			1. sample $z \sim p_{data}$
			2. sample $x \sim p_t(\cdot | z)$
		- We still discuss the marginal probability path for theoretical completeness.
- Example: Gaussian path: $p_t(\cdot | z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$ (which is used in Diffusion models)
	- Let $\alpha_t$, $\beta_t$ be noise schedules with $\alpha_0 = \beta_1 = 0$, $\alpha_1 = \beta_0 = 1$
	- Conditional path:
		- $p_0(\cdot | z) = \mathcal{N}(\alpha_0 z, \beta_0^2 I_d) = \mathcal{N}(0, I_d) = p_{init}$
		- $p_1(\cdot | z) = \mathcal{N}(\alpha_1 z, \beta_1^2 I_d) = \mathcal{N}(z, 0) = \delta_z(\cdot)$
	- Marginal path:
		- Since $p_t(x | z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$, sampling from the marginal is equivalent to:
			- sample $z \sim p_{data}$
			- compute $x = \alpha_t z + \beta_t \epsilon$ where $\epsilon \sim \mathcal{N}(0,I_d)$
					<center><img src="ReadingNotesSupplements/gaussian_path_example.png" alt="" style="width:850px; margin-top: 10px"/></center>	
		- In this example, we can see as $t$ goes from $0 \rightarrow 1$:
			- sampling from the conditional on $z$ goes from 0-1 noise to a dirac delta at $z$
			- sampling from the marginal goes from 0-1 noise to being the exact data distribution
			- (lighter color = higher density)

**Conditional & Marginal Vector Fields**
- Conditional Vector Field (for Gaussian case):
	- We define the conditional vector field $u_t^{target}$ s.t., given a chosen datapoint $z$: if $X_0 \sim p_{init}(x)$, then $\frac{d}{dt} X_t = u_t^{target}(X_t | z) \implies X_t \sim p_t(\cdot | z)$
		- Intuition: integrating along conditional $u_t^{target}$ takes $X_t$ along the conditional probability path
	- For Gaussian conditional probability path: $u_t^{target}(x|z) = (\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t) z + \frac{\dot{\beta}_t}{\beta_t} x$    (where $\dot{*}$ = time derivative of $*$)
		- Intuition: weighted average btwn $x$ and $z$
		- Derivation:
			- Recall Gaussian conditional path: $x_t = \alpha_t z + \beta_t \epsilon \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot | z)$ 
			- Define a conditional flow $\psi_t^{target}(x | z) = \alpha_t z + \beta_t x$. To see that this is a valid flow, we want to show $\psi_t(X_0) = X_t$ (the defn. of a flow)
				- $\psi_t^{target}(X_0 | z) = \alpha_t z + \beta_t X_0 = X_t \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot | z)$
			- Then, we derive the conditional vector field $u_t^{target}$ by differentiating the conditional flow $\psi_t^{target}$: $$\begin{align*}\frac{d}{dt} \psi_t^{target}(x | z) &= u_t^{target} ( \psi_t^{target}(x | z) | z) \qquad && \text{(using the Flow ODE } \frac{d}{dt} \psi_t(x_0) = u_t(\psi_t(x_0)) ~) \\
				\dot{\alpha}_t z + \dot{\beta}_t x &= u_t^{target}(\alpha_t z + \beta_t x | z) && \text{(plugging in our defn. of } \psi_t^{target}(x|z) )\\
				\dot{\alpha}_t z + \dot{\beta}_t (\frac{x' - \alpha_t z}{\beta_t}) &= u_t^{target}(x' | z) && \text{(defining } x' \text{ s.t. } x = \frac{x' - \alpha_t z}{\beta_t} ) \\
				u_t^{target}(x'|z) &= (\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t) z + \frac{\dot{\beta}_t}{\beta_t} x'
				\end{align*}$$
- Marginal Vector Field:
	- $u_t^{target}(x) = \int u_t^{target}(x|z) ~\frac{p_t(x|z) ~p_{data}(z)}{p_t(x)} ~dz$
		- Using Bayes rule, $\frac{p_t(x|z) ~p_{data}(z)}{p_t(x)} = p_t(z | x)$  i.e. probability that datapoint $z$ induced $x$
			- Intuition: to find the vector field at $x$, we take a weighted average of all conditional vector fields $x$ (for all $z$), with the weights being the likelihood of $x$ being induced by $z$
		- Proof: see ["continuity equation"]([An Introduction to Flow Matching and Diffusion Models](https://arxiv.org/pdf/2506.02070))
	- Similarly, integrating along marginal $u_t^{target}$ takes  along the marginal probability path
		- $X_0 \sim p_0(x) = p_{init}(x)$ and $\frac{d}{dt} X_t = u_t^{target} (X_t) \quad \implies \quad X_t \sim p_t(\cdot) \quad \quad \text{for } t \in [0,1]$
- KEY TAKEAWAY: we have now mathematically defined the vector field we want to learn $u_t^{target}(x)$ 
	- In theory, we could use $u_t^{target}(x)$ directly to denoise images; there is no need for learning.
	- HOWEVER: $u_t^{target}(x)$ is intractable to compute due to the integral over $z$.
		- Thus, we use learning to approximate $u_t^{target}(x)$.
- Example visual:
						<center><img src="ReadingNotesSupplements/conditional_marginal_vector_fields.png" alt="" style="width:750px; margin-top: 10px"/></center>	
	<center><sub><sup>The GT distribution = 5 blue discs; conditional (on z) vector fields point back to z; marginal vector fields point back to each disc</sub></sup></center>

**Score Function**
- Analog of vector field but for SDEs (instead of ODEs)
- Conditional Score Function
	- $\nabla_x \log p_t(x | z)$
- Marginal Score Function$$\begin{align*}\nabla_x \log p_t(x) &= \frac{\nabla_x p_t(x)}{p_t(x)}\\ 
											&= \frac{\nabla_x \int p_t(x | z) ~ p_{data}(z) ~ dz}{p_t(x)}  \qquad \text{(using defn. of marginal probability path above)} \\
											&= \int \frac{\nabla_x p_t(x | z) ~ p_{data}(z) ~ dz}{p_t(x)} \qquad \text{(}p_t(x) \text{ is a constant w.r.t. integration variable } z \text{ so can be moved into the integral)}\\
											&= \int \frac{p_t(x | z) ~\nabla_x \log p_t(x | z) ~ p_{data}(z) ~ dz}{p_t(x)} \qquad \text{(using the identity } \nabla_x p_t(x | z) = p_t(x | z) ~\nabla_x \log p_t(x | z) -\text{this is just a re-arranged form of the chain rule when differentiating a log)} \\
											&= \int \nabla_x \log p_t(x | z) \frac{p_t(x | z) ~ p_{data}(z)}{p_t(x)}~dz \\
											\end{align*}$$
	- Note: we again see the conditional posterior appear through Bayes: $\frac{p_t(x|z) ~p_{data}(z)}{p_t(x)} = p_t(z | x)$
	- Interpretation of this score function: weighted average of individual conditional (on $z$) scores, weighted by likelihood of $x$ being induced by $z$
- For Gaussian probability path, Gaussian Score: $$
\begin{align*}
p_t(x \mid z)
&= \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d)
= \frac{1}{(2\pi)^{d/2} |\beta_t^2 I_d|^{1/2}}
\exp\!\left(-\tfrac{1}{2}(x - \alpha_t z)^\top (\beta_t^2 I_d)^{-1} (x - \alpha_t z)\right) \\[6pt]
\log p_t(x \mid z)
&= -\frac{d}{2}\log(2\pi)
-\frac{1}{2}\log|\beta_t^2 I_d|
-\frac{1}{2}(x - \alpha_t z)^\top (\beta_t^2 I_d)^{-1} (x - \alpha_t z) \\[6pt]
\nabla_x \log p_t(x \mid z)
&= -\frac{1}{2} \nabla_x \!\left[(x - \alpha_t z)^\top (\beta_t^2 I_d)^{-1} (x - \alpha_t z)\right] \\[6pt]
&= -(\beta_t^2 I_d)^{-1} (x - \alpha_t z) \\[6pt]
&= -\frac{x - \alpha_t z}{\beta_t^2}.
\end{align*}
$$
- Why the Score Function is useful:
	- Theorem: $X_0 \sim p_0(x)=p_{init}(x), \quad \mathrm{d}X_t = \left[u_t^{\text{target}}(X_t)+ \frac{\sigma_t^2}{2} \nabla \log p_t(X_t)\right] \mathrm{d}t+ \sigma_t \mathrm{d}W_t \quad \implies \quad X_t \sim p_t(x) = p_{data}(x) \quad \quad \text{for } t \in [0,1]$ 
		- Proof: ["Fokker-Planck equation"]([An Introduction to Flow Matching and Diffusion Models](https://arxiv.org/pdf/2506.02070))
		- Contrast this with the equivalent theorem using ODEs (with marginal vector fields) instead of SDEs:  $X_0 \sim p_0(x) = p_{init}(x)$ and $\frac{d}{dt} X_t = u_t^{target} (X_t) \quad \implies \quad X_t \sim p_t(\cdot) \quad \quad \text{for } t \in [0,1]$
		- The SDE version includes the $\sigma_t \mathrm{d}W_t$ noise term, but also includes a "correction" term that is $\frac{\sigma_t^2}{2}$ times the score function
		- What's surprising is that, no matter how large $\sigma_t$ is, no matter how much noise you inject, the theorem holds; i.e. $X_1 \sim p_{data}(x)$
- Why use SDE's instead of ODEs?
	- The ODE formulation is called ***flow matching*** -- gives you a deterministic probability path from $X_0 \sim p_{init}(x)$ to $X_1 \sim p_{data}(x)$. 
		- This works -- i.e. Stable Diffusion 3 uses this
	- The SDE formulation is called ***diffusion*** -- gives you a whole family of probability paths (with varying $\sigma_t$) that do the same thing, but $\sigma_t$ gives an empirically tunable knob that people have found to work better sometimes

#### Flow Matching Loss
- The idea loss function would be:$$\begin{align}
\mathcal{L}_{\text{FM}}(\theta)
&= \mathbb{E}_{t \sim \text{Unif},\, x \sim p_t} \left[ \left\| u_t^{\theta}(x) - u_t^{\text{target}}(x) \right\|^2 \right] \\
&= \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)} 
\left[ \left\| u_t^{\theta}(x) - u_t^{\text{target}}(x) \right\|^2 \right] \qquad \text{(Recall that sampling $x \sim p_t(x)$ equivalent to sampling $z \sim p_{data}$, $x \sim p_t(\cdot | z)$)}
\end{align}$$
	- During training, we evaluate this loss on a batch of $x$
- However, computing $u_t^{target} = \int u_t^{target}(x|z) ~\frac{p_t(x|z) ~p_{data}(z)}{p_t(x)} ~dz$ is not tractable due to the integral
- Instead, use Conditional Flow Matching Loss: $$\mathcal{L}_{\text{CFM}}(\theta)
= \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)}
\left[ \left\| u_t^{\theta}(x) - u_t^{\text{target}}(x \mid z) \right\|^2 \right]$$
	- This becomes tractable because $u_t^{target}(x|z) = (\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t) z + \frac{\dot{\beta}_t}{\beta_t} x$ has closed form
- Theorem: $\mathcal{L}_{\text{FM}}(\theta) = \mathcal{L}_{\text{CFM}}(\theta) + C$  where $C$ is constant w.r.t. $\theta$
	- Thus, $\mathcal{L}_{\text{FM}}(\theta)$ and $\mathcal{L}_{\text{CFM}}(\theta)$ have same minimizer
	- In fact, the training process will be identical (for any gradient-based optimizer, since $\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$)
	- Proof:
		- $\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif},\,x \sim p_t(\cdot)} \left[ \left\| u_t^{\theta}(x) - u_t^{\text{target}}(x) \right\|^2 \right] = \mathbb{E}_{t \sim \text{Unif},\, x \sim p_t(\cdot)} \left[ \left\| u_t^{\theta}(x) \right\|^2 + \left\| u_t^{\text{target}}(x) \right\|^2 - 2 u_t^{\theta}(x) u_t^{\text{target}}(x) \right]$
		- $\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)} \left[ \left\| u_t^{\theta}(x) - u_t^{\text{target}}(x \mid z) \right\|^2 \right] = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)} \left[ \left\| u_t^{\theta}(x) \right\|^2 - \left\| u_t^{\text{target}}(x \mid z) \right\|^2 - 2 u_t^{\theta}(x) u_t^{\text{target}}(x \mid z) \right]$
			- Notice: $\left\| u_t^{\theta}(x) \right\|^2$ appears identically in both, $\left\| u_t^{\text{target}}(x) \right\|^2$ and $\left\| u_t^{\text{target}}(x \mid z) \right\|^2$ are constant w.r.t. $\theta$, thus, the only difference between $\mathcal{L}_{\text{FM}}(\theta)$ and $\mathcal{L}_{\text{CFM}}(\theta)$ are the $2 u_t^{\theta}(x) u_t^{\text{target}}(x)$ vs $2 u_t^{\theta}(x) u_t^{\text{target}}(x \mid z)$ term.
			- Now, we show $\mathbb{E}_{t \sim \text{Unif},\, x \sim p_t(\cdot)} \left[ 2 u_t^{\theta}(x) u_t^{\text{target}}(x) \right] = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)} \left[ 2 u_t^{\theta}(x) u_t^{\text{target}}(x \mid z)\right]$: $$\begin{aligned}
\mathbb{E}_{t \sim \mathrm{Unif},\, x \sim p_t}
\big[ u_t^{\theta}(x)^\top u_t^{\mathrm{target}}(x) \big]
&=
\int_0^1 \int p_t(x)\, u_t^{\theta}(x)^\top u_t^{\mathrm{target}}(x)\, dx\, dt
&& \text{(Expand the expectation as an integral over $x$ and $t$)} \\[1em]
%
&=
\int_0^1 \int p_t(x)\, u_t^{\theta}(x)^\top
\left[ \int u_t^{\mathrm{target}}(x \mid z)\,
\frac{p_t(x \mid z)\, p_{\mathrm{data}}(z)}{p_t(x)}\, dz \right] dx\, dt
&& \text{(Use $u_t^{\mathrm{target}}(x) = \mathbb{E}_{z \sim p(z \mid x)}[u_t^{\mathrm{target}}(x \mid z)]$)} \\[1em]
%
&=
\int_0^1 \int \int u_t^{\theta}(x)^\top u_t^{\mathrm{target}}(x \mid z)\,
p_t(x \mid z)\, p_{\mathrm{data}}(z)\, dz\, dx\, dt
&& \text{(Move $p_t(x)$ inside and cancel $p_t(x)$ terms)} \\[1em]
%
&=
\mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\mathrm{data}},\, x \sim p_t(\cdot \mid z)}
\big[ u_t^{\theta}(x)^\top u_t^{\mathrm{target}}(x \mid z) \big]
&& \text{(Recognize as expectation over $t,z,x$ sampling hierarchy)}.
\end{aligned}$$
	- Full training algorithm:
						<center><img src="ReadingNotesSupplements/flow_matching_training.png" alt="" style="width:500px; margin-top: 10px"/></center>	
		- In actual training, rather than compute the expected value in full, we sample $x$ and $z$ (potentially batches of them) to get an MC estimate of $\mathcal{L}_{CFM}(\theta)$
	- For Gaussian probability path:
		- Recall: sampling from Gaussian path is equivalent to:
			- sample $z \sim p_{data}$
			- compute $x = \alpha_t z + \beta_t \epsilon$ where $\epsilon \sim \mathcal{N}(0,I_d)$
		- Recall: $u_t^{target}(x|z) = (\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t) z + \frac{\dot{\beta}_t}{\beta_t} x$
			- Then: $$\begin{align*} \mathcal{L}_{\text{CFM}}(\theta) &= \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)} \left[ \left\| u_t^{\theta}(x) - u_t^{\text{target}}(x \mid z) \right\|^2 \right] \\
					&= \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)} \left[ \left\| u_t^{\theta}(x) - (\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t) z - \frac{\dot{\beta}_t}{\beta_t} x \right\|^2 \right]  \\
					&= \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0,I_d)} \left[ \left\| u_t^{\theta}(x) - (\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t) z - \frac{\dot{\beta}_t}{\beta_t} (\alpha_t z + \beta_t \epsilon) \right\|^2 \right] \\
					&= \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0,I_d)} \left[ \left\| u_t^{\theta}(x) - (\dot{\alpha}_t z +\dot{\beta}_t \epsilon) \right\|^2 \right] \quad \text{where } x=tz + (1-t)\epsilon
					\end{align*}$$
		- In the case where we take the simplest $\alpha_t$ and $\beta_t$ schedule: $\alpha_t = t$, $\beta_t = 1-t$ ($\implies \dot{\alpha_t} = 1, \dot{\beta_t} = -1$): $$\begin{align*} \mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0, I_d)} \left[ \left\| u_t^{\theta}\big(x\big) - (z - \epsilon) \right\|^2 \right]  \quad \text{where } x=tz + (1-t)\epsilon
			\end{align*}$$
			- This is simply MSE loss between NN's prediction and the path from $\epsilon$ to $z$
						<center><img src="ReadingNotesSupplements/flow_matching_loss_gaussian.png" alt="" style="width:400px; margin-top: 10px"/></center>	

#### Score Matching Loss
- Recall the SDE sampling process: $X_0 \sim p_0(x)=p_{init}(x), \quad \mathrm{d}X_t = \left[u_t^{\text{target}}(X_t)+ \frac{\sigma_t^2}{2} \nabla \log p_t(X_t)\right] \mathrm{d}t+ \sigma_t \mathrm{d}W_t \quad \implies \quad X_t \sim p_t(x) = p_{data}(x) \quad \quad \text{for } t \in [0,1]$ 
- As in Flow Matching, we must learn a separate network $u_t^{\theta}(X_t)$ to approximate $u_t^{\text{target}}(X_t)$
- We must also learn the score function $\nabla \log p_t(X_t)$ using:$$\mathcal{L}_{\mathrm{SM}}(\theta) = 
\mathbb{E}_{t \sim \mathrm{Unif},\, x \sim p_t(\cdot)}
\left[ \left\| s_t^{\theta}(x) - \nabla \log p_t(x) \right\|^2 \right]$$
	- Again, we replace the actual (marginal) score function $\nabla_x \log p_t(x) = \int \nabla_x \log p_t(x | z) \frac{p_t(x | z) ~ p_{data}(z)}{p_t(x)}~dz$ w/ the conditional score function $\nabla_x \log p_t(x | z)$, since the integral in the marginal is computationally intractable (but the conditional usually has a closed form, i.e. for a Gaussian probability path, $p_t(x | z)$ is a Gaussian which we can easily take the log and derivative of): $$\mathcal{L}_{\mathrm{CSM}}(\theta) = 
\mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)}
\left[ \left\| s_t^{\theta}(x) - \nabla \log p_t(x \mid z) \right\|^2 \right]$$
	- Following the identical proof (but replacing $u_t^{target}$ with $\nabla_x \log p_t$) as we used above to show the minimizer and gradients of $\mathcal{L}_{\mathrm{FM}}(\theta)$ and $\mathcal{L}_{\mathrm{CFM}}(\theta)$ are the same, we show that the minimizers and gradients for $\mathcal{L}_{\mathrm{SM}}(\theta)$ and $\mathcal{L}_{\mathrm{CSM}}(\theta)$ are the same
- The training procedure is identical as for flow matching except we use the score-matching loss function.
- For Gaussian Probability Path:
	- Recall: $\nabla_x \log p_t(x \mid z) = -\frac{x - \alpha_t z}{\beta_t^2}$. Then: $$\begin{align*}
\mathcal{L}_{\text{CSM}}(\theta)
&= \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_t(\cdot \mid z)}
  \left[ \left\| s_t^{\theta}(x) + \frac{x - \alpha_t z}{\beta_t^2} \right\|^2 \right] \\
&= \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0, I_d)}
  \left[ \left\| s_t^{\theta}(x) + \frac{\epsilon}{\beta_t} \right\|^2 \right] \qquad \text{(replacing } x \text{ with } x=\alpha_t z + \beta_t \epsilon \text{)}\\
  &= \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0, I_d)}
  \left[ \frac{1}{\beta_t^2} \left\| \beta_t s_t^{\theta}(x) + \epsilon \right\|^2 \right]
\end{align*}$$
		- For small $\beta_t$ (i.e. when $t$ is close to 1), dividing by $\beta_t$ is unstable
			- the [DDPM paper]([[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)) suggested dropping $\frac{1}{\beta_t^2}$ and reparameterizing $s_t^\theta$ into a "noise predictor" network $\epsilon_t^\theta$: $$-\beta_t s_t^\theta(x) =: \epsilon_t^\theta(x) \quad \implies \quad \mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t \sim \text{Unif},\, z \sim p_{\text{data}},\, \epsilon \sim \mathcal{N}(0, I_d)} \left[ \left\| \epsilon_t^{\theta} (x) - \epsilon \right\|^2 \right]$$
				- $\epsilon_t^\theta$ predicts precisely the noise added to $z$ to get $x$
				- Note: this does have an effect on the resulting vector field, though empirically still works:
											<center><img src="ReadingNotesSupplements/ddpm_vs_score_matching.png" alt="" style="width:1000px; margin-top: 10px"/></center>	
				<center><sub><sup>Top row shows vector field learned using score matching; Bottom row shows vector field learned using DDPM's reparameterization</sub></sup></center>
		- This is extremely simple: the neural network is directly predicting the noise injected to generate $x$ from the data sample $z$ (Recall that $x = \alpha_t z + \beta_t \epsilon$)
	- Note: for Gaussian Probability Path, we also don't actually need to learn $u_t^{\theta}(X_t)$; $u_t^{target}$ can be derived from score $\nabla_x \log p_t(x | z)$ (they are equivalent): $$\begin{align*}
u_t^{\text{target}}(x|z) &= 
\left( 
\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t}
- \dot{\beta}_t \beta_t
\right) 
\nabla \log p_t(x|z)
+ \frac{\dot{\alpha}_t}{\alpha_t} x \\
u_t^{\text{target}}(x) &= 
\left( 
\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t}
- \dot{\beta}_t \beta_t
\right) 
\nabla \log p_t(x)
+ \frac{\dot{\alpha}_t}{\alpha_t} x \quad \end{align*}$$
		- The ODE $\frac{dX_t}{dt} = u_t^{target}(x)$ where $u_t^{target}(x)$ is defined like so is called the "Probability Flow ODE"
		- Thus, training a diffusion model need only learn the score function (not the vector field).
- Comparison to DDPM & DDIM
	- Both DDPM & DDIM apply to the score-matching training process and rely on $\sigma_t$
	- DDPM is analogous to the SDE sampling process but for specific case of Gaussian process
	- DDIM is analogous to the probability-flow ODE sampling process applied to diffusion models

#### Note: Generalization vs Memorization
- There is a distinction between $p_{data}$ and $p_{empirical}$
	- $p_{empirical}$ is a normalized sum of dirac delta functions at each datapoint in the dataset
	- $p_{data}$ is smooth, containing the underlying distribution of $p_{empirical}$
- Generally (unless our data corpus is that large), we do NOT want to simply learn to denoise $p_{init}$ to $p_{empirical}$ -- this is memorization; what we want is generalization
	- If we just wanted to denoise to $p_{empirical}$, we *could* compute $u_t^{target}$(this will still require summing over every $z$ in the dataset, but it's possible)
	- But the model will only learn to reproduce samples in the dataset
- By learning $u_t^\theta$, we hope to learn to denoise to a smoother data distribution $p_{data}$.

# Denoising Score Matching (Variance-Exploding) Diffusion - A Practitioners Guide to Diffusion
[https://chenyang.co/diffusion.html]

Background/Terminology:
- noise vector $\epsilon \in \mathbb{R}^n$
- noise level $\sigma>0$
- clean image $x_0$
- noisy image $x_\sigma = x_0 + \sigma \epsilon$
- data manifold $\mathcal{K}$ (space of clean images/unaltered dataset)

Train a supervised learning model $\epsilon_\theta$ (whose inputs are $x_\sigma$, the noisy image, and $\sigma$, and who tries to predict $\epsilon$; we call the parameters of the network $\theta$) to minimize squared norm loss between the predicted $\epsilon$ and the actual $\epsilon$:
$$\min_\epsilon L(\theta) := \mathbb{E} || \epsilon_\theta(x_\sigma, \sigma) - \epsilon ||^2$$
<center><sub><sup>Loss function for one training example</sub></sup></center>
Where:
- $x_\sigma$ is a noisy image from which we want to extract the noise vector (and therefore, also the clean image)
- $\sigma$ sampled from the training noise schedule (discussed below)
- $\epsilon$ sampled from $N(0, I_n)$ 

The idea is that $\sigma$ is a given value; if you can predict $\epsilon$, then you can recover the un-noisy image $x_0$ from the noisy image $x_\sigma$ using $x_\sigma = x_0 + \sigma \epsilon$. This is a more numerically efficient + stable method than predicting the un-noisy image directly.

In practice, you wouldn't compute loss on a single sample; you would mini-batch multiple samples and sum the loss over those multiple samples (and then compute the gradient and update $\theta$).

**Noise Schedule:**
- discrete set of noise levels (usually 100-1000), sampled uniformly
- Vary widely; usually depicted on log scale:

<img src="ReadingNotesSupplements/DiffusionNoiseSchedule0.png"  alt="" style="width:50%; margin-top: 10px"/>

Common noise schedules include a log-linear schedule, `ScheduleDDPM` (often used in image/pixel-space diffusion models), `ScheduleLDM` (often used in latent diffusion models (models trained on latent space; i.e. have an autoencoder and decoder to shrink the data space)).

Note: during training, you sample a variety of noise level so the model learns how to denoise no matter how much noise the image currently has.

Implementation of generic noise schedule + Log Linear schedule:

```python
class Schedule:
    def __init__(self, sigmas: torch.FloatTensor):
        self.sigmas = sigmas
    def __getitem__(self, i) -> torch.FloatTensor:
        return self.sigmas[i]
    def __len__(self) -> int:
        return len(self.sigmas)
    def sample_batch(self, x0:torch.FloatTensor) -> torch.FloatTensor:
        """
        x0 is a batch of data samples, with first dimension of x0 being the batch size.

        returns tensor of size (x0.shape[0],) containing random values from self.sigmas.
        """
        return self.sigmas[torch.randint(len(self), (x0.shape[0],))].to(x0)

class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float=0.02, sigma_max: float=10):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))

```

### Training a Diffusion Model

In the simplest case, your diffusion model can just be a multi-layer deep NN. As input, it takes an "image" with added noise, and the noise level sample. It outputs the noise vector.

<center><img src="ReadingNotesSupplements/DiffusionToyModel.png" alt="" style="width:100%; margin-top: 10px"/></center>

Then, the training data consists of $N$ sets of (clean image, noisy image, noise level, noise vector).

Basic training loop:
```python
def generate_train_sample(x0: torch.FloatTensor, schedule: Schedule):
    """Generate noise vector and noise level"""
    sigma = schedule.sample_batch(x0)  # Grab random noise level
    eps = torch.randn_like(x0)
    return sigma, eps

def training_loop(loader  : DataLoader,
                  model   : nn.Module,
                  schedule: Schedule,
                  epochs  : int = 10000):
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(epochs):
        for x0 in loader:  # iterate through clean images
            optimizer.zero_grad()
            sigma, eps = generate_train_sample(x0, schedule)  # Every epoch, each image is paried with one sigma
            eps_hat = model(x0 + sigma * eps, sigma)
            loss = nn.MSELoss()(eps_hat, eps)
            optimizer.backward(loss)
            optimizer.step()

```

Consider a toy example of trying to train a diffusion model to produce swill rolls:

<center><img src="ReadingNotesSupplements/DiffusionSwissRoll.png" alt="" style="width:40%; margin-top: 10px"/></center>

This dataset contains 2D points; each point is like a "clean image"; the goal of the diffusion model will be to take arbitrary points and denoise them to produce the closest point that is on the spiral.

We parameterize our diffusion model as a multi-layer NN:
```python
def get_sigma_embeds(sigma):
    """Embed sigma into higher dimensional space"""
    sigma = sigma.unsqueeze(1)
    return torch.cat([torch.sin(torch.log(sigma)/2),
                      torch.cos(torch.log(sigma)/2)], dim=1)

class TimeInputMLP(nn.Module):
    def __init__(self, dim, hidden_dims):
        """
        dims = output dimensions
        hidden_dims = tuple of hidden layer sizes
        """
        super().__init__()
        layers = []
        for in_dim, out_dim in pairwise((dim + 2,) + hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], dim))
        self.net = nn.Sequential(*layers)
        self.input_dims = (dim,)

    def rand_input(self, batchsize):
        """ Generate random noisy input image"""
        return torch.randn((batchsize,) + self.input_dims)

    def forward(self, x, sigma):
        sigma_embeds = get_sigma_embeds(sigma)         # shape: batchsize x 2
        nn_input = torch.cat([x, sigma_embeds], dim=1) # shape: batchsize x (dim + 2)
        return self.net(nn_input)

model = TimeInputMLP(dim=2, hidden_dims=(16,128,128,128,128,16))

schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)
trainer  = training_loop(loader, model, schedule, epochs=15000)
losses   = [ns.loss.item() for ns in trainer]
```

What is the "$\sigma$ embedding"? Typically, instead of taking the scalar $\sigma$ value directly as input, NN's embed $\sigma$ into a higher dimension using `log` and `sin` and `cos` before feeding as input to the model. This is beneficial because (personally, I find these explanations handwavy) -- 1: $\sigma$ might range from 0.01 to 100 (a large range), so it's numerically beneficial to bound it (this is what the `log` does). Then, it's beneficial to map $\sigma$ from a linear structure onto a circle using sines and cosines; this is inspired by "positional encodings" in Transformer models, it makes $\sigma$ cyclic, potentially helping the model see similarities in different noise levels.

The results of the diffusion model, plotted as a vector field (of $\epsilon$) look like this:

<center><img src="ReadingNotesSupplements/DiffusionSwissRollResults.png" alt="" style="width:80%; margin-top: 10px"/></center>

One might observe an interesting phenomena; for small $\sigma$, the $\epsilon$ vectors point clearly to the nearest point in $\mathcal{K}$. But, for larger $\sigma$, the $\epsilon$ vectors appear to just point to the origin; the mean of all the data. This is explained at the bottom of the *Analytically Ideal Denoiser* section. 

### Analytically Ideal Denoiser

Actually... training a diffusion model is not technically necessary. For the loss function $\min_\epsilon L(\theta) := \mathbb{E} || \epsilon_\theta(x_\sigma, \sigma) - \epsilon ||^2$, there's an analytical solution for $\epsilon(x_\sigma, \sigma)$ (assuming $\mathcal{K}$ is a finite set of discrete samples):
$$\epsilon^*(x_\sigma, \sigma) = \frac{\sum_{x_0 \in \mathcal{K}} (x_\sigma - x_0) \exp(-\|x_\sigma - x_0\|^2 / 2\sigma^2)}{\sigma \sum_{x_0 \in \mathcal{K}} \exp(-\|x_\sigma - x_0\|^2 / 2\sigma^2)}$$
One reason we don't just use the analytically ideal denoiser in practice, is because, during inference, it's too computationally expensive to sum over a large dataset; instead, evaluating a large neural network is more efficient.

Let's look at what this analytically ideal denoiser is doing though. The numerator is a weighted average of all $x_\sigma - x_0 = \sigma \epsilon$ (recall that $x_\sigma = x_0 + \sigma \epsilon \; \rightarrow \;x_\sigma - x_0 = \sigma \epsilon$), with weights equal to $\exp(-\|x_\sigma - x_0\|^2 / 2\sigma^2)$ (a Gaussian centered at $x_0$). The denominator provides a scaling factor that normalizes the weights and divides out the $\sigma$ from the numerator. The intuition is this: smaller errors $x_\sigma - x_0$ get a higher weighting; so this ideal denoiser takes a weighted average of all clean images in the dataset, giving more weight to those that are closer to the noisy image.

The way this ideal denoiser is derived is simple; our loss function is $\min_\epsilon L(\theta) := \mathbb{E} || \epsilon_\theta(x_\sigma, \sigma) - \epsilon ||^2$, therefore the ideal denoiser is $\epsilon^*(x_\sigma, \sigma) = \mathbb{E}[\epsilon | x_\sigma, \sigma]$. Then, simply expand.

Here's why, for large $\sigma$, the resulting $\epsilon$ vectors appear to simply point to the mean of all the data. When $\sigma$ is large, the Gaussian $\exp(-\|x_\sigma - x_0\|^2 / 2\sigma^2)$ becomes almost flat, and the weighted average becomes basically just an average. This means, for large $\sigma$, the denoiser basically returns a wrong result, and the smaller $\sigma$ is, the more accurate the result will be.

### Interpreting what Diffusion is Doing

One interpretation of diffusion is as an approximate projection onto the data manifold $\mathcal{K}$, i.e. minimizing the distance from $x$ to $x_0$.

We define a distance function: 
$$\text{dist}_\mathcal{K}(x) := \min\{\|x - x_0 \| : x_0 \in \mathcal{K}\}$$
And a projection function:
$$\text{proj}_\mathcal{K}(x): \{x_0 \in \mathcal{K} : \text{dist}_\mathcal{K}(x)\}$$
We claim, without proof, that $\nabla \frac{1}{2} \text{dist}_\mathcal{K}^2(x, \sigma) = x - \text{proj}_\mathcal{K}(x)$. Therefore, (half) the gradient of the approximate ($\sigma$-smoothed) squared-distance function is approximately the vector between $x$ and the projection of $x$ onto $\mathcal{K}$.

Actually the distance function above is not always differentiable. Therefore, let's instead define our distance function using this other approximate, "$\sigma$-smoothed" squared distance:
$$\text{dist}_{\mathcal{K}}^2(x, \sigma) := \text{softmin}_{x_0 \in \mathcal{K}} \|x_0 - x\|^2 = -\sigma^2 \log \left( \sum_{x_0 \in \mathcal{K}} \exp \left( -\frac{\|x_0 - x\|^2}{2\sigma^2} \right) \right)$$
Of course, the smaller $\sigma$ is, the smaller the "smoothing" is, and the closer this approximate distance function is to the true distance function. (This will be important later -- smaller noise levels lead to more accurate noise vector estimates).

It's been shown that the ideal denoiser for a given $\sigma$ is actually equivalent to (half) the gradient of a $\sigma$-smoothed squared-distance function (which is equivalent to an approximate projection onto $\mathcal{K}$). Therefore, we can claim that $\text{Ideal Denoiser} = \nabla \frac{1}{2} \text{dist}_\mathcal{K}^2(x, \sigma) \approx x - \text{proj}_\mathcal{K}(x)$. This is why Diffusion can be interpreted as an approximate projection onto $\mathcal{K}$.

The Ideal Denoiser then is like, if you are optimizing on the cost function $f(x) = \text{dist}_{\mathcal{K}}^2(x, \sigma)$, then the Ideal Denoiser is like taking one big step equal to the gradient of $f(x)$. One interpretation of DDIM sampling, then, is literally performing gradient descent on $f(x)$. The Ideal Denoiser doesn't perform well because a single large gradient step doesn't take you close to the minimum of $f(x)$ -- but DDIM does perform well because gradient descent performs well.


### Diffusion Model Inference (DDIM Sampling Algorithm)

When using diffusion in the wild, we generally begin with a very noisy image (i.e. very large $\sigma$) and attempt to build a clean image from it. Thus, since $\sigma$ is large, one-step sampling (i.e. just using inferencing from the Model once and taking that $\epsilon$ prediction as the answer) wouldn't work well; it would likely predict very close to the mean of all data samples in $\mathcal{K}$, which isn't very useful. What we want is to end up *on* the manifold (specifically, at the closest point on the manifold).

Instead, the strategy is a multi-step inference, beginning at large $\sigma$ and gradually reducing $\sigma$. 

**DDIM Sampling Algorithm**:
1. Begin with $x_T$ (pure noise).
2. At each step $t = T, ..., 1$:
	1. A small $\sigma_t$ is used
	2. The model estimates $\epsilon \approx \nabla_x (\text{dist}_\mathcal{K}^2(x, \sigma_t))$ 
	3. We perform a partial update $x_t \rightarrow x_{t-1}$ by: $x_{t-1} = x_t - (\sigma_t - \sigma_{t-1})\epsilon_\theta(x_t, \sigma_t)$ 

Each step can also be viewed as a *small gradient step* pushing $x_\sigma$ closer to the manifold $\mathcal{K}$ (i.e. gradient descent on function $f(x) = \frac{1}{2} \text{dist}_\mathcal{K}^2(x)$). Recall the "Approximate Projection" interpretation of the Ideal Denoiser -- taking one large step according to the Idea Denoiser is like taking one huge gradient step on the distance function, which takes you somewhere to the middle of the manifold $\mathcal{K}$... meanwhile, DDIM is like performing gradient descent along the distance function (so we end up at 0 distance from the manifold).
- Step size is $(\sigma_t - \sigma_{t-1})$
- $\nabla f(x_t)$ is estimated by $\epsilon_\theta(x_t, \sigma_t)$

Critical to the performance of DDIM is the $\sigma_t$ schedule; this affects the sizes and number of steps during inference. **Definition**: "Admissible Schedule" $\{\sigma_t\}_{t=0}^T$ ensures $\frac{1}{\nu}\,\mathrm{dist}_\kappa(x_t) \;\le\; \sqrt{n}\,\sigma_t \;\le\; \nu\,\mathrm{dist}_\kappa(x_t)$ (for constant $\nu \geq 1$ and $0 \leq n < 1$). Intuitively, this requirement is saying that $\sigma_t$ roughly corresponds with how far $x_t$ is from the manifold $\mathcal{K}$ (if you are far, $\sigma_t$ is big, if you are close, $\sigma_t$ is small). A common admissible schedule is a log-linear sequence. **Theorem**: if $\{\sigma_t\}_{t=0}^T$ is an admissible schedule, then DDIM converges.

In practice, we may subsample some values from our noise schedule used during training: (adding a function to the `Schedule` class defined above):

```python
class Schedule:
    ...
    def sample_sigmas(self, steps: int) -> torch.FloatTensor:
	    """
	    subsample `steps` number of sigma values from the noise schedule to be
	    used during DDIM sampling
	    """
        indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                       .round().astype(np.int64) - 1)
        return self[indices + [0]]
```

Implementation of DDIM sampling:

```python
batchsize = 2000
sigmas = schedule.sample_sigmas(20)
xt = model.rand_input(batchsize) * sigmas[0]
for sig, sig_prev in pairwise(sigmas):
    eps = model(xt, sig.to(xt))
    xt -= (sig - sig_prev) * eps
```

With 20-step DDIM, the toy example produces the result:
![350](ReadingNotesSupplements/DiffusionSwissRollResultsDDIM.png)

The more DDIM steps, the better the result (at the cost of computation).

### DDPM Sampler

A small modification of DDIM sampling that adds noise during sampling (empirically performs better; "explores distribution more"). To do this, it denoises to a smaller $\sigma_{t'} < \sigma_{t-1}$ at each step, and then adds back noise $w_t \sim N(0,I)$. To be clear: $\sigma_{t'} < \sigma_{t-1} < \sigma_t$.

While the original DDIM sampling scheme looks like:
$$x_{t-1} = x_t - (\sigma_t - \sigma_{t-1})\epsilon_\theta(x_t, \sigma_t)$$DDPM sampling looks like:
$$x_{t-1} = x_t - (\sigma_t - \sigma_{t'})\epsilon_\theta(x_t, \sigma_t) + \eta w_t$$
where $\eta = \sqrt{\sigma_{t-1}^2 - \sigma_{t'}^2}$. Critical to DDPM is how to set $\sigma_{t'}$:
$$\sigma_{t-1} = \sigma_t^\mu \sigma_{t'}^{1-\mu}$$
- if $\mu=0$: $\sigma_{t-1} = \sigma_t \rightarrow$ DDIM sampling
- If $\mu=\frac{1}{2}$: $\sigma_{t-1} = \sqrt{\sigma_t \sigma_{t'}}$ (Geometric mean of $\sigma_t$ and $\sigma_{t'}$) $\rightarrow$  DDPM sampling

The reason we set $\sigma_{t'}$ this way is that it allows us to approximately balance the contribution of the deterministic sampling with the contribution of the added noise in the update vector $x_{t-1} - x_t$ (this, empirically, is a good thing to do): $\mathbb{E}\left[\|\eta w_t\|^2\right] \approx \mathbb{E}\left[\|(\sigma_t - \sigma_t') \epsilon_\theta(x_t, \sigma_t)\|^2\right]$. There's a complex derivation in Chenyang's paper.

Derivation of $\eta$: we want $x_{t-1}$ to end up with a variance equal to $\sigma_{t-1}^2$ to follow our sampling noise schedule. Realize that $x_t - (\sigma_t - \sigma_{t'})\epsilon_\theta(x_t, \sigma_t) = x_{t'}$ has variance $\sigma_{t'}$; also, $\eta \omega_t$ has a variance $\eta$. 

$x_{t-1} = x_t - (\sigma_t - \sigma_{t'})\epsilon_\theta(x_t, \sigma_t) + \eta w_t \implies x_{t-1} = x_{t'} + \eta \omega_t$. Therefore, $\text{Var}(x_{t-1}) = \text{Var}(x_{t'}) + \text{Var}(\eta \omega_t)$. Plugging in the variances explained in the paragraph above, we get: $\sigma_{t-1}^2 = \sigma_{t'}^2 + \eta^2 \implies \eta = \sqrt{\sigma_{t-1}^2 - \sigma_{t'}^2}$.

### Improved Sampler with Gradient Estimation

(recent research by Frank Permenter and Chenyang Yuan)

As mentioned above, $\epsilon_\theta(x_t, \sigma_t)$ can be interpreted as an approximation of the gradient of the distance function $\nabla \text{dist}_\mathcal{K}(x)$, but it is not perfectly accurate. This method aims to add some correction.

It is actually very simple; at each step $t$, it re-estimates $\epsilon_\theta$ using a combination of the $\epsilon_\theta$ estimate at $t$ and $t+1$, ***with*** $\gamma=2$:
$$\bar{\epsilon}_t = \gamma \epsilon_\theta(x_t, \sigma_t) + (1-\gamma) \epsilon_\theta(x_{t+1}, \sigma_{t+1})$$
A visualization (RECALL THAT $\gamma=2$):

![300](ReadingNotesSupplements/DiffusionAcceleratedSampling.png)

Why? Firstly, this can be interpreted as a "correction" for any error from the last time step ($t+1$). Every time step, as $\sigma$ gets smaller, we expect our estimate of $\epsilon$ to get better. Therefore, at $x_{t}$, we have a better estimate of $\epsilon$ than at $x_{t+1}$, and we know that the estimate of $\epsilon$ at $x_{t+1}$ has at least $\epsilon_\theta(x_t) - \epsilon_\theta(x_{t+1})$ error. Therefore, when computing $\bar{\epsilon}$ at $x_t$, we correct for the error from $x_{t+1}$ by adding: $\bar{\epsilon}_t = \epsilon_\theta(x_t) + (\epsilon_\theta(x_t) - \epsilon_\theta(x_{t+1}))$, which is precisely the $\bar{\epsilon}$ update written above, with $\gamma=2$. (Think about the math -- I don't think the diagram is very informative here.)

Another interpretation of this improved sampler is as adding a "momentum" term, similar to accelerated GD. Recall the interpretation of a DDIM sampling step ($x_{t-1} = x_t - (\sigma_t - \sigma_{t-1})\epsilon_\theta(x_t, \sigma_t)$) as a gradient descent step. Also, recall the algorithm for gradient descent with momentum:
$$\begin{align*}v_{t+1} &= \gamma \nabla f(x) + (1 - \gamma) v_t,\\
x_{t+1} &= x_t - \eta v_{t+1}\end{align*}$$
Then, $v_t$ is analogous to $\bar{\epsilon}_t$ , while $\nabla f(x)$ is analogous to $\epsilon_\theta(x_t, \sigma_t)$. In short this improved sampler is almost perfectly analogous to gradient descent with momentum, with $\gamma=2$. (One note here -- conventional wisdom says that gradient descent with momentum wouldn't converge unless $\gamma \in [0,1]$; however, this accelerated sampling algorithm is different in one key way -- it has reducing step sizes based on the noise schedule, which allows convergence; also recall the theorem earlier about "admissible noise schedules" and convergence guarantees).

#### Common Diffusion Model Architectures

A multi-layer NN is not enough for advanced data modalities. Here, we outline 2 common alternative architectures used.

**Convolutional U-Nets**:
- General idea: Encoder and Decoder blocks; encoder decreases dimension of the data while increasing # channels; decoder does the reverse

**Patch-wise Transformers**
- Break image into grid of patches; input patches into NN that creates patch embeddings

## Interpretations of Diffusion
[https://lilianweng.github.io/posts/2021-07-11-diffusion-models/]
#### Probabilistic Interpretation: 

Define data sample from data distribution $x_0 \sim q(x)$.

Forward diffusion = adding Gaussian noise in $T$ steps, producing sequence of noisy samples $x_1, ..., x_T$, with variance schedule $\{\beta_t \in (0,1)\}_{t=1}^T$:
$$\begin{align*}q(x_t | x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I}) \\ q(x_1, ..., x_T | x_0) &= \prod_{t=1}^T q(x_t | x_{t-1}) \end{align*}$$
The first line says that each step $t$ involves sampling $x_t$ from a Gaussian centered at $\sqrt{1-\beta_t} x_{t-1}$, with variance $\beta_t \mathbf{I}$. Clearly, $\beta_t$ controls the amount of added noise versus re-use of the image at the previous step. The second line simply exemplifies the Markov Chain (each state $x_t$ only depends on previous state $x_{t-1}$) nature of the forward diffusion process.

Note: $\beta_t \leftrightarrow \sigma_t^2$ (to draw a connection to the prior interpretation based on Chenyang Yuan's work). $\beta_t$ is a variance; $\sigma_t$ is a standard deviation. 

We can rewrite these equations so that can compute $x_t$ at any $t$ in closed form, without needing to recurse from $x_0$ (define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$): 

$$\begin{align*}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1} & \text{where } \epsilon_{t-1}, \epsilon_{t-2}, \dots \sim \mathcal{N}(0, \mathbf{I}), \\  &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \epsilon_{t-2} & \text{where } \epsilon_{t-2} \text{ merges two Gaussians (*)} \\ &= \dots \\
   &= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \\ 
q(x_t|x_0) &= \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I}).
\end{align*}$$

We can clearly see that this interpretation is the same as above: 
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \quad \leftrightarrow \quad x_\sigma = x_0 + \sigma \epsilon$$


### SDE (Stochastic Diff Eq) Interpretation

### ODE (Ordinary Diff Eq) Interpretation

### Flow Models

### Diffusion Models as Autoregressive Models


## Conditional Diffusion

Above, we've discussed unconditional diffusion; you start from noise, and the diffusion model denoises to the closest image in the dataset. What if your image dataset is super diverse? What if you want to condition on some text prompt?

Consider a simple example: a dataset of images of dogs/cats. An unconditional diffusion model has no way to select between outputting a dog or cat. A conditional diffusion model, conditioned on a label that takes value either "dog" or "cat", can.

Consider your image $x$ and condition/label $y$. Unconditional diffusion learns the clean data distribution $p(x)$; the goal of conditional diffusion is to predict $p(x|y)$.

**The dumb way**: Train your diffusion model to take in both the noisy image and a condition as input and output the clean image. This works... for low-dimensional problems.

### Classifier Guidance

**Guidance**: Only train the model to denoise $x_T$ to $x_0$; only include the condition at sampling time.

**Classifier**: The given condition is a class label (i.e. "cat", "dog").

Let's begin by simplifying the expression $p(x|y)$:
$$\begin{align*} p(x \mid y) &= \frac{p(y \mid x) \cdot p(x)}{p(y)} \\
\implies \log p(x \mid y) &= \log p(y \mid x) + \log p(x) - \log p(y) \\
\implies \nabla_x \log p(x \mid y) &= \nabla_x \log p(y \mid x) + \nabla_x \log p(x), \end{align*}$$
We apply Bayes' Rule, take the $\log$, then take the gradient wrt $x$.

This final equation is useful, because we want to know $\nabla_x \log p(x|y)$ (for optimization/gradient-ascent purposes), and we already know $\nabla_x \log p(x)$ from training the unconditional diffusion model; we just need to figure out how to compute $\nabla_x \log p(y|x)$.

Intuitively, $p(y|x)$ can be approximated by a classifier; it takes an image $x$ and outputs a label/condition $y$ (although $x$ may be noisy during intermediate sampling steps). Therefore, all we have to do is train a good classifier of these images. There is one complexity -- at earlier steps in the diffusion process, $x$ will be very noisy, so we have to train a classifier that is robust to this noise.

**Empirical Hack**: Often, people insert a constant $s>1$ before $\nabla_x \log p(y \mid x)$ called a "guidance strength". This just biases the distribution $\nabla_x \log p(x \mid y)$ further toward the conditional term $\nabla_x \log p(y \mid x)$; it seems to make the effect of the conditioning stronger. 
$$\nabla_x \log p(x \mid y) = s \nabla_x \log p(y \mid x) + \nabla_x \log p(x)$$
Guidance does add an inherent bias (i.e. the $\log p(y \mid x)$ term). Technically, if the classifier were trained on the same training data as the diffusion model and it were perfect, then it would perfectly output $p(y|x)$, and there would be no bias. But the classifier is only an approximation of $p(y|x)$ and contain biases. Therefore, the sampler can output images not-in-distribution. Normally, this is fine/this is what we want, but good to be aware of it.

### Classifier-free Guidance

KEY IDEA:
- Train diffusion model to work both with conditioning variable and without
- At inference time, call the diffusion model twice: once w/ conditioning variable and once without
- Linearly combine the two outputs (with a "classifier guidance" scale/weight)
- Allows trading off prompt-adherence vs realism