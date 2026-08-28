
### Dyna's Take (https://www.youtube.com/watch?v=f8ckFlzGWGo)
- Online RL is not necessary
- Only need a reward model
- The most efficient use of robot time is just use it to find the best intervention points for collecting HG-DAgger corrections (based on the reward stagnating)
- Dyna's first model used purely AWR
- Dyna's second model (called Dyna-1) used the reward model for determining when to have human intervene
- Dyna's thesis might not hold for robots already in the wild?

### $\pi_{0.6}^*$
- Policy conditions on binary good/bad label
	- First, computes continuous advantage label, but then thresholds this before passing to the model as conditioning
	- Advantage is computed using a $N$-step returns bootstrapped with their trained value function: $A^{\pi_{\mathrm{ref}}}(o_t, a_t) = \left[ \sum_{t'=t}^{t+N-1} r_{t'} + V^{\pi_{\mathrm{ref}}}(o_{t+N}) \right] - V^{\pi_{\mathrm{ref}}}(o_t)$
- Value function is trained using simple supervised targets (no value-bootstrapping/TD learning)
	- It's also not a "value function" per se -- it doesn't measure the value of a state under a given policy. It's more of a general reward function.

#### [RLPD - Efficient Online Reinforcement Learning with Offline Data](https://arxiv.org/pdf/2302.02948))
Background
- Problem setting: kick-start RL training with offline data, without any explicit offline pre-training phase (Avoids "offline-to-online" 2-step training procedures).
- Claims to generalize to many types of offline data, including human demos and sub-optimal trajectories; agnostic to offline data quality
3 Core Contributions:
1. **Symmetric Sampling**
	- For each batch, draw 50% from offline dataset, 50% from online interactions dataset
2. **Layer Normalization**
	- During environment interaction, actor may enter OOD states $\rightarrow$ critic is queried w/ OOD actions $\rightarrow$ over-estimation bias
		- *over-estimation bias* in Q-learning: critic over-estimates values of states $\rightarrow$ compounds due to bootstrap critic targets 
			- In Q-learning, the critic target is $y_t = r_t + \gamma \max_{a} Q_\phi(s_{t+1}, a)$
			- since critic is used in its own training target, an over-estimated Q-value (even for just a single $a$ in $s_{t+1}$, because of the $\max$) leads the critic to learn to over-estimate other states to $\rightarrow$ divergence
	- Intuition: constrain critic's ability to extrapolate to unseen inputs
		- Using *LayerNorm* in networks bounds extrapolated values of networks on OOD inputs
			- They show, mathematically, using LayerNorm on the last layer bounds the network output to $\le \| w \|_2$ where $w$ are the network weights of the last layer $$\begin{aligned}
\underbrace{Q_{\theta,w}(s,a)}_{\text{scalar}}
&= \|w^T (\psi_\theta(s,a))\| \\
&\leq \|w\| \, \|(\psi_\theta(s,a))\| \\
&\leq \|w\|
\end{aligned}$$
				- Intuition: the last layer of the network takes dot product of weight vector $w$ with the intermediate feature vector $\psi_\theta(s,a)$ (after the LayerNorm is already applied -- therefore $\|\psi_\theta(s,a)\| \le 1$).
		- This is a helpful bound because $\psi_\theta(s,a)$ is likely close to 1 for some in-distribution features; OOD features therefore can't produce significantly larger Q-values that in-distribution features.
3. High UTD (Update-to-Data) Ratio + Ensemble of Q-networks
	- High UTD alone causes critic overfitting
	- To resolve this, use a large number (i.e. 10) of Q-critic networks each initialized randomly
		- During each critic update, all critics use the same boostrapped target, which is determined using the *min* (to prevent over-estimation) Q-value from a random subset (i.e. of 2) critics from the ensemble
			- i.e. $y_t = r_t + \gamma \min_{i \in Z} Q_i(s_{t+1}, a_{t+1})$ where $Z$ is a random subset of critics
		- A different subset of critics is used for each critic update
		- During policy updates, use the average $Q$ value over all critics 
		- The reason Ensemble of Q-networks helps increase UTD is that, by using a *different* subset of critics for the bootstrapped critic target for each critic update, we prevent overfitting to i.e. a single fixed, noisy critic target.
- Besides these 3 design decisions, typical off-policy actor-critic algorithm

#### [SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning]([2401.16013](https://arxiv.org/pdf/2401.16013))
Summary: a framework for real-world RL
- Core algorithm: RLPD (off-policy actor critic)
- Reward function: 
	- Either, train a binary classifier off state-based information with pos./neg. examples
	- Or: VICE
		- Problem: image-based RL policies require a success classifier to give the model its reward
			- RL policy can learn to exploit states where the classifier erroneously gives high success probability
		- Solution: Add all states visited by policy into training set for the classifier with negative labels and retrain the classifier in a DAgger fashion
			- Eventually, the classifier will learn to correctly classify the state distribution induced by RL policy
- Forward-Backward Controllers
	- Idea: to avoid manual resetting of task during training, train 2 completely separate RL actors/critics at once: one to complete the task, the other to reset the task
[HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning]([hil-serl-paper.pdf](https://hil-serl.github.io/static/hil-serl-paper.pdf))
- Basically just adds human intervention during the rollout phases of the RLPD algorithm
	- The human intervention transitions get added to both the rollout buffer and the demo buffer, so it may be sampled more often (since RLPD samples 50% of data from rollout buffer and 50% of data from demo buffer)

## Residual RL
[Residual Off-Policy RL for Finetuning Behavior Cloning Policies](https://arxiv.org/pdf/2509.19301)
- Key idea: frozen action-chunking diffusion base policy; separate RL policy producing residual actions at every timestep (non-chunked)
- RL policy learns:
	- $\pi_\theta(s_t, a_t^{\text{base}})$ -- actor, conditions on current observations and base policy action
	- $Q_\phi(s_t, a_t)$ -- critic, conditions on current observations and full action where full action $a_t = a_t^{\text{base}} + \pi_\theta(s_t, a_t^{\text{base}})$ (sum of base policy and RL policy actions)
- Critic Loss:  $$L(\phi) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}, d_t) \sim \mathcal{D}} \bigg[\bigg(Q_\phi(s_t, a_t) - \left(r_t + \gamma (1 - d) Q_\phi \bigl( s_{t+1}, a_{t+1} \bigr)\right)\bigg)^2\bigg]$$
	- $d \in \{0,1\}$ indicates if current state is terminal
	- $a_{t+1} = a_{t+1}^{\text{base}} + \pi_\theta(s_{t+1}, a_{t+1}^{\text{base}})$ is the next full action
	- This is effectively the difference between the current critic's value $Q_\phi(s_t, a_t)$ and the *optimal* value according to Bellman equation: $Q^*(s_t,a_t) = \mathbb{E}[r_t + \gamma \max_{a'} Q^*(s_{t+1}, a')]$
		- Since $Q^*$ is unknown, they use the approximation $\max_{a'} Q(s_{t+1}, a') \approx Q\bigl(s_{t+1}, \pi_\theta(s_{t+1})\bigr)$, which is true when $\pi_\theta(s_{t+1})$ is close to the optimal $a'$; i.e. the policy is already quite good
			- This is called ***bootstrapping***
- Actor Loss:  $$L(\theta) = - \mathbb{E}_{(s_t, a_t^{\text{base}}) \sim \mathcal{D}} \left[ Q_\phi \bigl(s_t, a_t^{\text{base}} + \pi_\theta(s_t, a_t^{\text{base}})\bigr) \right]$$
	- Basically, just use gradient ascent to train actor to maximize critic's value
- Training Procedure:
	- Sample mini-batch of transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$ from replay buffer
	- Compute next action $a_{t+1} = \pi^{\text{base}}(s_{t+1}) + \pi_\theta(s_{t+1}, a_{t+1}^{\text{base}})$
	- Use $a_{t+a}$ to perform a critic update (gradient descent)
	- Perform actor update
- Training Notes & Training Stability 
	- Add small Gaussian noise to $a_{t+1}$ before it is used for the critic update/loss function
		- Adds regularization -- policy must learn general regions of actions that are good, not learn to exploit sharp peaks in the learned Q-function
	- Delayed actor updates -- only update actor every $k$ mini-batches
		- Allows the critic to meaningfully learn the value of the policy before the policy changes
	- Use UTD (update-to-data ratio) $>1$ $\rightarrow$ multiple model update steps per real-world data point/environment step collected
		- Increases data efficiency
	- Randomized Ensembled Double Q-Learning -- ensemble of (i.e. 10) Q-networks initialized independently, trained together
		- At every critic eval, randomly select 2 networks, evaluate, and take minimum of the two as the value
	- EMA on Critic Networks
	- $n$-step returns: Instead of using only $r_t$ (i.e. a 1-step reward) in the critic loss's target, use $r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{n-1} r_{t+n-1}$
		- Makes the target more accurate and helps learn with sparse rewards -- the sparse reward at the end of the episode propagates to earlier timesteps faster
		- The critic loss becomes this (for $n=3$): $$L(\phi) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}, d_t) \sim \mathcal{D}} \bigg[\bigg(Q_\phi(s_t, a_t) - \left(r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 (1 - d) Q_\phi \bigl( s_{t+3}, a_{t+3} \bigr)\right)\bigg)^2\bigg]$$
			- Each sample in the minibatch is of the form $(s_t, a_t, r_t, r_{t+1}, r_{t+2}, s_{t+3}, d_t)$
	- Replay Buffer contains 50% frozen demonstration data + 50% online buffer data


___
### Practical Tips
- High entropy --> jittery robot
	- should see this go away by epoch 100-200. Or else, entropy might be too high.

### High Level
- Use value function to estimate total discounted future reward to decide what action to take next
- Use immediate reward $r(s^{(t+1)})$ reaped by that action to update Value function

### Jargon
- "on-policy" + "online" = agent interacts with env to generate new data, only the new data generated by *current* policy is used for training (PPO).
- "off-policy" + "online" = agent interacts with the env to generate new data, but may keep data from interactions generated by old versions of the policy (i.e. keeps a replay buffer) (SAC).
- "off-policy" + "offline" = pre-generated dataset (no agent interaction with env) by some other policy; agent trying to learn a new policy.
- "on-policy" + "offline" isn't really possible
- Value function $V_\pi(s)$: expected future return if you start in $s$ and execute $\pi$
- Q function $Q_\pi(s, a)$: expected future return if you start in $s$, execute $a$ immediately, and follow $\pi$ after
- Advantage function $A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s)$: how much better $a$ is than the average action from $s$

### Offline-to-Online RL
Given an offline dataset (i.e. human demonstrations); you start by pre-training base policy on offline dataset (i.e. using BC, advantage weighted regression, AWR, CAL-QL, etc.)
- This is the current paradigm in real-world RL for manipulation (RL from scratch in the real world is basically impossible)
- The Online stage can be either On-policy or Off-policy
	- Off-policy: you keep the offline dataset in the replay buffer; 50/50 sample from offline data vs newly generated online data
		- Pro: Much more sample efficient, regularizes policy toward good behavior, may improve coverage of very rare states
		- Con: Much less stable training
			- Very wide data distribution and very little data in the actor's actual current state/action regime $\rightarrow$ critic much more likely to be wrong
	- On-policy: you discard the offline dataset after the offline pre-training phase; only fine-tune based on on-policy data
		- Pro: Much more stable training
		- Con: Much less sample efficient

### Types of RL
Policy-based: train a neural net policy to maximized expected returns
Value-based: train a neural net value network to satisfy the Bellman optimality equation; then the policy is simply a greedy execution according to the value network
- i.e. DQN (Deep Q-Network)
- Bellman optimality equation: $Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s', a') | s, a]$
	- Interpretation:
		- Start in state $s$, take action $a$
		- Get immediate reward $r$, then land in next state $s'$
		- Behave *optimally* from $s'$ onward, total (discounted) future value from $s'$ is $\gamma \max_{a'} Q^*(s', a')$
		- Self-consistency condition: "being optimal from now on" $\implies$ value now = immediate reward + optimal discounted future reward
- Loss: $L_Q(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}[(Q_\theta(s,a) - y)^2]$, where $y$ is the "training target": $y = r + \gamma \max_{a'} Q_{\theta-}(s', a')$, where $\theta-$ are parameters of target network (delayed copy of $\theta$)
	- $y$ is intended to mimic the Bellman-optimal value $Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s', a') | s, a]$, but since $Q^*$ is unknown, we approximate using our learned $Q_{\theta-}$
	- The reason we use $\theta-$ instead of $\theta$ is training stability; we want $y$ to be a constant so our loss is a simple MSE loss. During training, we update $\theta-$ every so often, but mostly keep it fixed for training stability.
		- In the ideal case (for the loss to "correctly" encode the Bellman optimality equation), we use $\theta$ to compute $y$.
Actor-Critic: train both a policy and value network
- Off-policy (SAC):
	- Q-function loss ($\phi_j$ are the Q-network parameters)
		- $L_Q(\phi_j) = \mathbb{E}_{(s,a,r,s')}[(Q_{\phi_j}(s,a) - y(r,s'))^2]$, where $y(r,s') = r + \gamma \mathbb{E}_{a' \sim \pi_\theta(\cdot | s')}[\min_{j=[1,2]} Q_{\phi_j -}(s', a') - \alpha \log \pi_\theta(a' | s')]$
			- This tries to satisfy a soft (with-entropy) version of the Bellman Optimality Equation
			- $\phi_j -$ is once again a delayed copy of parameters of Q-network
			- The $\alpha \log \pi_\theta(a' | s')$ term comes from entropy: $\alpha H(\pi(\cdot | s_t)) = -\alpha \mathbb{E}_{\alpha \sim \pi}[ \log \pi_\theta(a | s) ]$
	- Policy loss 
		- $L_\pi(\theta) = \mathbb{E}_{s,a \sim \pi_\theta}[\alpha \log \pi_\theta(a | s) - Q_\phi^{soft}(s,a)]$
			- Maximize both entropy and expected Q-value (weighted by $\alpha$)
		- Equivalent to solving $\arg\min_\phi D_{KL}\big(\phi(\cdot | s) ~||~ \exp(\frac{1}{\alpha} Q^*(s,a)))\big)$
			- - To maximize the value function w.r.t. $\pi$, you get that $\pi^*(a | s) \propto \exp(\frac{1}{\alpha} Q^*(s,a))$. Thus, we parameterize $\pi$ like so
	- Policy loss encourages both high reward and high entropy
		- $J(\pi) = \mathbb{E}[\sum_t \gamma^t(r(s_t, a_t) + \alpha H(\pi(\cdot | s_t)))]$
		- Optimal policy is $\pi^*(a | s) \propto \exp(\frac{1}{\alpha} Q^*(s,a))$
	- Alternate between actor and critic updates

### Training
- 2-phase loop:
	1. Rollout/data collection (with exploration)
	2. Update/gradient optimization (for both actor + critic)
- Rollout/data collection phase:
	- Actor samples $a_t \sim \pi_\theta(\cdot | s_t)$
	- Step the environment
	- Store transition $(s_t, a_t, r_t, s_{t+1}, V_\phi(s_t))$ into replay buffer
	- Repeat for a fixed horizon
- Gradient optimization phase:
	- Use critic to get value estimates $V_\phi(s_t)$
	- Use $r_t$, $V_\phi(s_t)$, discount factor $\gamma$ to compute:
		- Discounted returns $G_t$
		- Advantages $A_t$: how much better or worse this action $G_t$ was than expected ($V_\phi(s_t)$): $A_t = G_t - V_\phi(s_t)$
	- Split replay buffer into minibatches, compute policy loss and value loss, compute entropy bonus to encourage exploration, combine into a total loss $L = L_{policy} + L_{value} - \text{entropy}$
	- Apply back-prop
	- Stop early if $D_{KL}(\pi_{\theta_{old}} \| \pi_{\theta_{new}})$ is too large
- Notes:
	- entropy is rewarded, so at the start of training you should expect entropy to increase (become more negative); but as the policy learns well, it should lower loss more by decreasing entropy and repeating the optimal actions; thus entropy should eventually decrease.

### Reward Shaping
- General Idea: want dense rewards to guide agent toward desired behavior without changing optimal policy
- **Potential-Based Reward Shaping**:
	- formal guarantee that adding a specific kind of dense reward **will not change the optimal policy**
	- $\Phi(s)$: potential function
		- high at "good" states, low at "bad" states (can be based on expert or heuristic)
		- Reward Shaping: $F^{(t+1)} = \gamma * \Phi(s^{(t+1)}) - \Phi(s^{(t)})$
			- $\gamma$ = "discount factor"
		- $R_{total}^{(t)} = r(s^{(t)}) + F^{(t)}$
			- $r(s^{(t)})$ = current sparse environment reward
		- $R_{total}^t$ is used to train the policy
		- Over entire episode, total discounted sum of the future shaping rewards:
			- $\sum_{t=0}^{t_{goal}-1} \gamma^t F^{(t+1)} = \gamma^{t_{goal}} * \Phi(s^{(t_{goal})}) - \Phi(s^{(0)})$
				- $\Phi(s^{(0)})$ is constant w.r.t. path taken to the final state
				- $\Phi(s^{(t_{goal})})$ is constant where all $s^{(t_{goal})}$ reap same reward
			- Thus, total shaping rewards are constant w.r.t. path taken by the agent
			- KEY IDEA: When agent estimates the total discounted future reward to decide which action to take next, does not affect the agent's choice
- **More Common in Practice: Generic Dense Rewards**
	- Ensure monotonic progress during phase transitions by making the minimum reward in next phase equal to the maximum reward of the previous phase
	- Reward early success by giving final reward for all remaining timesteps