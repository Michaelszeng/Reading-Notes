
## 10/24/25 - Shao’s Talk
- context: shao’s goal is to train a model that outputs *diverse* trajectories (so that he can use beam search to teach it multi-modal actions)
- Models “generalize” before they “memorize”
	- learn low frequency signals before then lean higher frequency signals
	- After a certain training step, validation gets worse but loss continues to go down—this is basically memorization/overfitting
	- SOLUTIONS:
		- terminate training when memorization occurs
		- introduce more diversity into dataset so even the memorized actions have diversity
- smoothing (gaussian smoothing kernel) the score function “identifies manifold structure”
- the model conditions on two things: the observations and the noisy image
	- shaos results show that the model seems to care much more about observations than noisy image
	- because, by introducing multiple labels per observation in the dataset, loss is much higher
	- also, in the case where there is a single label per observation in the dataset, no matter the noisy image passed to the model, it always denoises to the same trajectory. so it learns to ignore the noise and just predict the trajectory from the observations
- interesting idea: more diversity in dataset (i.e. multi-label per observation) makes more robust (has more options to get out of infinite failure loops) but slower (switches between modes more often)

## 10/3/25 - Abhinav’s talk
Long-context diffusion-policy
- Literature Review
	- PTP (past token prediction) — basically claiming it doesn’t actually work that well
		- original diffusion policy paper also does some level of PTP (i.e. predicts the 2_obs_steps)
		- possible that PTP is actually very important, chelsea’s paper is just bad
	- TODO: understand history guided diffusion paper
	- Trace VLA, interesting, selects key points of moving object using LLM’s and adds visual traces to the image — helps surprisingly a lot
		- adding more images hurts training
- problem w long context is overfitting to really old observations (and ignoring new observations)
	- this is remedied by having more training data
	- note that this should not be a real problem — the model can theoretically learn to ignore older observations to at least match performance
	- also, this doesn’t occur for much simpler problems like linear systems
- How conditioning is done
	- observation encoder spits out obs tokens
	- different MLP to reshape obs tokens into various shapes
	- each reshaped obs token injected into each layer of unet
	- unet denoises actions
	- recent work by nvidia on using video encoders instead of image encoders
		-  note: usually for single task policies we don’t *freeze* the image encoder, it trains as well. for multi task policies (i.e. LBM) we do freeze the image encoder? or fine tune it somehow 

- Results: 
	- image based policies are more sensitive to long context than state based. (i.e. larger drop in performance at longer contexts)
	- first idea, freeze observation encoder from 2 obs context, then run that for 16 obs context
		- helps a lot
		- also, freezing obs encoder and training diffusion policy separately helps
		- another idea — condition obs encoder on time step instead?
			- abhinavs argument: better to leave time conditioning external to obs encoder which is optimized for spatial feature extraction (i.e. via self-attention)
		- he did also try adding a time embedding — said to just helped a little bit
	- second idea — add self attention to obs tokens
		- also helps a lot
		- adds temporal understanding?
	- test - at inference time — adding random noise to oldest images in context 
		- this completely messes with the model, even for the parts of the task that should not need long context; implies that the model is learning to use very old observations even when it doesn’t have to
	- idea: randomly sample context length during training (i.e. sometimes you predict more past tokens and fewer future tokens, sometimes vice versa)
		- at inference time, fix to the maximum context length
		- reduces catastrophic failure — so this does help learn to rely more on the recent observations

## 9/26/25 - Adam’s talk
Ambient Diffusion
- fact from information theory: adding noise “contracts” distributions — i.e. if you add noise to distributions, they become more similar. at the limit, if you add infinite noise to 2 images, they become the same distribution
- in diffusion, if you add enough noise, the diffusion score function of both distributions becomes approximately the same
- thus, there always exists a noise level at which even a bad image is close enough to the distribution of noises good images

## 7/6/25 - Random RSS Papers
- [A Biconvex Method for Minimum-Time Motion Planning Through Sequences of Convex Sets · Robotics: Science and Systems](https://roboticsconference.org/program/papers/44/)
	- Basically GCS that, instead of using convex relaxation, solves alternating programs fixing the set-continuity points and set-continuity-point-velocities
- [Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking · Robotics: Science and Systems](https://roboticsconference.org/program/papers/116/)
	- Evaluation showing that:
		- geometric controllers perform better in steady state
		- RL controllers perform better for agile transient behaviors
- [Building Rome with Convex Optimization · Robotics: Science and Systems](https://roboticsconference.org/program/papers/32/)
	- Faster SLAM/Bundle Adjustment using learned-depth and with a new SDP solver
- [Effective Sampling for Robot Motion Planning Through the Lens of Lattices · Robotics: Science and Systems](https://roboticsconference.org/program/papers/48/)
	- Maximally efficient sampling in <= 21 dimensions using mathematical lattice structure
	- 100x faster than uniform sampling
	- $\delta$-$\epsilon$ theorems for samples
- 

## 3/14/25 - RLG Group Meeting - Alex Talk
- Shortest path as linear program — “Primal Dual Method” to solve Dijkstras
	- Sketch: 
		- 4 programs: Primal, Dual, RP, DRP (Dual of RP)
		- RP is used to validate optimality; introduces slack into constraints; if slack is 0, then primal is feasible (and dual is feasible, so optimality is reached)
		- DRP is used to figure out how much you can change the dual variables by; conceptually similar to setting theta in dual simplex 
		- iterative algorithm to update dual variables until RP reports optimality
- Main goal: GCS solver using “Primal Dual-like” Method


## 3/7/25 - RLG Group Meeting - Shao New Project Proposal
- When training diffusion model - want prediction long enough to capture the multi-modal decision  (i.e. in box unloading, long enough to decide on a box)
	- otherwise, the robot begins moving without any clear direction
	- but, you don’t necessarily need super long horizon, so long as the horizon captures sufficient multi-modality to accurately guide the robot when it begins the move
- Longer horizon prediction (without any sort of receding horizon) also weights each action less
	- i.e. for very very long horizon prediction, they usually don’t output executable plans; need an additional low level controller to try to follow the long plan
	- meanwhile, for shorter horizons, the plan is directly executable

- How to use search to increase performance beyond training from an expert (i.e. try to outperform the expert)
	- Simplest idea: Simple Lookahead/Beam Search
		- Predict n steps in b beams (use trained network to make discrete decisions, then solve trajectory to global optimality using Kin. Traj. Opt.); evaluate value network at last step in each beam; pick the best beam, then roll forward its first step (similar to MPC)
		- Repeat this many times; use the resulting output/trajectory as a new, better, training data point
		- KEY IDEA: basically takes the best output from the expert of each b sampled beams
	- Using the better training data points, do some post-training

- key reading: expert iteration paper with alpha go?

**Bernard’s Talk - SDP Relaxation of Spherical Constraints**
- Spherical constraints: i.e. obstacle avoidance w spherical obstacles, or enforcing robot link length (i.e. distance btwn 2 robot joints is constant)
	- a subset of quadratic constraints where Q is identity (sort of)
- KEY IDEA 1: SDP Relaxation is possible specifically for Spherical constraints (not general quadratic) bc Q=Identity then x^T Q x => x^T x which is an entry of the matrix variable X (but you cannot substitute x^T Q x with a matrix variable)
- KEY IDEA 2: Very geometric interpretation; i.e. for 2D shortest piece wise linear path w/ spherical obstacles if solution is higher rank, the path is sort of shortcut the goes thru 3rd dimension
- COOL IDEA (tho concluded probably won’t work) — solve the unrelaxed SDP using alternation methods; maintain copy of G and x and try to get them to reach convergence
- Future direction: new branch and bound algorithm to perform the relaxation?
- Often: take relaxed solution, pass to SNOPT as initial guess, do nonlinear rounding

## 2/21/25 - Robotics @ MIT - Zac Manchester (Composable Optimization for Robotics)
- ReLU-QP -- analytically reformulate QP as weight matrix? Solves much faster (2.5 KHz Atlas full-body control)
- Non-smooth dynamics
	- Classic hybrid approaches: pre-specify contact sequence, MPC works between the contacts
	- Contact-Implicit approaches: optimize contact forces also
- **Key Idea:** "Composable Optimization": 
	- For an optimization problem: $\min_x f(x,y) \quad s.t. \; c(x) \geq 0$
	- Old idea: Log barrier $\min f(x) - \frac{1}{\rho} log(c(x))$
		- smaller $\rho$ --> smoother transition
		- **Key Idea**: iterative solving starting from small $\rho$ to large $\rho$
	- cost explodes when x = 0
	- Optimality conditions: $r(x,y) = 0$
	- Implicit function theorem: $\frac{\delta x}{\delta y}$

## 2/21/25 - RLG Group Meeting - Lu job talk
- Ways Model-based and Learning are combined:
    
    - Model-based cross-embodiment data generation/augmentation from few demonstrations (Lu’s most recent work)
        
    - Learning cost functions, models, for a Model-based controller
        
    - Developing NN’s inspired by model-based methods w/ certifiable guarantees (another of Lu’s work)
        
    - RL-inspired contact-dynamics smoothing for model-based planners/controllers (Terry, Pang, and Lu’s work)
        
- Overall theme: bridging the gap between deep understanding of Model-based methods on simple systems, and empirical (but shallow understanding) of learning-based methods on complex systems

- Also; Russ is very good at spinning together narratives/selling it

## 2/7/25 - RLG Group Meeting
- PRM is bad in high dimensions — probability of finding a path scales with Volume(C_free) / Volume(B_eps)