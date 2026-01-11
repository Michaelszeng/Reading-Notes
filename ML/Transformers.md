[3Blue1Brown]()

**Notation**
- Learned weight matrices:
	- $W_E \in \mathbb{R}^{token\_dim\times vocabulary\_size}$: embedding matrix: map every possible word in the vocabulary to a token in latent space
	- $W_U \in \mathbb{R}^{vocabulary\_size \times token\_dim}$: Unembedding matrix: map last token (in the last column of last layer of the network) to a vector of probabilities for each word in the vocab.
	- $W_Q \in \mathbb{R}^{token\_dim \times key/query\_space}$: Query matrix: map token into key/query space
	- $W_K \in \mathbb{R}^{token\_dim \times key/query\_space}$: Key matrix: map token into key/query space
	- $W_V \in \mathbb{R}^{token\_dim \times token\_dim}$: Value matrix: maps token into value vector for token (which lives in embedding/token space)
		- Parameterized by two smaller matrices: $W_V = W_{V_{\uparrow}}^\top W_{V_{\downarrow}}$, where $W_{V_{\uparrow}}^\top$ and $W_{V_{\downarrow}}$ $\in \mathbb{R}^{key/query\_space \times token\_dim}$
			- Basically, this is just to reduce the parameter count. There's no real reason we choose $W_{V_{\uparrow}}^\top$ and $W_{V_{\downarrow}}$ to be these precise dimensions, other than to be consistent with $W_Q$ and $W_K$.

### Self-Attention
- Consider $x$ and $x'$, token vectors in embedding space
- Queries and Keys:
	- Rough idea: queries and keys measure how relevant a word is to another
	- $W_Q x$ is a query vector of $x$
	- $W_K x'$ is a key vector of $x'$
	- We measure the dot product $W_Q x \cdot W_K x'$ to see how much the $x'$ ***attends*** $x$.
- Attention pattern/matrix:
	- For a whole sequence of tokens, we create a matrix of dot products of every combination of tokens $W_Q x \cdot W_K x'$
		<center><img src="ReadingNotesSupplements/self_attention_pattern.png" alt="" style="width:600px; margin-top: 10px"/></center>	
	- We apply softmax to every column so that every column sums to 1.
		- This way, all key tokens share 1 unit of weight to attend to a particular query token
- Values:
	- $W_V x'$ generates value vector, a vector in embedding space, that indicates how you apply the meaning of $x'$ to another token $x$
	- $W_V$ is multiplied to all tokens, then we add a weighted sum of all values to every token, weighted by the attention matrix
		- Example for a single $x =$ *"creature"*, which is attended to by "fluffy" and "blue":
				<center><img src="ReadingNotesSupplements/self_attention_value_Example.png" alt="" style="width:600px; margin-top: 10px"/></center>	
		- Again, this is applied to all tokens:
			<center><img src="ReadingNotesSupplements/self_attention_value.png" alt="" style="width:800px; margin-top: 10px"/></center>	
- Attention masks:
	- usually, want to mask later tokens from affecting earlier tokens (in a next-token-prediction setting)
	- Before applying softmaxes, set lower-diagonal terms to $-\infty$ (so they are zeroed by softmax)
			<center><img src="ReadingNotesSupplements/self_attention_mask.png" alt="" style="width:600px; margin-top: 10px"/></center>	
- Summary:
				<center><img src="ReadingNotesSupplements/self_attention_equation.png" alt="" style="width:600px; margin-top: 10px"/></center>	

### Cross Attention
- Same as self-attention except applying to two distinct "groups" of tokens (i.e. in translation -- cross attend between tokens from each language)
- $W_Q$ applied to one group, $W_K$ applied to the other group
		<center><img src="ReadingNotesSupplements/cross_attention_pattern.png" alt="" style="width:600px; margin-top: 10px"/></center>	
- Don't typically use any attention masking


### Multi-headed Attention
In 1 multi-headed attention block:
- Run $N$ self-attention head in parallel
- Each outputs a vector $\Delta \vec{E}^{(j)}_i$ for each token $i$ in the sequence (for self-attention head $j$)
- Sum these together: $\Delta \vec{E}_i = \sum_{j=1}^N \Delta \vec{E}^{(j)}_i$
- $\Delta \vec{E}_i$ is what's actually added to token $i$ after this single block

Most architectures then have many multi-headed attention blocks in series


### Practical Code Implementation
- For all attention heads, $W_Q, W_K$ horizontally stacked into single matrices
	- Most implementations constrain $key/query\_dim * N = token\_dim$ where $N$ is the number of heads
		- In code, you'll often see:  `assert token_dim % num_heads == 0`
	- Therefore, $W_Q$ and $W_K$ are both square matrices (technically, they're $N$ separate $token\_dim \times key/query\_dim$ matrices stacked together)
- $W_V$ matrices are handled a little differently
	- $W_{V_\downarrow}$ for all attention blocks are horizontally stacked and called the "Value" matrix
	- $W_{V_\uparrow}$ for all attention blocks are horizontally stacked and called the "Output" matrix
	- "Value matrix" ($W_{V_\downarrow}$) is multiplied by each key token $x'$ to get a value in $query/key$ space
	- A weighted sum (for all key tokens $x'$) is computed
	- This weighted sum is mapped back into token-space by multiplying by "Output matrix" ($W_{V_\uparrow}$)


**Transformers as BERT-style Encoders**
- Note: Transformers can be used for things other than next token-prediction; i.e. encoding sequential information into a single output (in an autoencoder)
	- A special `[CLS]` token is prepended to the token sequence. Its value is learned
	- It goes through the same multi-head attention + MLP layers until it gets the meaning of all the other tokens baked into it
	- The final output of the transformer is the embedded version of the `[CLS]` token. 
		- This is different from next-token-prediction transformers, where the output is the embedded version of the last token given in the token sequence.