# Question: Why do we want Transformer architecture over previous ones? 

## Answer: Transformers are capable of parallel processing rather than strictly sequential, and they are able to retain earlier information as a result to perform better.

# Question: What is a transformer? 

## Answer: It is still a neural network at the end of the day, it simply processes inputs and generates outputs differently

# Question: What makes a transformer different from a simple feed-forward neural network? 

## Answer: The attention mechanism between layers. It allows all tokens to in some way communicate with one another directly for learning purposes. Each token looks at all other tokens and decides which ones are important for better learning. 

# Question: What benefit does the attention mechanism give to the transformer over previous architectures? 

## Answer: The attention mechanism gives the benefit of efficient capturing of context across tokens. 

# Question: What are the key parts of the transformer architecture. 

## Answer: There are two key parts to this architecture, seen below. The combination of these two layers is what allows the transformer to learn so efficiently.
* The feed-forward layer: tokens privately refine themselves based on standard backpropagation mechanisms. 
* The attention mechanism: tokens communicate with one another to refine themselves in some way 

# Question: Can you walk me through how the transformer takes a piece of text, and then learns on that text to generate the next piece of text? 

## Answer: First, the input text is tokenized by the tokenizer into smaller pieces called tokens. These tokens are embedded, meaning they are transformed into numerical vectors intended to capture their semantic meaning after training. These embeddedings are injected with positional information to give a sense of order to the transformer, so the transformer can know where each token is in the sequence. Now, the tokens pass into the attention layer to communicate with one another, and then into the feed-forward layer to refine themselves. This process is repeated n times, n > 0. We will finally get the embeddings back out at the end, but the context in which we use these results differs. 

# Question: Can you explain more about the attention mechanism? 

## Answer: The attention mechanism first creates three vectors for each of the embeddings: a query vector, key vector, and a value vector. The query vector asks "what am I looking for?", the key contains "here is what I have", and the value carries the actual content to share. In other words, the query vector implicitly asking what concept it refers to, the key vector describes what information it is holding, and the value vector carries the meaning of the token. To decide which tokens are relevant, we take the query vector of a token and dot product it against all key vectors in the sequence. The higher the value of the dot product of a particular key vector and query vector, the more relevant the token with that key vector is to the token with that query vector. These dot product results are normalized (typically using softmax) so that they represent attention weights. These weights act as focus labels, indicating which token is the most relevant to that particular token. Then, each query vector is updated to be the weighted sum of all the value vectors across all other tokens in the sequence, where the weights are the attention weights. This allows the context for each token to grow richer, which is important for learning. This process gives a new context-aware representation for each token, where the most relevant information for each token is blended in. This is represented mathematically as Attention(Q, K, V) = softmax((Q(K)^T) / (sqrt(d_k))) * V, where Q, K, and V are matrices of their respective vectors, and `^T` is the transpose of the Key matrix. This allows for parallel matrix multiplication to occur rather than linear processing of each token, one by one. At the beginning of training, all query, key, and value parameters are random and meaningless. Over time, the attention scores allow the model to learn meaningful patterns. 