# this is taken from the link : https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
import torch 


# Step 1: embedding an input sentence 

sentence = 'Life is short, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)

#assign integer index to each word in the sentence 
sentence_int = torch.tensor([dc[i] for i in sentence.replace(',','').split()])

# encode the input in an embedding layer. we use 16 dimensional embedding. 
# total words in sentence : 6, so it is 6 x 16 embedding 
# each input word is represented by a 16-dimensional vector. 

torch.manual_seed(123)
embed = torch.nn.Embedding(num_embeddings = 6, embedding_dim = 16) 
embedded_sentence = embed(sentence_int).detach()

# embedding layer is like a linear layer but ineffficient (https://medium.com/@gautam.e/what-is-nn-embedding-really-de038baadd24)
# embed_linear  = torch.nn.Linear(4, 3, bias = False)
# embed_linear.weight 
# embed = torch.nn.Embedding(4, 3).from_pretrained(embed_linear.weight) 
# embed.weight 

# the benefit of embedding is that each word has same lengh ( 16 in this case) 

# STEP 2 :  Define the weight matrices 
# scaled dot product attention is just one of the self attention mechanism. 
# now each word has the same length d = 16 
d = embedded_sentence.shape[1]

d_q, d_k, d_v = 24, 24, 28

W_query = torch.rand(d_q, d)    # 24x 16 
W_key = torch.rand(d_k, d)    # 24x 16 
W_value = torch.rand(d_v, d)  # 28x 16 

# STEP 3 : Computing the Unnormalized Attention Weights

# a query is that input vector multiplied with W_q for which attention vector is calculated
x_2 = embedded_sentence[1]   # as an example we calculate it for one word 
query_2 =  torch.matmul(W_query, x_2)   # can alos be written as  W_query.matmul(x_2)   shape of 1x24 
# key_2 =  torch.matmul(W_key, x_2)   # can alos be written as  W_query.matmul(x_2)
# value_2 =  torch.matmul(W_value, x_2)   # can alos be written as  W_query.matmul(x_2)

# now scale it to calculate for all the words
keys = torch.matmul(W_key, embedded_sentence.T).T   # shape is 6 x 24  ( number of word x embedding size )
values = torch.matmul(W_value, embedded_sentence.T).T

# now calcuate the unnormalized attention weights 

# a(2,i) = softmax(w(2,i)/ sqrt(d(k)))
import torch.nn.functional as F 
omega_2 = torch.matmul(query_2, keys.T)
attention_weights_2 = F.softmax(omega_2/((d_k)**0.5), dim = 0)

context_vector_2 = torch.matmul(attention_weights_2, values) 


# Now we get to multi head attention , lets say head  = 3 
h = 3 
multihead_W_query = torch.rand(h, d_q, d)    # 3x 24x 16 
multihead_W_key = torch.rand(h, d_k, d)    # 3x 24x 16 
multihead_W_value = torch.rand(h, d_v, d)  # 3x 28x 16 

multihead_query_2 =  torch.matmul(multihead_W_query, x_2)   # can alos be written as  W_query.matmul(x_2)   shape of 3x24 

stacked_inputs = embedded_sentence.T.repeat(3,1,1)

multihead_keys = torch.bmm(multihead_W_key, stacked_inputs)   # shape is 3x 6 x 24 
multihead_values = torch.bmm(multihead_W_value, stacked_inputs)


# since keys are in order ( number of word x embedding size ), we permute multihead_keys 

multihead_keys = multihead_keys.permute(0,2,1)
multihead_values  = multihead_values.permute(0,2,1)

# omega_2 = torch.matmul(query_2, keys.T)
multihead_omega_2 = torch.bmm(multihead_query_2.repeat(1,1,1).permute(1,0,2), multihead_keys.permute(0,2,1))

# attention_weights_2 = F.softmax(omega_2/((d_k)**0.5), dim = 0)
# multihead_attention_weights_2 = TODO 

# context_vector_2 = torch.matmul(attention_weights_2, values) 
# multihead_context_vector_2 = TODO 


print('done')
