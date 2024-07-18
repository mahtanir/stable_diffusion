import torch 
from torch import nn 
from torch.nn import functional as F
import math 

#for multihead we actually split d_model after getting query, key and value vectors. 
#Make sure input shape channels d_embed is the same as what is passed in.
#TODO: check that dim ae n_word * n_embed for qkv
class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_embed, d_embed*3, bias=in_proj_bias) #seq, d_embed -> seq, d_embed W, K, V matrices hence the times three at once. 
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias) #W0 matrix at the end with the heads seq, d_v*n_heads -> seq, d_model
        self.d_head = d_embed // n_heads 
        self.n_heads = n_heads
    
    def forward(self, x:torch.tensor, casual_mask=False) -> torch.tensor:
        

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape #dembed is channels, seq_len is the number of pixels here 
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head) #if multi-headed attention

        #batch_size, seq, d_embed -> batch_size, seq, d_embed*3
        x = self.in_proj(x) #W, K, V
        #batch_size, seq, d_embed -> 3 tensors batch_size, seq, d_embed
        Q, K, V = torch.chunk(x, 3, dim=-1) #same as three diff projections 
        #batch_size, self.n_heads, seq_len, self.d_head
        Q = Q.view(interim_shape).transpose(1,2)
        K = K.view(interim_shape).transpose(1,2)
        V = V.view(interim_shape).transpose(1,2)

        attention_weights = torch.matmul(Q, K.transpose(-1,-2)) #remember in matmul we consider n_seq, dim for the matrix as transpose. Batch and head count is same. 
        if (casual_mask):
            #mask is diagonal bottom right to top left. All above  are 1, else 0. This is what we mask, so n1 top is completely masked but for itself, n2 has n1,n2 only, etc... 
            mask = torch.ones([seq_len, seq_len], dtype=torch.bool).triu(1)
            attention_weights.masked_fill_(mask, -torch.inf) #replaces w/ value where mask is true

            # attention_weights = torch.dot(attention_weights, mask) not DOT, we add! 
        attention_weights /= math.sqrt(self.d_head)
        attention_weights = F.softmax(attention_weights, dim=-1) #softmax across the columns, makes sense for each pixel. 
        
        #(batch_size, self.n_heads, seq_len, seq_len) @ (batch_size, self.n_heads, seq_len, self.d_head) ->
        # (batch_size, self.n_heads, seq_len, self.d_head)
        output = torch.matmul(attention_weights, V)

        #concat the heads (batch_size, seq_len, self.n_heads, self.d_head)
        output = output.transpose(1,2)
        output = output.reshape(input_shape)
        #W0 matrix 
        output = self.out_proj(output)
        return output 
    
class CrossAttention(nn.Module):
    def __init__(self, n_head, d_embed, d_cross, in_proj_bias = True, out_proj_bias = False) -> None:
        super().__init__()
        #do projections and then split up 
        self.proj_Q = nn.Linear(d_embed, d_embed)
        self.proj_k = nn.Linear(d_cross, d_embed) #context
        self.proj_v = nn.Linear(d_cross, d_embed)
        self.n_head = n_head
        self.d_head = d_embed // n_head #how much info each head will see 
        self.out_proj = nn.Linear(d_embed, d_embed)
    
    def forward(self, latent, context):
        #latent dims are batch_size, seq_len_q (pixels), d_embed
        #context dims are batch size, seq_len_kv (token len), d_cross = batch size, 77, 768)
        input_shape_latent = latent.shape
        batch_size, _, d_embed = input_shape_latent 
        new_shape = (batch_size, -1, self.n_head, self.d_head)
        Q = self.proj_Q(latent)
        K, V = self.proj_k(context), self.proj_v(context)

        # Batch_size,seq_q, d_embed-> Batch_size,n_head, seq_q, d_h
        Q = Q.view(new_shape).transpose(1,2) 
        # Batch_size,seq_kv, d_cross-> Batch_size,n_head, seq_kv, d_h
        K = Q.view(new_shape).transpose(1,2)
        V = Q.view(new_shape).transpose(1,2)

        attention_matrix = torch.matmul(Q, K.transpose(-1,-2)) #Batch_size, n_head, seq_q, seq_kv -> so the attention of pixels to each token and vise versa (each token pays attention to each pixel and vise versa)
        attention_matrix = attention_matrix / math.sqrt(self.d_head)
        
        weights = F.softmax(cross_attention, dim=-1) #how much attention each pixel relative to the token 
        #-> Batch_size,n_head, seq_q, d_h
        cross_attention = attention_matrix @ V
        output = cross_attention.transpose(1,2).contiguous()
        #-> batch_size, seq_len_q (pixels), d_embed
        output = output.view(input_shape_latent)

        return self.out_proj(output)











