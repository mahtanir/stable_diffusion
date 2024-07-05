import torch 
from torch import nn 
from torch.nn import F 
from attention import SelfAttention

#MAKE SURE nth WORD AT BOTTOM or first word is first in. 

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embed, n_tokens) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed)) #seq_len, embed. ie for each embedding feature, pos info. IS N_TOKENS vocab? 
        #don't use sinuosidal like in normal transofrmer, use learnt params that tell model positional info. 

    def forward(self, tokens):
        #batch_size, seq_len -> batch_size, seq_len, n_vocab
        #nn.Embedding just simplifies this. Instead of giving it a big one-hot vector, you just give it an index. This index basically is the same as the position of the single 1 in the one-hot vector.

        #batch_size, seq_len, n_vocab -> batch_size, seq_len, embedding
        x = self.token_embedding(tokens)

        #batch_size, seq_len, embedding ->  batch_size, seq_len, embedding
        x = x + self.positional_embedding
        return x 

class CLIPLayer(nn.Module): #like encoder here 
    def __init__(self, n_head: int, n_embed: int) -> None:
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4*n_embed) #feed forward layers
        self.linear_2 = nn.Linear(4*n_embed, n_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x 
        x = self.layernorm_1(x)
        x = self.attention(x, casual_mask = True)
        x += residual #same as transformer, attention, add residual, norm 
        #FEED FORWARD LAYER
        residual = x 
        x = self.layernorm2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(x*1.702) #quick GELU activation function (really only diff with normal transformer -> activation)
        # JUST EXPERIMENTAL no reason in theory why sigmoid and constant
        x = self.linear_2(x)
        x += residual 


#WHERE DOES THE IMAGE EMBEDDING CORRELATION LOSS COME UP? Was it implicit? i.e learn to denoise based on prompt so learns similar prompt -> image 

class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77) #vocab size, embedding size, max_seq_len  --> start by putting positional info, then get embeddings with semantic meaning.
        #https://chat.openai.com/c/83d96046-6b56-4be5-9319-97233bd6c235 for input embedding, supposadly CLIP simialr to encoder. one hot encode then embedding weight matrix. 
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12) #num of heads of attention, embedding size,
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor: 
        #long tensor because tokens are positions of token in the vocab 
        tokens = tokens.type(torch.long) #is this tokens one hot encoding or just a number? Seems to just be a number. A long tensor in PyTorch refers to a tensor with integer numbers as its data type. Specifically, in PyTorch, a long tensor is a tensor with a torch.int64 data type
        #Batch size, seq_len -> Batch_size, Seq_Len, Dim
        state = self.embedding(tokens)
        for layer in self.layers: #attention infuse the embeddings? --> attention is stacked. 
            #I think general logic is CLIP_Embedding for token imagey encoding then attention to infuse contextaul meaning. 
            state = layer(state) #maybe like stacked attention? Diff though because embeddings for text to match image. 
            #i.e helps to encode similar text similarly to match to same images, so attention helps to add contextual and 
            #ease semantic information encoding. 
        
        # (Batch size, Seq Len, Dim) 
        output = self.layernorm(state)
        return output
        
