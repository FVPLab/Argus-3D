import torch
import torch.nn as nn
from torchvision import models
# from transformer.fast_transformers.builders import TransformerEncoderBuilder
# from transformer.fast_transformers.masking import TriangularCausalMask
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, embed_dim, n_head):
        super().__init__()
        # assert config.n_embd % config.n_head == 0
        n_embd = embed_dim
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):  ## x : B x L x Dimension
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))   # Weight Matrix
        # mask:[B,1,L,L]
        att = att.masked_fill(mask == 0, float('-inf'))

        if x.dtype == torch.float16:
            att = att.to(torch.float32)
            fp16 = True
        else:
            fp16 = False
        att = F.softmax(att, dim=-1)
        if fp16:
            att = att.to(torch.float16)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, embed_dim, n_head):
        super().__init__()
        n_embd = embed_dim
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(embed_dim=n_embd, n_head=n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            # nn.GELU(),  # nice, GELU is not valid in torch<1.6
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.01),
        )

    def forward(self, x, mask=None):        # x: 
        x = x + self.attn(self.ln1(x), mask)   # mask: tril mask
        x = x + self.mlp(self.ln2(x))
        return x


class GPTPrediction(nn.Module):
    def __init__(self, embed_dim):
        super(GPTPrediction, self).__init__()
        n_embd = embed_dim
        self.ln1 = nn.LayerNorm(n_embd)
        self.dense = nn.Linear(n_embd, n_embd)
        self.gelu = GELU()
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.ln2(x)
        return x

class FAST_transformer_builder_uncond_baseline_reso32_cond_quant_single_cond_class(nn.Module):
    """
        Fast Transformer Builder:
        Containing Embedding 
        Args:
            cfg
    """
    def __init__(self, cfg):
        super(FAST_transformer_builder_uncond_baseline_reso32_cond_quant_single_cond_class, self).__init__()
        print("FAST_transformer_builder_baseline No drop Quant single cond class")
        # FAST Autoregressive Transformer
        fast_transformer_kwargs = cfg['model']['fast_transformer_kwargs']
        self.attention_type = fast_transformer_kwargs['attention_type']
        # self.fast_transformer = TransformerEncoderBuilder.from_kwargs(**fast_transformer_kwargs).get()

        ## Origin Transformer
        transformer_layer = cfg['model']['stage2_model_kwargs']['transformer_layer']
        transformer_embed_dim = cfg['model']['stage2_model_kwargs']['transformer_embed_dim']
        transformer_n_head = cfg['model']['stage2_model_kwargs']['transformer_n_head']

        self.blocks = nn.ModuleList([Block(embed_dim=transformer_embed_dim,n_head=transformer_n_head) for _ in range(transformer_layer)])
       
        self.dec = GPTPrediction(embed_dim=transformer_embed_dim)

        self.init_with_vqvae = cfg['model']['stage2_model_kwargs']['init_with_vqvae']
        # self.drop_sign = nn.Linear(2,1)         # 248 Checkpoint
        # self.no_drop_sign = nn.Linear(4,1)      # Youtu Checkpoint 
       
        # Embedding
        self.vocab_size = cfg['model']['quantizer_kwargs']['embedding_num']


        if not self.init_with_vqvae:
            embed_dim = cfg['model']['stage2_model_kwargs']['stage2_embed_dim']   # e.g 768
            self.emb_proj = None
            self.dec_proj = None
        else:
            embed_dim = cfg['model']['quantizer_kwargs']['embedding_dim']        # e.g 32
            self.emb_proj = nn.Linear(embed_dim, cfg['stage2_model_kwargs']['stage2_embed_dim'])    # 32 --> 768
            self.dec_proj = nn.Linear(cfg['model']['stage2_model_kwargs']['stage2_embed_dim'], embed_dim)    # 768 --> 32
            
        self.emb = nn.Embedding(self.vocab_size, embed_dim)       # 1024: End Token   1025: Padding Token

        print('z emb shape',self.emb.weight.shape)

        position_num = cfg['model']['stage2_model_kwargs']['position_num']

        self.pos_emb = nn.Embedding(position_num + 1, embed_dim)
        self.value_start = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, embed_dim)))  # Value Start Token

        # Class embedding:
        self.class_embedding = nn.Embedding(55, embed_dim)

        # value dec 
        self.z_value_dec = nn.Linear(self.emb.weight.size(1), self.emb.weight.size(0), bias=False)    # 768 --> 1024 + 1 + 1
        self.z_value_dec.weight = self.emb.weight
        self.bias_value = nn.Parameter(torch.zeros(self.emb.weight.size(0)))

        self.sequence_length = cfg['model']['stage2_model_kwargs']['sequence_length']  # 1024 + 1 = 1025

        self.apply(self._init_weights)
        self.config = cfg
        self.drop = nn.Dropout(cfg['model']['stage2_model_kwargs']['embed_drop_p'])
    
    # @autocast()
    def forward(self, plane_index, class_idx):
        class_embed = self.class_embedding(class_idx).unsqueeze(1)  # B x 1 x C
        # import pdb
        # pdb.set_trace()

        embedding = self.emb(plane_index) # B reso^ C /8,1,e

        embedding = torch.cat([self.value_start.unsqueeze(0).repeat(plane_index.shape[0], 1, 1), embedding], dim=1)     # [B, L+1, C]

        bs = embedding.shape[0]
        pos_embedding = self.pos_emb(torch.arange(embedding.shape[1], device=embedding.device).reshape(1,-1)).repeat(bs, 1, 1)#8,2,e
        assert embedding.shape[1] <= self.sequence_length, "Cannot Forward Sequence Length Error"
        x = embedding + pos_embedding

        x = torch.cat([class_embed, x], dim=1)#8,3,e

        mask = torch.ones(bs, 1, x.shape[1], x.shape[1])#8,1,3,3

        mask = torch.tril(mask).to(x.device)
        # Origin Transformer:
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.dec(x)#8,3,e
        logits_value = self.z_value_dec(x) + self.bias_value#8,3,8192
        logits_value = logits_value[:,1:-1,:]
        return logits_value


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            

        




        








    


