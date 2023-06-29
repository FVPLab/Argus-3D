import torch
import torch.nn as nn
from torchvision import models
# from transformer.fast_transformers.builders import TransformerEncoderBuilder
# from transformer.fast_transformers.masking import TriangularCausalMask

class FAST_transformer_builder_uncond(nn.Module):
    """
        Fast Transformer Builder:
        Containing Embedding 
        Args:
            cfg
    """
    def __init__(self, cfg):
        super(FAST_transformer_builder_uncond, self).__init__()
        print("FAST_transformer_builder_uncond")
        # FAST Autoregressive Transformer
        fast_transformer_kwargs = cfg['model']['fast_transformer_kwargs']
        self.attention_type = fast_transformer_kwargs['attention_type']
        self.fast_transformer = TransformerEncoderBuilder.from_kwargs(**fast_transformer_kwargs).get()
        self.init_with_vqvae = cfg['model']['stage2_model_kwargs']['init_with_vqvae']

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
            
        self.z_emb = nn.Embedding(self.vocab_size, embed_dim)
        self.z_start = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, embed_dim)))  # Start Token
        self.position_embed = nn.Embedding(cfg['model']['stage2_model_kwargs']['sequence_length'], embed_dim)

        # z dec 
        self.z_dec = nn.Linear(self.z_emb.weight.size(1), self.z_emb.weight.size(0), bias=False)    # embed_dim --> vocab_size
        self.z_dec.weight = self.z_emb.weight
        self.z_bias = nn.Parameter(torch.zeros(self.z_emb.weight.size(0)))

        self.sequence_length = cfg['model']['stage2_model_kwargs']['sequence_length']
        self.apply(self._init_weights)
        self.config = cfg
        self.drop = nn.Dropout(cfg['model']['stage2_model_kwargs']['embed_drop_p'])

        self.attn_mask = TriangularCausalMask(self.sequence_length-1) # Causal Triangular Mask
    
    def forward(self, z_idx):
        # Forward the Fast Transformer Model
        z_embedding = self.z_emb(z_idx)                         # z_idx: B,L-1,C
        if self.init_with_vqvae:
            z_embedding = self.emb_proj(z_embedding)            # 32 Codebook --> 768
        z_embedding = torch.cat([self.z_start.unsqueeze(0).repeat(z_idx.shape[0], 1, 1), z_embedding], dim=1)  # [B, L, C]
        assert z_embedding.shape[1] <= self.sequence_length, "Cannot forward  Sequence Length Error"
        position_ids = torch.arange(z_embedding.shape[1], dtype=torch.long, device=z_idx.device)
        position_ids = position_ids.unsqueeze(0).repeat(z_idx.shape[0],1)
        position_embeddings = self.position_embed(position_ids)
        x = self.drop(z_embedding + position_embeddings)        # B x L x C
        x = self.fast_transformer(x, attn_mask=self.attn_mask)  # B x L x C
        if self.init_with_vqvae:
            x = self.dec_proj(x)
        
        logits_z = self.z_dec(x) + self.z_bias  # B x L x C

        return logits_z
    
    def autoregressive_forward(self, z_idx):
        # Forward the Fast Transformer Model
        z_embedding = self.z_emb(z_idx)
        if self.init_with_vqvae:
            z_embedding = self.emb_proj(z_embedding)    # 32 Codebook --> 768
        z_embedding = torch.cat([self.z_start.unsqueeze(0).repeat(z_idx.shape[0], 1, 1), z_embedding], dim=1)  # [B, L, C]
        assert z_embedding.shape[1] <= self.sequence_length, "Cannot forward  Sequence Length Error"
        position_ids = torch.arange(z_embedding.shape[1], dtype=torch.long, device=z_idx.device)
        position_ids = position_ids.unsqueeze(0).repeat(z_idx.shape[0],1)
        position_embeddings = self.position_embed(position_ids)
        x = self.drop(z_embedding + position_embeddings)   # B x (L) x C
        # x = z_embedding + position_embeddings

        attn_mask = TriangularCausalMask(x.shape[1], device=x.device)
        x = self.fast_transformer(x, attn_mask=attn_mask)  # B x (L) x C
        if self.init_with_vqvae:
            x = self.dec_proj(x)
        
        logits_z = self.z_dec(x) + self.z_bias  # B x (L) x 1024     [img, start_token, index1=0]

        return logits_z


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            

        




        








    


