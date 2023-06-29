import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):# B,res/8,res/8,4   4   0.4
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.eps = 1e-6

        self.embedding = nn.Embedding(self.n_e, self.e_dim) # 16384 * 256 x  second param is project dim, val is 4
        # print('\n\n\n\n\n\n\n\n\n\nn-e')
        # print(self.n_e)
        # print(self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        # self.unet3d = UNet3D(**unet3d_kwargs)   # Plus Unet 3d

    def forward(self, z, mask=None):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,reso,reso,reso)
            2. flatten input to (B*H*W,C)   (B,C,reso,reso,reso) ??
        """
         # Shape: B x C x reso x reso
        bs = z.shape[0]
        if z.dtype == torch.float16:
            fp16 = True
            z = z.to(torch.float32)
        else:
            fp16 = False

        # z.shape: B x C x reso x reso  ---> reshape (B,H,W,C) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
        z_flatten = z.view(-1, self.e_dim)          # shape: (Num, C)

        # distances from z to embedding e_j  = (z - e)^2  = z^2 + e^2 - 2 e * z   shape: Num_1 x embed_num   Distance Matrix
        d = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flatten, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)   # Num_1 x 1

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)  # zeros: Num_1 x Num_2
        min_encodings.scatter_(1, min_encoding_indices, 1)                          # scatter function: get the min position 0-1 matrix  Shape: Num_1 x embed_num

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)      # B x reso x reso x C


        # [B,D,H,W]
        if mask is None:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        else:
            mask = F.interpolate(mask, (z.shape[2], z.shape[3]), mode='nearest')
            mask = mask.repeat(1, z.shape[1], 1, 1)
            mask_sum = torch.sum(mask, dim=[2, 3]) + self.eps
            loss_z = (z_q.detach() - z) ** 2
            loss_zq = (z_q - z.detach()) ** 2
            loss = torch.mean(torch.sum((loss_z + self.beta * loss_zq) * mask, dim=[2, 3]) / mask_sum)

        # preserve gradients直通估计        Shape: B x reso x reso x reso x C
        z_q = z + (z_q - z).detach()

        # # perplexity
        # e_mean = torch.mean(min_encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape  ----> Origin: B x C x reso x reso 
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()        # B x reso x reso x C

        if fp16:
            z_q = z_q.to(torch.float16)

        return z_q, loss, [min_encodings, min_encoding_indices.reshape(bs,-1)]


    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        indices = indices.reshape(-1,1)     # num x 1
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)   # B x reso x reso x C
            # reshape back to match original input shape
            # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

    def approximate_codebook(self, probs, shape):
        # get quantized latent vectors
        # [B,L,V]X[V,D]=[B,L,D]
        z_q = torch.matmul(probs.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
    
    def get_zero_index(self):
        embed = self.embedding.weight.clone().cpu()
        zero = torch.zeros(1,self.e_dim)
        d = torch.sum(zero ** 2, dim=1, keepdim=True) + \
            torch.sum(embed ** 2, dim=1) - 2 * \
            torch.matmul(zero, embed.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1)   # Num_1
        index = int(min_encoding_indices[0])
        return index





