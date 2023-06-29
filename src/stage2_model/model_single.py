import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from src.encoder import encoder_dict
from src.conv_onet import models as VQVAE
from torch.nn import CrossEntropyLoss
# from .fast_transformer_builder_uncond_single import FAST_transformer_builder_uncond_single
import math
from tqdm import tqdm
import pandas as pd

class GET_FAST_Transformer_Single(nn.Module):
    """
        Get Fast Transformer Stage 2 Model  Single
        Args:
            cfg, device, dataset
    """
    def __init__(self, cfg, device=None, dataset=None):
        super(GET_FAST_Transformer_Single, self).__init__()

        self._device = device
        self.config = cfg
        self.name = 'FAST_Transformer_Single'
        print('Model: ====> ', self.name)
        self.nll_loss = CrossEntropyLoss()
        self.autoregressive_steps = cfg['model']['stage2_model_kwargs']['sequence_length'] - 2   # 4098 -2 = 4096
        self.reso = cfg['model']['encoder_kwargs']['plane_resolution']  # 64

        # Get Occupancy Network
        decoder = cfg['model']['decoder']
        encoder = cfg['model']['encoder']
        dim = cfg['data']['dim']
        c_dim = cfg['model']['c_dim']
        decoder_kwargs = cfg['model']['decoder_kwargs']
        encoder_kwargs = cfg['model']['encoder_kwargs']
        padding = cfg['data']['padding']

        # VQVAE Quantizer
        quantizer = cfg['model']['quantizer']
        embed_num = cfg['model']['quantizer_kwargs']['embedding_num']
        embed_dim = cfg['model']['quantizer_kwargs']['embedding_dim']
        self.codebook_dim = embed_dim
        self.embed_num = embed_num
        beta = cfg['model']['quantizer_kwargs']['beta']
        # unet3d_kwargs = cfg['model']['encoder_kwargs']['unet3d_kwargs']
        unet2d_kwargs = cfg['model']['encoder_kwargs']['unet_kwargs']
        c_dim = cfg['model']['c_dim']
        quantizer = VQVAE.quantizer_dict[quantizer](n_e=embed_num,e_dim=embed_dim,beta=beta,c_dim=c_dim,unet_kwargs=unet2d_kwargs)

        # VQVAE Decoder
        decoder = VQVAE.decoder_dict[decoder](dim=dim, c_dim=c_dim, padding=padding,**decoder_kwargs)

        # VQVAE Encoder
        encoder = encoder_dict[encoder](dim=dim, c_dim=c_dim, padding=padding, **encoder_kwargs)

        self.vqvae_model = VQVAE.ConvolutionalOccupancyNetwork(
            decoder, quantizer, encoder, device=device
        )

        self.vqvae_model.eval()
        self.restore_from_stage1()      # Restore VQVAE Model

       # Fast Transformer
        self.transformer_model_single = FAST_transformer_builder_uncond_single(cfg).to(device)


        # Fast Transformer Optimizer
        # named_params = self.transformer_model.parameters()
        self.optim = optim.Adam(self.transformer_model_single.parameters(), lr=float(cfg['training']['stage2_lr']))    # default: 1e-4
     

        for name, params in self.transformer_model_single.named_parameters():
            print(name,':',params.size())

    def forward(self):
        raise NotImplementedError
    

    def get_losses(self, info_dict):
        '''Calculate Losses  3 Plane'''
        nll_loss_all = 0
        tgt_plane_index_xz = info_dict['xz'][1]     # B x (reso x reso)
        tgt_plane_index_xy = info_dict['xy'][1]     # B x (reso x reso)
        tgt_plane_index_yz = info_dict['yz'][1]     # B x (reso x reso)

        # 3B x L x C
        logits = self.transformer_model_single(tgt_plane_index_xz[:,:-1],tgt_plane_index_xy[:,:-1],tgt_plane_index_yz[:,:-1])    
        
        _, _, c = logits.shape
        concat_tgt_index = torch.cat([tgt_plane_index_xz, tgt_plane_index_xy, tgt_plane_index_yz],dim=0)    # 3B x L
        nll_loss_all = self.nll_loss(logits.reshape(-1,c), concat_tgt_index.reshape(-1))
        
        return nll_loss_all
    

    def get_weight_losses(self,info_dict):
        "Calculate weight loss 3 plane"
        nll_loss = 0
        tgt_plane_index_xz = info_dict['xz'][1]     # B x (reso x reso)
        tgt_plane_index_xy = info_dict['xy'][1]     # B x (reso x reso)
        tgt_plane_index_yz = info_dict['yz'][1]     # B x (reso x reso)

        # logits_predict = self.transformer_model(tgt_index, shape_img)   # B x reso^3 x vocab_size
        # nll_loss = self.nll_loss(logits_predict.reshape(-1, logits_predict.shape[-1]), tgt_index.reshape(-1))
        logits_xz = self.transformer_model_xz(tgt_plane_index_xz[:,:-1])    
        logits_xy = self.transformer_model_xy(tgt_plane_index_xy[:,:-1])
        logits_yz = self.transformer_model_yz(tgt_plane_index_yz[:,:-1])
        

        _, seq_len, c = logits_xz.shape
        nll_loss_xz = self.nll_loss_weight(logits_xz.reshape(-1,c), tgt_plane_index_xz.reshape(-1))    # revise 2.23
        nll_loss_xy = self.nll_loss_weight(logits_xy.reshape(-1,c), tgt_plane_index_xy.reshape(-1))
        nll_loss_yz = self.nll_loss_weight(logits_yz.reshape(-1,c), tgt_plane_index_yz.reshape(-1))

        mask_xz = ((tgt_plane_index_xz == self.mask_index_xz) * 1.0).reshape(-1)      # revise: 2.23   index_xz = 251
        num_xz = mask_xz.sum()
        mask_xy = ((tgt_plane_index_xy == self.mask_index_xy) * 1.0).reshape(-1)      # index_xy = 1020
        num_xy = mask_xy.sum()   
        mask_yz = ((tgt_plane_index_yz == self.mask_index_yz) * 1.0).reshape(-1)      # index_yz = 1020
        num_yz = mask_yz.sum()

        xz_bg_loss, xz_fg_loss = (nll_loss_xz * mask_xz).sum() / (num_xz + 1e-5), (nll_loss_xz * (1-mask_xz)).sum() / (mask_xz.shape[0] - num_xz + 1e-5)
        xy_bg_loss, xy_fg_loss = (nll_loss_xy * mask_xy).sum() / (num_xy + 1e-5), (nll_loss_xy * (1-mask_xy)).sum() / (mask_xy.shape[0] - num_xy + 1e-5)
        yz_bg_loss, yz_fg_loss = (nll_loss_yz * mask_yz).sum() / (num_yz + 1e-5), (nll_loss_yz * (1-mask_yz)).sum() / (mask_yz.shape[0] - num_yz + 1e-5)
       
        nll_loss_xz = xz_bg_loss + xz_fg_loss
        nll_loss_xy = xy_bg_loss + xy_fg_loss
        nll_loss_yz = yz_bg_loss + yz_fg_loss

        nll_loss = nll_loss_xz + nll_loss_xy + nll_loss_yz

        return nll_loss, nll_loss_xz, nll_loss_xy, nll_loss_yz, xz_bg_loss, xz_fg_loss, xy_bg_loss, xy_fg_loss, yz_bg_loss, yz_fg_loss
    
    @torch.no_grad()
    def Stage1_encode_inputs(self, inputs):
        '''
            Stage1 VQVAE Encoder --> Encode inputs
        '''
        c = self.vqvae_model.encode_inputs(inputs)
        return c

    @torch.no_grad()
    def Stage1_decode(self, p, c, **kwargs):
        '''
            Stage1 VQVAE Decoder --> returns occupancy probabilities for the sampled point
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        p_r = self.vqvae_model.decode(p, c, **kwargs)
        return p_r
    
    @torch.no_grad()
    def Stage1_quantize(self, inputs):
        '''
            Stage1 VQVAE Quantize ---> Quantize the Feature Grid
        '''
        c_quantize, loss_dict, info_dict = self.vqvae_model.quantize(inputs)
        return c_quantize, loss_dict, info_dict
    
    @torch.no_grad()
    def Stage1_quantize_get_index(self, inputs):
        '''
            Stage1 Quantize Without Unet ---> Get index
        '''
        _, _, info_dict = self.vqvae_model.quantize_get_index(inputs)
        return _, _, info_dict

    
    def backward(self, loss=None):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def restore_from_stage1(self):
        load_path = self.config['stage1_load_path']
        if os.path.exists(load_path):
            print('=> Loading Stage1 Model From {}'.format(load_path))
            state_dict = torch.load(load_path)
            self.vqvae_model.load_state_dict(state_dict['model'])
    

    @torch.no_grad()
    def batch_predict_auto_plane(self,bs, plane_type,device, temperature=1.0, top_k=250,greed=False):
        steps = self.autoregressive_steps
        print('Current plane type: {}, Greed: {}'.format(plane_type,greed))
        x = torch.zeros(bs,steps,dtype=torch.long).to(device)
        if plane_type == "xz":
            transformer_model = self.transformer_model_xz
        elif plane_type == 'xy':
            transformer_model = self.transformer_model_xy
        elif plane_type == 'yz':
            transformer_model = self.transformer_model_yz

        for i in tqdm(range(steps-1)):
            logits_predict = transformer_model(x[:,:i+1])
            logits = logits_predict[:,i,:] / temperature
            if top_k:
                    logits = self.top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)       # B x C
            if not greed:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)  # B x 1  --->   [5]
            ix = ix.squeeze(1)
            x[:,i] = ix             # Out: x shape: B x sequence_length 
        # y = x
        # y = y.cpu().numpy()
        # data = pd.DataFrame(y)
        # data.to_csv("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/vis_index_3plane/out_{}.csv".format(plane_type))
        return x
    
    @torch.no_grad()
    def batch_predict_auto_plane_single(self,bs,device,temperature=1.0, top_k=250,greed=False):
        steps = self.autoregressive_steps   # 4098
        print('Generating 3plane with Single Transformer Model')
        x = torch.zeros(3*bs, steps, dtype=torch.long).to(device)    # xz plane sequence
 
        for i in tqdm(range(steps)):
            logits_predict = self.transformer_model_single(x[:bs,:i+1],x[bs:2*bs,:i+1],x[2*bs:3*bs,:i+1])
            logits = logits_predict[:,i,:] / temperature    # 3B x C
            if top_k:
                    logits = self.top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)       # 3B x C
            if not greed:
                ix = torch.multinomial(probs, num_samples=1) # 3B x 1
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)  # 3B x 1
 
            xz, xy, yz = ix.split(bs,0)   # B 
            xz, xy, yz = xz.squeeze(1), xy.squeeze(1), yz.squeeze(1)
            x[:bs, i], x[bs:2*bs, i], x[2*bs:3*bs, i] = xz, xy, yz  # Update 3 Plane Parallel
        
        xz_plane_idx = x[:bs,:]       #    B x seq_len
        xy_plane_idx = x[bs:2*bs,:]   #    B x seq_len
        yz_plane_idx = x[2*bs:3*bs,:] #    B x seq_len

        return xz_plane_idx, xy_plane_idx, yz_plane_idx


    @torch.no_grad()
    def batch_predict_autoregressive(self,bs,device):
        '''
            Autoregressive generate latent Feature Grid c  with single transformer
        '''
        index_xz, index_xy, index_yz = self.batch_predict_auto_plane_single(bs,device=device)

        # Generate Feature Plane From codebook
        feature_plane_shape = (bs, self.reso, self.reso, self.codebook_dim)
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape)
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape)
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape)
        fea_dict = {'xz':plane_xz, 'xy':plane_xy, 'yz':plane_yz}
        return fea_dict


    
    # @torch.no_grad()
    # def batch_predict_autoregressive(self,bs,device):
    #     '''
    #         Autoregressive generate latent Feature Grid c
    #     '''
    #     index_xz = self.batch_predict_auto_plane(bs,plane_type='xz',device=device)  # B x seq_len
    #     index_xy = self.batch_predict_auto_plane(bs,plane_type='xy',device=device)  # B x seq_len
    #     index_yz = self.batch_predict_auto_plane(bs,plane_type='yz',device=device)  # B x seq_len
        
    #     # Generate Feature Grid From Codebook
    #     feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
    #     plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
    #     plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
    #     plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
    #     fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
    #     return fea_dict   # B x C x reso x reso x reso


    @torch.no_grad()
    def batch_predict_directly(self, shape_img, temperature=1.0, top_k=False, greed=False):
        '''
            Input Noise Sequence & Condition image ---> Output Sequence directly
        '''
        bs = shape_img.shape[0]
        noise = torch.randint(self.embed_num, (bs, self.autoregressive_steps)).to(shape_img.device)
        logits = self.transformer_model.autoregressive_forward(noise, shape_img)    # B x L x C
        logits = logits / temperature
        if top_k:
            logits = self.top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if not greed:
            bs, seq_len, c_dim = probs.shape
            probs = probs.reshape(-1, c_dim)
            ix = torch.multinomial(probs, num_samples=1)
            ix = ix.reshape(bs,seq_len)         # ix: B x Sequence_length
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        x = ix
        assert x.shape[1] == self.autoregressive_steps

    

        # Generate Feature Grid From Codebook
        feature_grid_shape = (bs, self.reso,self.reso,self.reso,self.codebook_dim)  # B x reso x reso x reso x C
        c_quantize = self.vqvae_model.quantizer.get_codebook_entry(x, feature_grid_shape)    # B x C x reso x reso x reso
        return c_quantize   # B x C x reso x reso x reso


    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out
                

            









                



        
        


        
        

