import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from src.encoder import encoder_dict
from src.conv_onet import models as VQVAE
from torch.nn import CrossEntropyLoss
# from src.losses import SigmoidFocalClassificationLoss
from .fast_transformer_builder_uncond import FAST_transformer_builder_uncond
import math
from tqdm import tqdm
import pandas as pd
import numpy as np

class GET_FAST_Transformer(nn.Module):
    """
        Get Fast Transformer Stage 2 Model
        Args:
            cfg, device, dataset
    """
    def __init__(self, cfg, device=None, dataset=None):
        super(GET_FAST_Transformer, self).__init__()

        self._device = device
        self.config = cfg
        self.name = 'FAST_Transformer'
        self.nll_loss = CrossEntropyLoss()
        self.autoregressive_steps = cfg['model']['stage2_model_kwargs']['sequence_length'] - 1
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
        quantizer = VQVAE.quantizer_dict[quantizer](cfg, n_e=embed_num,e_dim=embed_dim,beta=beta,c_dim=c_dim,unet_kwargs=unet2d_kwargs)

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
        self.transformer_model_xz = FAST_transformer_builder_uncond(cfg).to(device)
        self.transformer_model_xy = FAST_transformer_builder_uncond(cfg).to(device)
        self.transformer_model_yz = FAST_transformer_builder_uncond(cfg).to(device)

        # # Revise 2.22   Ignore Blank Index Case
        # print('Using Ignore Index')
        # self.nll_loss_xz = nn.CrossEntropyLoss(ignore_index=251)    # 110000 codebook
        # self.nll_loss_xy = nn.CrossEntropyLoss(ignore_index=1020)
        # self.nll_loss_yz = nn.CrossEntropyLoss(ignore_index=1020)
        self.mask_index_xz = 251
        self.mask_index_xy = 501
        self.mask_index_yz = 1020
        self.nll_loss_xz = nn.CrossEntropyLoss(ignore_index=self.mask_index_xz)
        self.nll_loss_xy = nn.CrossEntropyLoss(ignore_index=self.mask_index_xy)
        self.nll_loss_yz = nn.CrossEntropyLoss(ignore_index=self.mask_index_yz)


        # Focal Loss    # revise 2.25
        # self.focal_loss = SigmoidFocalClassificationLoss()

        print('Using Weight Index')
        self.nll_loss_weight = nn.CrossEntropyLoss(reduction='none')


        # Fast Transformer Optimizer
        # named_params = self.transformer_model.parameters()
        # self.optim = optim.Adam(self.transformer_model.parameters(), lr=float(cfg['training']['stage2_lr']))    # default: 1e-4
        self.optim = optim.Adam([{'params':self.transformer_model_xz.parameters()},
                                 {'params':self.transformer_model_xy.parameters()},
                                 {'params':self.transformer_model_yz.parameters()}
                                ], lr = float(cfg['training']['stage2_lr']))

        for name, params in self.transformer_model_xz.named_parameters():
            print(name,':',params.size())

    def forward(self):
        raise NotImplementedError
    

    def get_losses(self, info_dict):
        '''Calculate Losses  3 Plane'''
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
        nll_loss_xz = self.nll_loss(logits_xz.reshape(-1,c), tgt_plane_index_xz.reshape(-1))    # revise 2.23
        nll_loss_xy = self.nll_loss(logits_xy.reshape(-1,c), tgt_plane_index_xy.reshape(-1))
        nll_loss_yz = self.nll_loss(logits_yz.reshape(-1,c), tgt_plane_index_yz.reshape(-1))
        

        nll_loss = nll_loss_xz + nll_loss_xy + nll_loss_yz
        return nll_loss, nll_loss_xz, nll_loss_xy, nll_loss_yz
    
    def get_ignore_losses(self, info_dict):
        '''Calculate Losses  3 Plane Ignore'''
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
        nll_loss_xz = self.nll_loss_xz(logits_xz.reshape(-1,c), tgt_plane_index_xz.reshape(-1))    # revise 2.23
        nll_loss_xy = self.nll_loss_xy(logits_xy.reshape(-1,c), tgt_plane_index_xy.reshape(-1))
        nll_loss_yz = self.nll_loss_yz(logits_yz.reshape(-1,c), tgt_plane_index_yz.reshape(-1))
        

        nll_loss = nll_loss_xz + nll_loss_xy + nll_loss_yz
        return nll_loss, nll_loss_xz, nll_loss_xy, nll_loss_yz
    
    def get_focal_losses(self, info_dict):
        '''Get Focal Loss 3 Plane'''
        nll_loss = 0
        tgt_plane_index_xz = info_dict['xz'][1]     # B x (reso x reso)
        tgt_plane_index_xy = info_dict['xy'][1]     # B x (reso x reso)
        tgt_plane_index_yz = info_dict['yz'][1]     # B x (reso x reso)

     
        logits_xz = self.transformer_model_xz(tgt_plane_index_xz[:,:-1])    
        logits_xy = self.transformer_model_xy(tgt_plane_index_xy[:,:-1])
        logits_yz = self.transformer_model_yz(tgt_plane_index_yz[:,:-1])
        
        _, seq_len, c = logits_xz.shape
        tgt_plane_index_xz = F.one_hot(tgt_plane_index_xz, num_classes=self.embed_num).squeeze(1)   # B x (resoxreso) x 1024
        tgt_plane_index_xy = F.one_hot(tgt_plane_index_xy, num_classes=self.embed_num).squeeze(1)   # B x (resoxreso) x 1024
        tgt_plane_index_yz = F.one_hot(tgt_plane_index_yz, num_classes=self.embed_num).squeeze(1)   # B x (resoxreso) x 1024

        nll_loss_xz = self.focal_loss(logits_xz.reshape(-1,c), tgt_plane_index_xz.reshape(-1,c))    # revise 2.23
        nll_loss_xy = self.focal_loss(logits_xy.reshape(-1,c), tgt_plane_index_xy.reshape(-1,c))
        nll_loss_yz = self.focal_loss(logits_yz.reshape(-1,c), tgt_plane_index_yz.reshape(-1,c))
        

        nll_loss = nll_loss_xz + nll_loss_xy + nll_loss_yz
        return nll_loss, nll_loss_xz, nll_loss_xy, nll_loss_yz
    

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
        mask_xy = ((tgt_plane_index_xy == self.mask_index_xy) * 1.0).reshape(-1)      # index_xy = 1020   ---> 501
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
    
    @torch.no_grad()
    def Stage1_quantize_get_non_empty_index(self, inputs):
        '''
            Stage1 Quantize Without Unet ---> Get index
        '''
        _, _, info_dict = self.vqvae_model.quantize_get_index(inputs)

        mask_xz = self.get_empty_mask(info_dict['xz'][1], plane_type='xz')  # B x L
        mask_xy = self.get_empty_mask(info_dict['xy'][1], plane_type='xy')  # B x L
        mask_yz = self.get_empty_mask(info_dict['yz'][1], plane_type='yz')  # B x L

        info_dict['mask_xz'] = mask_xz
        info_dict['mask_xy'] = mask_xy
        info_dict['mask_yz'] = mask_yz
        return _, _, info_dict
    
    @torch.no_grad()
    def Stage1_quantize_get_non_empty_index_auto(self, inputs, zero_dict):
        '''
            Stage1 Quantize Without Unet ---> Get index
        '''
        _, _, info_dict = self.vqvae_model.quantize_get_index(inputs)
        
        mask_xz = self.get_empty_mask_auto(info_dict['xz'][1], plane_type='xz',zero_dict=zero_dict)  # B x L
        mask_xy = self.get_empty_mask_auto(info_dict['xy'][1], plane_type='xy',zero_dict=zero_dict)  # B x L
        mask_yz = self.get_empty_mask_auto(info_dict['yz'][1], plane_type='yz',zero_dict=zero_dict)  # B x L

        info_dict['mask_xz'] = mask_xz
        info_dict['mask_xy'] = mask_xy
        info_dict['mask_yz'] = mask_yz
        return _, _, info_dict
    
    @torch.no_grad()
    def Stage1_get_zero_index(self):
        '''Get Zero Index'''
        zero_index_dict = self.vqvae_model.get_zero_index()
        return zero_index_dict

    @torch.no_grad()
    def Stage1_quantize_get_non_empty_index_quantize_single(self, inputs):
        '''
            Stage1 Quantize Without Unet ---> Get index
        '''
        _, _, info_dict = self.vqvae_model.quantize_get_index(inputs)

        mask_xz = self.get_empty_mask_single(info_dict['xz'][1])  # B x L
        mask_xy = self.get_empty_mask_single(info_dict['xy'][1])  # B x L
        mask_yz = self.get_empty_mask_single(info_dict['yz'][1])  # B x L

        info_dict['mask_xz'] = mask_xz
        info_dict['mask_xy'] = mask_xy
        info_dict['mask_yz'] = mask_yz
        return _, _, info_dict

    @torch.no_grad()
    def get_empty_mask_single(self, index):
        '''
            Empty Mask Index Single ---> Single Quantize Empty Index
        '''
        mask = (index == 888)  # B x L: True /home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/out/pointcloud/shapenet_3plane_quantize_64_single/model_120000.pt  Empty Index
        return mask


    @torch.no_grad()
    def get_empty_mask(self, index, plane_type):
        '''
            Empty Mask Index ---> 
        '''
        if plane_type == 'xz':
            mask = (index == self.mask_index_xz)  # B x L: True
        elif plane_type == 'xy':
            mask = (index == self.mask_index_xy)  # B x L: True
        elif plane_type == 'yz':
            mask = (index == self.mask_index_yz)  # B x L: True
        return mask
    
    @torch.no_grad()
    def get_empty_mask_auto(self, index, plane_type, zero_dict):
        '''
            Empty Mask Index ---> 
        '''
        if plane_type == 'xz':
            mask = (index == zero_dict['xz_zero_index'])  # B x L: True
        elif plane_type == 'xy':
            mask = (index == zero_dict['xy_zero_index'])  # B x L: True
        elif plane_type == 'yz':
            mask = (index == zero_dict['yz_zero_index'])  # B x L: True
        return mask
        
    
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
    def batch_predict_auto_plane(self,bs, plane_type,device, temperature=1.0, top_k=10,greed=False):
        steps = self.autoregressive_steps
        print('Current plane type: {}, Greed: {} topk: {}'.format(plane_type,greed,top_k))
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
      
        return x
    
    @torch.no_grad()
    def batch_test(self,bs,device):
        # data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data/02691156/1a04e3eab45ca15dd86060f189eb133.npz")
        data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data/test_data/3d6b9ea0f212e93f26360e1e29a956c7_seq1024.npz")
        
        
        index_xz, index_xy, index_yz = data_dict['xz_index'], data_dict['xy_index'], data_dict['yz_index']  # 1024
        coord_xz, coord_xy, coord_yz = data_dict['xz_coord'], data_dict['xy_coord'], data_dict['yz_coord']  # 1024

        # test
        index_xz, coord_xz = torch.from_numpy(index_xz[:914]).to(device), torch.from_numpy(coord_xz[:914]).to(device)
        index_xy, coord_xy = torch.from_numpy(index_xy[:515]).to(device), torch.from_numpy(coord_xy[:515]).to(device)
        index_yz, coord_yz = torch.from_numpy(index_yz[:349]).to(device), torch.from_numpy(coord_yz[:349]).to(device)
        # index_xz, coord_xz = torch.from_numpy(index_xz).to(device), torch.from_numpy(coord_xz).to(device)
        # index_xy, coord_xy = torch.from_numpy(index_xy).to(device), torch.from_numpy(coord_xy).to(device)
        # index_yz, coord_yz = torch.from_numpy(index_yz).to(device), torch.from_numpy(coord_yz).to(device)

        xz =  torch.full((1,4096), 251).reshape(-1).to(device)
        xy =  torch.full((1,4096), 501).reshape(-1).to(device)
        yz =  torch.full((1,4096), 1020).reshape(-1).to(device)

        xz[coord_xz] = index_xz
        xy[coord_xy] = index_xy
        yz[coord_yz] = index_yz

        index_xz = xz.reshape(1,4096)
        index_xy = xy.reshape(1,4096)
        index_yz = yz.reshape(1,4096)
        

        feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return fea_dict
    

    @torch.no_grad()
    def batch_test_32_index(self, bs, device):
        data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data_32_plane_reduce4_revise_highres256_noisy_singlecodebook_6144/02691156/1a9b552befd6306cc8f2d5fe7449af61.npz")
        index_xz ,index_xy, index_yz = torch.from_numpy(data_dict['xz_index']), torch.from_numpy(data_dict['xy_index']), torch.from_numpy(data_dict['yz_index'])
        
        index_xz = index_xz.reshape(1,1024).to(device)
        index_xy = index_xy.reshape(1,1024).to(device)
        index_yz = index_yz.reshape(1,1024).to(device)

        feature_plane_shape = (bs, 32, 32, 4)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return fea_dict
    
    @torch.no_grad()
    def batch_test_64_index(self, bs, device):
        data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data_64_plane_index/03001627/11040f463a3895019fb4103277a6b93.npz")
        index_xz ,index_xy, index_yz = torch.from_numpy(data_dict['xz_index']), torch.from_numpy(data_dict['xy_index']), torch.from_numpy(data_dict['yz_index'])
        
        index_xz = index_xz.reshape(1,4096).to(device)
        index_xy = index_xy.reshape(1,4096).to(device)
        index_yz = index_yz.reshape(1,4096).to(device)

        feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return fea_dict
        

    @torch.no_grad()
    def batch_test_3plane(self,bs,device,plane_type,random=False):
        # data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data/02691156/3d6b9ea0f212e93f26360e1e29a956c7.npz")
        # data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data_all/02828884/229d510bace435811572ee5ddf1b55b.npz")
        data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data_multi_80/02691156/1ac29674746a0fc6b87697d3904b168b.npz")
        index_xz, index_xy, index_yz = data_dict['xz_index'], data_dict['xy_index'], data_dict['yz_index']  # 1024
        coord_xz, coord_xy, coord_yz = data_dict['xz_coord'], data_dict['xy_coord'], data_dict['yz_coord']  # 1024
        
        xz_end, xy_end, yz_end = int(np.where(index_xz == 1024)[0]), int(np.where(index_xy == 1024)[0]), int(np.where(index_yz == 1024)[0])
        
        index_xz, coord_xz = torch.from_numpy(index_xz[:xz_end]).to(device), torch.from_numpy(coord_xz[:xz_end]).to(device)
        index_xy, coord_xy = torch.from_numpy(index_xy[:xy_end]).to(device), torch.from_numpy(coord_xy[:xy_end]).to(device)
        index_yz, coord_yz = torch.from_numpy(index_yz[:yz_end]).to(device), torch.from_numpy(coord_yz[:yz_end]).to(device)

        # xz =  torch.full((1,4096), 251).reshape(-1).to(device)
        # xy =  torch.full((1,4096), 501).reshape(-1).to(device)
        # yz =  torch.full((1,4096), 1020).reshape(-1).to(device)

        xz =  torch.full((1,4096), 264).reshape(-1).to(device)  # New dataset 
        xy =  torch.full((1,4096), 535).reshape(-1).to(device)
        yz =  torch.full((1,4096), 337).reshape(-1).to(device)

        xz[coord_xz] = index_xz     #
        xy[coord_xy] = index_xy
        yz[coord_yz] = index_yz

        index_xz = xz.reshape(1,4096)       # GT
        index_xy = xy.reshape(1,4096)       # GT
        index_yz = yz.reshape(1,4096)       # GT

        # Predict
        print('Generating Plane {}, Fixing other Plane'.format(plane_type))

        if plane_type == 'xz':
            index_xz = self.batch_predict_auto_plane(bs,plane_type='xz',device=device)  # B x seq_len
        elif plane_type == 'xy':
            index_xy = self.batch_predict_auto_plane(bs,plane_type='xy',device=device)  # B x seq_len
        elif plane_type == 'yz':
            index_yz = self.batch_predict_auto_plane(bs,plane_type='yz',device=device)  # B x seq_len
        
        if plane_type == 'xz_fix':
            index_xy = self.batch_predict_auto_plane(bs,plane_type='xy',device=device)  # B x seq_len
            index_yz = self.batch_predict_auto_plane(bs,plane_type='yz',device=device)  # B x seq_len
        elif plane_type == 'xy_fix':
            index_xz = self.batch_predict_auto_plane(bs,plane_type='xz',device=device)  # B x seq_len
            index_yz = self.batch_predict_auto_plane(bs,plane_type='yz',device=device)  # B x seq_len
        elif plane_type == 'yz_fix':
            index_xz = self.batch_predict_auto_plane(bs,plane_type='xz',device=device)  # B x seq_len
            index_xy = self.batch_predict_auto_plane(bs,plane_type='xy',device=device)  # B x seq_len

        print(plane_type)
        if plane_type == 'xz_random':
            index_xz = torch.randint(0,1024,(1, 4096)).to(device)
        elif plane_type == 'xy_random':
            index_xy = torch.randint(0,1024,(1, 4096)).to(device)
        elif plane_type == 'yz_random':
            index_yz = torch.randint(0,1024,(1, 4096)).to(device)

        feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return fea_dict
    

    
    @torch.no_grad()
    def batch_predict_autoregressive(self,bs,device):
        '''
            Autoregressive generate latent Feature Grid c
        '''
        index_xz = self.batch_predict_auto_plane(bs,plane_type='xz',device=device)  # B x seq_len
        index_xy = self.batch_predict_auto_plane(bs,plane_type='xy',device=device)  # B x seq_len
        index_yz = self.batch_predict_auto_plane(bs,plane_type='yz',device=device)  # B x seq_len
        
        # Generate Feature Grid From Codebook
        feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return fea_dict   # B x C x reso x reso x reso


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
    


    @torch.no_grad()
    def batch_test_3plane_partial(self,bs,device,plane_type):
        data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data/02691156/3d6b9ea0f212e93f26360e1e29a956c7.npz")
        # data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data/02691156/1d5708929a4ae05842d1180c659735fe.npz")
        index_xz, index_xy, index_yz = data_dict['xz_index'], data_dict['xy_index'], data_dict['yz_index']  # 1024
        coord_xz, coord_xy, coord_yz = data_dict['xz_coord'], data_dict['xy_coord'], data_dict['yz_coord']  # 1024
        
        xz_end, xy_end, yz_end = int(np.where(index_xz == 1024)[0]), int(np.where(index_xy == 1024)[0]), int(np.where(index_yz == 1024)[0])
        
        xz_len, xy_len, yz_len = 1, 1, 1
        if plane_type == 'xz':
            xz_len = 2
        elif plane_type == 'xy':
            xy_len = 2
        elif plane_type == 'yz':
            yz_len = 2

        index_xz, coord_xz = torch.from_numpy(index_xz[:xz_end//xz_len]).to(device), torch.from_numpy(coord_xz[:xz_end//xz_len]).to(device)
        index_xy, coord_xy = torch.from_numpy(index_xy[:xy_end//xy_len]).to(device), torch.from_numpy(coord_xy[:xy_end//xy_len]).to(device)
        index_yz, coord_yz = torch.from_numpy(index_yz[:yz_end//yz_len]).to(device), torch.from_numpy(coord_yz[:yz_end//yz_len]).to(device)

        # Partial Index
        partial_xz_index, partial_xz_coord = index_xz, coord_xz  # 1 x M
        partial_xy_index, partial_xy_coord = index_xy, coord_xy  # 1 x M
        partial_yz_index, partial_yz_coord = index_yz, coord_yz  # 1 x M

        xz_pos = int(coord_xz[-1].cpu().numpy())
        xy_pos = int(coord_xy[-1].cpu().numpy())
        yz_pos = int(coord_yz[-1].cpu().numpy())

    
        xz =  torch.full((1,4096), 251).reshape(-1).to(device)
        xy =  torch.full((1,4096), 501).reshape(-1).to(device)
        yz =  torch.full((1,4096), 1020).reshape(-1).to(device)

        xz[coord_xz] = index_xz     #
        xy[coord_xy] = index_xy
        yz[coord_yz] = index_yz

        index_xz = xz.reshape(1,4096)       # GT
        index_xy = xy.reshape(1,4096)       # GT
        index_yz = yz.reshape(1,4096)       # GT

        # Predict
        print('Generating Plane {}, Fixing other Plane'.format(plane_type))

        if plane_type == 'xz':
            index_xz = self.batch_predict_auto_plane_partial(bs,index_xz,xz_pos,plane_type='xz',device=device)  # B x seq_len
        elif plane_type == 'xy':
            index_xy = self.batch_predict_auto_plane_partial(bs,index_xy,xy_pos,plane_type='xy',device=device)  # B x seq_len
        elif plane_type == 'yz':
            index_yz = self.batch_predict_auto_plane_partial(bs,index_yz,yz_pos,plane_type='yz',device=device)  # B x seq_len


        feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return fea_dict

    @torch.no_grad()
    def batch_predict_auto_plane_partial(self,bs,index,pos,plane_type,device, temperature=1, top_k=5,greed=False):
        '''
            Input: bs, plane_type
            Output: 
                index_seq: B x 4096
            Use Transformer generate non-empty (value, coordinate) B x M (M<=1024) ---> Truncated 
             ---> Add Empty Index ---> Get B x 4096 index Sequence
        '''
        steps = self.autoregressive_steps
        print('Current plane type: {}, Greed: {} topk: {}'.format(plane_type,greed,top_k))
        x = torch.zeros(bs,steps,dtype=torch.long).to(device)
        print('pos',pos)
        x[:,:pos] = index[:,:pos]
        if plane_type == "xz":
            transformer_model = self.transformer_model_xz
        elif plane_type == 'xy':
            transformer_model = self.transformer_model_xy
        elif plane_type == 'yz':
            transformer_model = self.transformer_model_yz

        for i in tqdm(range(pos, steps-1)):  
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
      
        return x




    @torch.no_grad()
    def batch_test_3plane_single(self,bs,device):
        # data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data/02691156/3d6b9ea0f212e93f26360e1e29a956c7.npz")
        print('Testing Single Quantize Dataset')
        data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data_single/02691156/1a6ad7a24bb89733f412783097373bdc.npz")
        
        index_xz, index_xy, index_yz = data_dict['xz_index'], data_dict['xy_index'], data_dict['yz_index']  # 1024
        coord_xz, coord_xy, coord_yz = data_dict['xz_coord'], data_dict['xy_coord'], data_dict['yz_coord']  # 1024
        
        xz_end, xy_end, yz_end = int(np.where(index_xz == 1024)[0]), int(np.where(index_xy == 1024)[0]), int(np.where(index_yz == 1024)[0])
        
        index_xz, coord_xz = torch.from_numpy(index_xz[:xz_end]).to(device), torch.from_numpy(coord_xz[:xz_end]).to(device)
        index_xy, coord_xy = torch.from_numpy(index_xy[:xy_end]).to(device), torch.from_numpy(coord_xy[:xy_end]).to(device)
        index_yz, coord_yz = torch.from_numpy(index_yz[:yz_end]).to(device), torch.from_numpy(coord_yz[:yz_end]).to(device)

        xz =  torch.full((1,4096), 888).reshape(-1).to(device)
        xy =  torch.full((1,4096), 888).reshape(-1).to(device)
        yz =  torch.full((1,4096), 888).reshape(-1).to(device)

        xz[coord_xz] = index_xz     #
        xy[coord_xy] = index_xy
        yz[coord_yz] = index_yz

        index_xz = xz.reshape(1,4096)       # GT
        index_xy = xy.reshape(1,4096)       # GT
        index_yz = yz.reshape(1,4096)       # GT

        feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape)    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape)    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape)    # B x C x reso x reso x reso
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return fea_dict

    
                

            









                



        
        


        
        

