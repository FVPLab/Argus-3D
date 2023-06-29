import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from src.encoder import encoder_dict
from src.conv_onet import models as VQVAE
from torch.nn import CrossEntropyLoss
# from .fast_transformer_builder_uncond_joint import FAST_transformer_builder_uncond_joint
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image

class GET_FAST_Transformer_Joint(nn.Module):
    """
        Get Fast Transformer Stage 2 Model
        Args:
            cfg, device, dataset
    """
    def __init__(self, cfg, device=None, dataset=None):
        super(GET_FAST_Transformer_Joint, self).__init__()

        self._device = device
        self.config = cfg
        self.name = 'FAST_Transformer_joint'
        print("Using Transformer Model: ",self.name)
        self.autoregressive_steps = cfg['model']['stage2_model_kwargs']['sequence_length'] - 1  # 1025 - 1 = 1024
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
        self.transformer_model_xz = FAST_transformer_builder_uncond_joint(cfg).to(device)
        self.transformer_model_xy = FAST_transformer_builder_uncond_joint(cfg).to(device)
        self.transformer_model_yz = FAST_transformer_builder_uncond_joint(cfg).to(device)

        # Fast Transformer Optimizer
        self.optim = optim.Adam([{'params':self.transformer_model_xz.parameters()},
                                 {'params':self.transformer_model_xy.parameters()},
                                 {'params':self.transformer_model_yz.parameters()}
                                ], lr = float(cfg['training']['stage2_lr']))

        for name, params in self.transformer_model_xz.named_parameters():
            print(name,':',params.size())
        
        self.nll_loss_value = CrossEntropyLoss(ignore_index=1025)  # Value Padding Index
        self.nll_loss_coord = CrossEntropyLoss(ignore_index=4096)  # Coordinate Padding Index

    def forward(self):
        raise NotImplementedError
    
    def get_joint_losses(self, data_dict, device):
        """Get 3Plane Joint Losses  (Value,Position) """
        nll_loss = 0

        plane_index_xz = data_dict['quantize.xz_index'].to(device)  # B x M
        plane_index_xy = data_dict['quantize.xy_index'].to(device)  # B x M
        plane_index_yz = data_dict['quantize.yz_index'].to(device)  # B x M

        xz_coord = data_dict['quantize.xz_coord'].to(device)       # B x M    
        xy_coord = data_dict['quantize.xy_coord'].to(device)       # B x M
        yz_coord = data_dict['quantize.yz_coord'].to(device)       # B x M
        
        logits_value_xz, logits_coord_xz = self.transformer_model_xz(plane_index_xz, xz_coord)    
        logits_value_xy, logits_coord_xy = self.transformer_model_xy(plane_index_xy, xy_coord)
        logits_value_yz, logits_coord_yz = self.transformer_model_yz(plane_index_yz, yz_coord)

        _, _, value_dim = logits_value_xz.shape
        _, _, coord_dim = logits_coord_xz.shape

        # Value Loss
        nll_value_xz = self.nll_loss_value(logits_value_xz.reshape(-1,value_dim), plane_index_xz.reshape(-1))
        nll_value_xy = self.nll_loss_value(logits_value_xy.reshape(-1,value_dim), plane_index_xy.reshape(-1))
        nll_value_yz = self.nll_loss_value(logits_value_yz.reshape(-1,value_dim), plane_index_yz.reshape(-1))
        nll_value_loss = nll_value_xz + nll_value_xy + nll_value_yz
    
        # Coord Loss
        nll_coord_xz = self.nll_loss_coord(logits_coord_xz.reshape(-1,coord_dim), xz_coord.reshape(-1))
        nll_coord_xy = self.nll_loss_coord(logits_coord_xy.reshape(-1,coord_dim), xy_coord.reshape(-1))
        nll_coord_yz = self.nll_loss_coord(logits_coord_yz.reshape(-1,coord_dim), yz_coord.reshape(-1))
        nll_coord_loss = nll_coord_xz + nll_coord_xy + nll_coord_yz

        nll_loss = nll_value_loss + nll_coord_loss
        loss_dict = {}
        loss_dict['value_xz'], loss_dict['value_xy'], loss_dict['value_yz'] = nll_value_xz, nll_value_xy, nll_value_yz
        loss_dict['coord_xz'], loss_dict['coord_xy'], loss_dict['coord_yz'] = nll_coord_xz, nll_coord_xy, nll_coord_yz
        loss_dict['value_loss'], loss_dict['coord_loss'] = nll_value_loss, nll_coord_loss
        return nll_loss, loss_dict
        

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
    def batch_predict_auto_plane(self,bs, plane_type,device, temperature=1, top_k=5,greed=False):
        '''
            Input: bs, plane_type
            Output: 
                index_seq: B x 4096
            Use Transformer generate non-empty (value, coordinate) B x M (M<=1024) ---> Truncated 
             ---> Add Empty Index ---> Get B x 4096 index Sequence
        '''
        steps = self.autoregressive_steps   # 1024 
        print('Current plane type: {}, Greed: {}'.format(plane_type,greed))
        x = torch.zeros(bs,steps,dtype=torch.long).to(device)

        if plane_type == "xz":
            transformer_model = self.transformer_model_xz
        elif plane_type == 'xy':
            transformer_model = self.transformer_model_xy
        elif plane_type == 'yz':
            transformer_model = self.transformer_model_yz

        '''Transformer generate (value, coordinate) tuple sequences'''
        value = torch.zeros(bs, steps,dtype=torch.long).to(device)  # B x 1024  Maximum length: 1024
        coord = torch.zeros(bs, steps,dtype=torch.long).to(device)  # B x 1024  Maximum length: 1024
    
        max_coord = coord[:,0].reshape(-1)      # 0
        for i in tqdm(range(steps)):    # Maximum loops 1024  (seq length)  Mask out Index
            logits_value_predict, logits_coord_predict = transformer_model(value[:,:i+1],coord[:,:i+1])   # B x L x C
            logits_value = logits_value_predict[:,i,:] / temperature    # B x C
            logits_coord = logits_coord_predict[:,i,:] / temperature    # B x C
        
            # Mask Coordinate Index --> Make sure monotonically
            logits_coord[:,:max_coord+1] = -np.inf
            if top_k:
                logits_value = self.top_k_logits(logits_value, top_k)
                logits_coord = self.top_k_logits(logits_coord, top_k)
            probs_value = F.softmax(logits_value, dim=-1)
            probs_coord = F.softmax(logits_coord, dim=-1)
            if not greed:
                value_ix = torch.multinomial(probs_value, num_samples=1)
                coord_ix = torch.multinomial(probs_coord, num_samples=1)
            else:
                _, value_ix = torch.topk(probs_value, k=1, dim=-1)
                _, coord_ix = torch.topk(probs_coord, k=1, dim=-1)
            value_ix = value_ix.squeeze(1)
            coord_ix = coord_ix.squeeze(1)
            value[:, i] = value_ix
            coord[:, i] = coord_ix
            end_step = i
            max_coord = coord_ix    # force monotonically
            # Check if Quit:
            if value_ix == 1024 or value_ix==1025 or coord_ix==4096: # If End Token or Padding Token appears then break loops
                break
        
        '''Restore (value, coord) sequences to B x 4096 Sequences'''
        value = value[:,:end_step]  # remove End Token --> Make sure no End token appears in Sequence   B x M (M<=1023)
        coord = coord[:,:end_step]  # remove Pad Token --> Make sure no Pad token appears in Sequence   B x M (M<=1023)
        
        '''
            # xz: Empty Feature ---> 251  Index
            # xy: Empty Feature ---> 501  Index
            # yz: Empty Feature ---> 1020 Index
        '''

        # Haven't Add Mask --> Make sure monotonically
        full_length = self.reso ** 2 # 4096
        if plane_type == 'xz':
            z = torch.full((1,full_length), 251).reshape(-1).to(device)
        elif plane_type == 'xy':
            z = torch.full((1,full_length), 501).reshape(-1).to(device)
        elif plane_type == 'yz':
            z = torch.full((1,full_length), 1020).reshape(-1).to(device)
        
        z[coord] = value   
    
        # vis_z = z.reshape(64,64).cpu().numpy()
        # data = pd.DataFrame(vis_z)
        # data.to_csv("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/joint_output_index/{}.csv".format(plane_type))
        z = z.reshape(1, full_length)   # 1 x 4096 Index Sequence
        
        return z
    
    @torch.no_grad()
    def batch_test_3plane(self,bs,device,plane_type):
        # data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data/02691156/3d6b9ea0f212e93f26360e1e29a956c7.npz")
        data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data/02691156/1d5708929a4ae05842d1180c659735fe.npz")
        
        
        index_xz, index_xy, index_yz = data_dict['xz_index'], data_dict['xy_index'], data_dict['yz_index']  # 1024
        coord_xz, coord_xy, coord_yz = data_dict['xz_coord'], data_dict['xy_coord'], data_dict['yz_coord']  # 1024
        
        xz_end, xy_end, yz_end = int(np.where(index_xz == 1024)[0]), int(np.where(index_xy == 1024)[0]), int(np.where(index_yz == 1024)[0])
        
        index_xz, coord_xz = torch.from_numpy(index_xz[:xz_end]).to(device), torch.from_numpy(coord_xz[:xz_end]).to(device)
        index_xy, coord_xy = torch.from_numpy(index_xy[:xy_end]).to(device), torch.from_numpy(coord_xy[:xy_end]).to(device)
        index_yz, coord_yz = torch.from_numpy(index_yz[:yz_end]).to(device), torch.from_numpy(coord_yz[:yz_end]).to(device)

        # Random Mask:
        # if plane_type == 'xz_random_mask':
        #     mask_pos = torch.bernoulli(torch.full((1, index_xz.shape[0]), 0.5),dtype=torch.long) * 251
            

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
            index_xz = torch.full((1,4096), 251).to(device)
        elif plane_type == 'xy_random':
            index_xy = torch.full((1,4096), 501).to(device)
        elif plane_type == 'yz_random':
            index_yz = torch.full((1,4096), 1020).to(device)
        
        feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}

        # Vis Plane
        # index_xz = ((index_xz.reshape(64,64) != 251) * 255).cpu().numpy().astype(np.int8)
        # index_xy = ((index_xy.reshape(64,64) != 501) * 255).cpu().numpy().astype(np.int8)
        # index_yz = ((index_yz.reshape(64,64) != 1020) * 255).cpu().numpy().astype(np.int8)

        # xz = Image.fromarray(index_xz).convert('L')
        # xy = Image.fromarray(index_xy).convert('L')
        # yz = Image.fromarray(index_yz).convert('L')
        
        # xz.save("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/vis_index_3plane/3plane_index/xz.png")
        # xy.save("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/vis_index_3plane/3plane_index/xy.png")
        # yz.save("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/vis_index_3plane/3plane_index/yz.png")
        # xz = pd.DataFrame(index_xz)
        # xy = pd.DataFrame(index_xy)
        # yz = pd.DataFrame(index_yz)

        # xz.to_csv("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/vis_index_3plane/3plane_index/xz.csv")
        # xy.to_csv("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/vis_index_3plane/3plane_index/xy.csv")
        # yz.to_csv("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/vis_index_3plane/3plane_index/yz.csv")
        return fea_dict
    

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
        partial_xz_index, partial_xz_coord = index_xz, coord_xz
        partial_xy_index, partial_xy_coord = index_xy, coord_xy
        partial_yz_index, partial_yz_coord = index_yz, coord_yz

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
            index_xz = self.batch_predict_auto_plane_partial(bs,partial_xz_index,partial_xz_coord,plane_type='xz',device=device)  # B x seq_len
        elif plane_type == 'xy':
            index_xy = self.batch_predict_auto_plane_partial(bs,partial_xy_index,partial_xy_coord,plane_type='xy',device=device)  # B x seq_len
        elif plane_type == 'yz':
            index_yz = self.batch_predict_auto_plane_partial(bs,partial_yz_index,partial_yz_coord,plane_type='yz',device=device)  # B x seq_len


        feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return fea_dict

    @torch.no_grad()
    def batch_predict_auto_plane_partial(self,bs,index_partial,coord_partial, plane_type,device, temperature=1, top_k=5,greed=False):
        '''
            Input: bs, plane_type
            Output: 
                index_seq: B x 4096
            Use Transformer generate non-empty (value, coordinate) B x M (M<=1024) ---> Truncated 
             ---> Add Empty Index ---> Get B x 4096 index Sequence
        '''
        steps = self.autoregressive_steps
        print('Current plane type: {}, Greed: {}'.format(plane_type,greed))
        x = torch.zeros(bs,steps,dtype=torch.long).to(device)

        if plane_type == "xz":
            transformer_model = self.transformer_model_xz
        elif plane_type == 'xy':
            transformer_model = self.transformer_model_xy
        elif plane_type == 'yz':
            transformer_model = self.transformer_model_yz

        '''Transformer generate (value, coordinate) tuple sequences'''
        value = torch.zeros(bs, steps,dtype=torch.long).to(device)  # B x 1024  Maximum length: 1024
        coord = torch.zeros(bs, steps,dtype=torch.long).to(device)  # B x 1024  Maximum length: 1024
        
        # Partial GT Input
        begin_pos = index_partial.shape[0]
        # print('begin_pos',begin_pos)
        value[:,:begin_pos] = index_partial
        coord[:,:begin_pos] = coord_partial
        # print('value',value)
        # print('coord',coord)
        # print('value_begin', value[:,begin_pos:begin_pos+5],value[:,:begin_pos+1])
        max_coord = coord[:,begin_pos].reshape(-1)      # 0
        for i in tqdm(range(begin_pos,steps)):    # Maximum loops 1024  (seq length)  Mask out Index
            logits_value_predict, logits_coord_predict = transformer_model(value[:,:i+1],coord[:,:i+1])   # B x L x C
            logits_value = logits_value_predict[:,i,:] / temperature    # B x C
            logits_coord = logits_coord_predict[:,i,:] / temperature    # B x C
        
            # Mask Coordinate Index --> Make sure monotonically
            logits_coord[:,:max_coord+1] = -np.inf
            if top_k:
                logits_value = self.top_k_logits(logits_value, top_k)
                logits_coord = self.top_k_logits(logits_coord, top_k)
            probs_value = F.softmax(logits_value, dim=-1)
            probs_coord = F.softmax(logits_coord, dim=-1)
            if not greed:
                value_ix = torch.multinomial(probs_value, num_samples=1)
                coord_ix = torch.multinomial(probs_coord, num_samples=1)
            else:
                _, value_ix = torch.topk(probs_value, k=1, dim=-1)
                _, coord_ix = torch.topk(probs_coord, k=1, dim=-1)
            value_ix = value_ix.squeeze(1)
            coord_ix = coord_ix.squeeze(1)
            value[:, i] = value_ix
            coord[:, i] = coord_ix
            end_step = i
            max_coord = coord_ix    # force monotonically
            # Check if Quit:
            if value_ix == 1024 or value_ix==1025 or coord_ix==4096: # If End Token or Padding Token appears then break loops
                break
        
        '''Restore (value, coord) sequences to B x 4096 Sequences'''
        value = value[:,:end_step]  # remove End Token --> Make sure no End token appears in Sequence   B x M (M<=1023)
        coord = coord[:,:end_step]  # remove Pad Token --> Make sure no Pad token appears in Sequence   B x M (M<=1023)
        
        '''
            # xz: Empty Feature ---> 251  Index
            # xy: Empty Feature ---> 501  Index
            # yz: Empty Feature ---> 1020 Index
        '''

        # Haven't Add Mask --> Make sure monotonically
        full_length = self.reso ** 2 # 4096
        if plane_type == 'xz':
            z = torch.full((1,full_length), 251).reshape(-1).to(device)
        elif plane_type == 'xy':
            z = torch.full((1,full_length), 501).reshape(-1).to(device)
        elif plane_type == 'yz':
            z = torch.full((1,full_length), 1020).reshape(-1).to(device)
        
        z[coord] = value   
    
        # vis_z = z.reshape(64,64).cpu().numpy()
        # data = pd.DataFrame(vis_z)
        # data.to_csv("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/joint_output_index/{}.csv".format(plane_type))
        z = z.reshape(1, full_length)   # 1 x 4096 Index Sequence
        
        return z


            

    
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
    def batch_test(self,bs,device):
        data_dict = np.load("/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data/02691156/1a04e3eab45ca15dd86060f189eb133.npz")
        index_xz, index_xy, index_yz = data_dict['xz_index'], data_dict['xy_index'], data_dict['yz_index']  # 1024
        coord_xz, coord_xy, coord_yz = data_dict['xz_coord'], data_dict['xy_coord'], data_dict['yz_coord']  # 1024

        # test
        index_xz, coord_xz = torch.from_numpy(index_xz[:1023]).to(device), torch.from_numpy(coord_xz[:1023]).to(device)
        index_xy, coord_xy = torch.from_numpy(index_xy[:571]).to(device), torch.from_numpy(coord_xy[:571]).to(device)
        index_yz, coord_yz = torch.from_numpy(index_yz[:665]).to(device), torch.from_numpy(coord_yz[:665]).to(device)

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
                

            









                



        
        


        
        

