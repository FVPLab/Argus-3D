import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from src.encoder import encoder_dict
from src.conv_onet import models as VQVAE
from torch.nn import CrossEntropyLoss
# from .fast_transformer_builder_uncond_baseline_reso32_cond import FAST_transformer_builder_uncond_baseline_reso32_cond
# from .fast_transformer_builder_uncond_baseline_reso32_cond_nodrop import FAST_transformer_builder_uncond_baseline_reso32_cond_nodrop
# from .fast_transformer_builder_uncond_baseline_reso32_cond_nodrop_rate import FAST_transformer_builder_uncond_baseline_reso32_cond_nodrop_rate
from .fast_transformer_builder_uncond_baseline_reso32_cond_quant_single_cond_class import FAST_transformer_builder_uncond_baseline_reso32_cond_quant_single_cond_class
import math
from tqdm import tqdm
import pandas as pd
import numpy as np



class GET_FAST_Transformer_Joint_baseline_reso32_cond_quant_single_cond_class(nn.Module):
    """
        Get Fast Transformer Stage 2 Model
        Args:
            cfg, device, dataset
    """
    def __init__(self, cfg, device=None, dataset=None):
        super(GET_FAST_Transformer_Joint_baseline_reso32_cond_quant_single_cond_class, self).__init__()

        self._device = device
        self.config = cfg
        self.name = 'FAST_Transformer_joint Baseline Quant Single !!'
        print("Using Transformer Model: ",self.name)
        self.autoregressive_steps = cfg['model']['stage2_model_kwargs']['sequence_length'] - 1  # 1025 - 1 = 1024
        self.reso = cfg['model']['encoder_kwargs']['plane_resolution']  # 64
        self.stage1_reduce_dim = cfg['model']['quantizer_kwargs']['reduce_dim']

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
        quantizer = VQVAE.quantizer_dict[quantizer](cfg, n_e=embed_num,e_dim=embed_dim,beta=beta,c_dim=c_dim,unet_kwargs=unet2d_kwargs)

        # VQVAE Decoder
        decoder = VQVAE.decoder_dict[decoder](dim=dim, c_dim=c_dim, padding=padding,**decoder_kwargs).to(device)

        # VQVAE Encoder
        encoder = encoder_dict[encoder](dim=dim, c_dim=c_dim, padding=padding, **encoder_kwargs).to(device)

        self.vqvae_model = VQVAE.ConvolutionalOccupancyNetwork(
            decoder, quantizer, encoder, device=device
        )

        self.vqvae_model.eval()
        # self.restore_from_stage1()      # Restore VQVAE Model

       # Fast Transformer

        self.transformer_model = FAST_transformer_builder_uncond_baseline_reso32_cond_quant_single_cond_class(cfg).to(device)

        self.optim = optim.Adam(self.transformer_model.parameters(), lr=float(cfg['training']['stage2_lr']))

        # for name, params in self.transformer_model.named_parameters():
        #     print(name,':',params.size())
        
        self.nll_loss_value = CrossEntropyLoss()  
        # self.nll_loss_coord = CrossEntropyLoss(ignore_index=4096)  # Coordinate Padding Index

    def forward(self):
        raise NotImplementedError
    

    @torch.no_grad()
    def batch_test_32_index(self, bs, device):
        data_dict = np.load("/home/hongyuxin/icml_3d/shapegrow_simplified/quantize_reso16/03001627/1a6f615e8b1b5ae4dbbc9440457e303e.npz")
        index_xz ,index_xy, index_yz = torch.from_numpy(data_dict['xz_index']), torch.from_numpy(data_dict['xy_index']), torch.from_numpy(data_dict['yz_index'])
        
        reso = 16
        reso_len = reso ** 2

        index_xz = index_xz.reshape(1,reso_len).to(device)
        index_xy = index_xy.reshape(1,reso_len).to(device)
        index_yz = index_yz.reshape(1,reso_len).to(device)

        feature_plane_shape = (bs, reso, reso, 4)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return fea_dict
    

    # Batch Testing 32 Single
    @torch.no_grad()
    def batch_test_32_index_quantize_single(self, bs, device):
        # data_dict = np.load("/home/hongyuxin/icml_3d/shapegrow_simplified/single_quantize_32_400K/03001627/1a6f615e8b1b5ae4dbbc9440457e303e.npz")
        # index_xz ,index_xy, index_yz = torch.from_numpy(data_dict['xz_index']), torch.from_numpy(data_dict['xy_index']), torch.from_numpy(data_dict['yz_index'])
        data_dict = np.load("/home/hongyuxin/icml_3d/shapegrow_simplified/single_quantize_256_600K/02691156/1a04e3eab45ca15dd86060f189eb133.npz")
        index = torch.from_numpy(data_dict['index'])

        reso = 32
        reso_len = reso ** 2

        # single quantize:
        index = index.reshape(1, reso_len).to(device)

        feature_plane_shape = (bs, reso, reso, 4)
        plane_xz, plane_xy, plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index, feature_plane_shape)
        fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return fea_dict


    
    def get_joint_losses(self, data_dict, device):
        """Get 3Plane Joint Losses  (Value,Position) """
        nll_loss = 0

        plane_index = data_dict['quantize.index'].to(device)

        class_idx = data_dict['c_idx'].to(device)

        logits_value = self.transformer_model(plane_index, class_idx)

        _, _, value_dim = logits_value.shape

        # Value Loss:
        # import pdb
        # pdb.set_trace()
        nll_value_loss = self.nll_loss_value(logits_value.reshape(-1, value_dim), plane_index.reshape(-1))
        nll_loss = nll_value_loss
        loss_dict = {}
        loss_dict['value_loss'] = nll_value_loss
        return nll_loss, loss_dict
    

    def get_joint_losses_pc_sample(self, data_dict, device):
        """Get 3Plane Joint Losses  (Value,Position) """
        nll_loss = 0

        # plane_index_xz = data_dict['quantize.xz_index'].to(device)  # B x M
        # plane_index_xy = data_dict['quantize.xy_index'].to(device)  # B x M
        # plane_index_yz = data_dict['quantize.yz_index'].to(device)  # B x M
        pc = data_dict['inputs'].to(device)
        c = self.Stage1_encode_inputs(pc)
        _, _, info_dict = self.Stage1_quantize_get_index(c)

        plane_index_xz = info_dict['xz'][1]
        plane_index_xy = info_dict['xy'][1]
        plane_index_yz = info_dict['yz'][1]
        
        logits_value_xz, logits_value_xy, logits_value_yz = self.transformer_model(plane_index_xz, plane_index_xy, plane_index_yz)    

        _, _, value_dim = logits_value_xz.shape

        # Value Loss
        nll_value_xz = self.nll_loss_value(logits_value_xz.reshape(-1,value_dim), plane_index_xz.reshape(-1))
        nll_value_xy = self.nll_loss_value(logits_value_xy.reshape(-1,value_dim), plane_index_xy.reshape(-1))
        nll_value_yz = self.nll_loss_value(logits_value_yz.reshape(-1,value_dim), plane_index_yz.reshape(-1))
        nll_value_loss = nll_value_xz + nll_value_xy + nll_value_yz

        nll_loss = nll_value_loss 
        loss_dict = {}
        loss_dict['value_xz'], loss_dict['value_xy'], loss_dict['value_yz'] = nll_value_xz, nll_value_xy, nll_value_yz
        loss_dict['value_loss'] = nll_value_loss
        return nll_loss, loss_dict
    
    
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
            print('\n=> Loading Stage1 Model From {}\n'.format(load_path))
            state_dict = torch.load(load_path,map_location='cuda:0')
            self.vqvae_model.load_state_dict(state_dict['model'])
    

    @torch.no_grad()
    def batch_predict_auto_plane(self,bs, plane_type,device, temperature=1, top_k=250,greed=False):
        '''
            Input: bs, plane_type
            Output: 
                index_seq: B x 4096
            Use Transformer generate non-empty (value, coordinate) B x M (M<=1024) ---> Truncated 
             ---> Add Empty Index ---> Get B x 4096 index Sequence
        '''
        steps = self.autoregressive_steps   # 1024 
        print('Current plane type: {}, Greed: {} Topk: {}'.format(plane_type,greed,top_k))
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
    def batch_predict_reso32_cond_get_quantize(self, data, device,top_k=250):
        index_xz = data['quantize.xz_index'].to(device)
        index_xy = data['quantize.xy_index'].to(device)
        index_yz = data['quantize.yz_index'].to(device)
        bs = index_xz.shape[0] 
        feature_plane_shape = (bs, self.reso,self.reso, self.stage1_reduce_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        # fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return  plane_xz, plane_xy, plane_yz
    
    

    @torch.no_grad()
    def batch_predict_reso32(self, bs, device):
        index_xz = self.batch_predict_auto_plane32(bs,plane_type='xz',device=device)
        index_xy = self.batch_predict_auto_plane32(bs,plane_type='xy',device=device)
        index_yz = self.batch_predict_auto_plane32(bs,plane_type='yz',device=device)
        
        feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        # fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return plane_xz, plane_xy, plane_yz
    

    @torch.no_grad()
    def batch_predict_reso32_cond(self, bs, device,top_k=250,greed=False):
        
        index = self.batch_predict_auto_plane32_cond(bs, device=device, top_k=top_k, greed=greed)

        feature_plane_shape = (bs, 32, 32, self.stage1_reduce_dim)

        plane_xz, plane_xy, plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index, feature_plane_shape)
        
        return plane_xz, plane_xy, plane_yz


        # index_xz, index_xy, index_yz = self.batch_predict_auto_plane32_cond(bs,device=device,top_k=top_k,greed=greed)
       
        # feature_plane_shape = (bs, 32, 32, self.stage1_reduce_dim)  # B x reso x reso x reso x C
        # plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        # plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        # plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        # fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        # return  plane_xz, plane_xy, plane_yz
    
    @torch.no_grad()
    def batch_predict_reso32_cond_class(self, bs, device,top_k=250,greed=False, class_idx=None):
        
        index = self.batch_predict_auto_plane32_cond_class(bs, device=device, top_k=top_k, greed=greed, class_idx=class_idx)

        feature_plane_shape = (bs, 32, 32, self.stage1_reduce_dim)

        plane_xz, plane_xy, plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index, feature_plane_shape)
        
        return plane_xz, plane_xy, plane_yz

    @torch.no_grad()
    def batch_predict_auto_plane32_cond_class(self, bs, device, temperature=1, top_k=250,greed=False, class_idx=None):
        steps = 1024
        x = torch.zeros(bs, 1024, dtype=torch.long).to(device)
        for i in tqdm(range(steps)):
            logits = self.transformer_model(x[:,:i+1], class_idx)
            logits = logits[:,i,:] / temperature
            if top_k:
                logits = self.top_k_logits(logits, top_k)
            
            probs_value = F.softmax(logits, dim=-1)
            if not greed:
                ix = torch.multinomial(probs_value, num_samples=1)

            else:
                _, ix = torch.topk(probs_value, k=1, dim=-1)
            ix = ix.squeeze(1)
            x[:, i] = ix
        x = x.reshape(bs, 1024)
        return x


    @torch.no_grad()
    def batch_predict_reso16_cond(self, bs, device,top_k=250,greed=False):
        index_xz, index_xy, index_yz = self.batch_predict_auto_plane16_cond(bs,device=device,top_k=top_k,greed=greed)
        feature_plane_shape = (bs, 16, 16, self.stage1_reduce_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        # fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return  plane_xz, plane_xy, plane_yz
    
    
    @torch.no_grad()
    def batch_predict_auto_plane16_cond(self, bs, device, temperature=1, top_k=250,greed=False):
        steps = 256
        xz = torch.zeros(bs,256,dtype=torch.long).to(device)
        xy = torch.zeros(bs,256,dtype=torch.long).to(device)
        yz = torch.zeros(bs,256,dtype=torch.long).to(device)
        for i in tqdm(range(steps)):
            logits_xz, logits_xy, logits_yz = self.transformer_model(xz[:,:i+1], xy[:,:i+1], yz[:,:i+1])
            logits_xz, logits_xy ,logits_yz = logits_xz[:,i,:] / temperature, logits_xy[:,i,:] / temperature, logits_yz[:,i,:] / temperature
            # print(F.softmax(logits_xz,dim=-1).max(),F.softmax(logits_xy,dim=-1).max(),F.softmax(logits_yz,dim=-1).max())
            if top_k:
                logits_xz = self.top_k_logits(logits_xz, top_k)
                logits_xy = self.top_k_logits(logits_xy, top_k)
                logits_yz = self.top_k_logits(logits_yz, top_k)
            probs_value_xz = F.softmax(logits_xz, dim=-1)
            probs_value_xy = F.softmax(logits_xy, dim=-1)
            probs_value_yz = F.softmax(logits_yz, dim=-1)
            if not greed:
                xz_ix = torch.multinomial(probs_value_xz, num_samples=1)
                xy_ix = torch.multinomial(probs_value_xy, num_samples=1)
                yz_ix = torch.multinomial(probs_value_yz, num_samples=1)
            else:
                _, xz_ix = torch.topk(probs_value_xz, k=1, dim=-1)
                _, xy_ix = torch.topk(probs_value_xy, k=1, dim=-1)
                _, yz_ix = torch.topk(probs_value_yz, k=1, dim=-1)
            xz_ix = xz_ix.squeeze(1)
            xy_ix = xy_ix.squeeze(1)
            yz_ix = yz_ix.squeeze(1)
            xz[:, i] = xz_ix
            xy[:, i] = xy_ix
            yz[:, i] = yz_ix

        xz = xz.reshape(bs,256)
        xy = xy.reshape(bs,256)
        yz = yz.reshape(bs,256)            
        return xz, xy, yz
    

    

    @torch.no_grad()
    def batch_predict_auto_plane32_cond(self, bs, device, temperature=1, top_k=250,greed=False):
        steps = 1024
        x = torch.zeros(bs, 1024, dtype=torch.long).to(device)
        for i in tqdm(range(steps)):
            logits = self.transformer_model(x[:,:i+1])
            logits = logits[:,i,:] / temperature
            if top_k:
                logits = self.top_k_logits(logits, top_k)
            
            probs_value = F.softmax(logits, dim=-1)
            if not greed:
                ix = torch.multinomial(probs_value, num_samples=1)

            else:
                _, ix = torch.topk(probs_value, k=1, dim=-1)
            ix = ix.squeeze(1)
            x[:, i] = ix
        x = x.reshape(bs, 1024)
        return x
    

    @torch.no_grad()
    def batch_predict_reso64_cond(self, bs, device):
        index_xz, index_xy, index_yz = self.batch_predict_auto_plane64_cond(bs,device=device)
       
        feature_plane_shape = (bs, self.reso,self.reso, self.codebook_dim)  # B x reso x reso x reso x C
        plane_xz = self.vqvae_model.quantizer.get_codebook_entry(index_xz, feature_plane_shape, plane_type='xz')    # B x C x reso x reso x reso
        plane_xy = self.vqvae_model.quantizer.get_codebook_entry(index_xy, feature_plane_shape, plane_type='xy')    # B x C x reso x reso x reso
        plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index_yz, feature_plane_shape, plane_type='yz')    # B x C x reso x reso x reso
        # fea_dict = {'xz':plane_xz,'xy':plane_xy,'yz':plane_yz}
        return  plane_xz, plane_xy, plane_yz
    
    @torch.no_grad()
    def batch_predict_auto_plane64_cond(self, bs, device, temperature=1, top_k=250,greed=False):
        steps = 4096
        xz = torch.zeros(bs,steps,dtype=torch.long).to(device)
        xy = torch.zeros(bs,steps,dtype=torch.long).to(device)
        yz = torch.zeros(bs,steps,dtype=torch.long).to(device)
        for i in tqdm(range(steps)):
            logits_xz, logits_xy, logits_yz = self.transformer_model(xz[:,:i+1], xy[:,:i+1], yz[:,:i+1])
            logits_xz, logits_xy ,logits_yz = logits_xz[:,i,:] / temperature, logits_xy[:,i,:] / temperature, logits_yz[:,i,:] / temperature

            if top_k:
                logits_xz = self.top_k_logits(logits_xz, top_k)
                logits_xy = self.top_k_logits(logits_xy, top_k)
                logits_yz = self.top_k_logits(logits_yz, top_k)
            probs_value_xz = F.softmax(logits_xz, dim=-1)
            probs_value_xy = F.softmax(logits_xy, dim=-1)
            probs_value_yz = F.softmax(logits_yz, dim=-1)
            if not greed:
                xz_ix = torch.multinomial(probs_value_xz, num_samples=1)
                xy_ix = torch.multinomial(probs_value_xy, num_samples=1)
                yz_ix = torch.multinomial(probs_value_yz, num_samples=1)
            else:
                _, xz_ix = torch.topk(probs_value_xz, k=1, dim=-1)
                _, xy_ix = torch.topk(probs_value_xy, k=1, dim=-1)
                _, yz_ix = torch.topk(probs_value_yz, k=1, dim=-1)
            xz_ix = xz_ix.squeeze(1)
            xy_ix = xy_ix.squeeze(1)
            yz_ix = yz_ix.squeeze(1)
            xz[:, i] = xz_ix
            xy[:, i] = xy_ix
            yz[:, i] = yz_ix

        xz = xz.reshape(bs,steps)
        xy = xy.reshape(bs,steps)
        yz = yz.reshape(bs,steps)            
        return xz, xy, yz
    
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


    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out
    

                

            









                



        
        


        
        

