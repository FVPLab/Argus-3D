import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    compute_iou, make_3d_grid, add_key,
)
from src.utils import visualize as vis
from src.training import BaseTrainer
import numpy as np

class Stage2_Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False,cfg=None):
        self.model = model      # GET_FAST_Transformer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        # self.output_dataset_path = "/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data_single/02691156"
        # self.output_dataset_path = "/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data_all"
        # self.output_dataset_path = "/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data/test_data"
        # if not os.path.exists(self.output_dataset_path):
        #     os.makedirs(self.output_dataset_path)
        self.truncate_len = 1024
        # self.output_dataset_path = "/home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/stage2_data_32_plane_index_650K"
        # print(cfg)
        try: 
            self.output_dataset_path = cfg['output_dataset_path']
        except:
            self.output_dataset_path = None
        # print('truncate_len:',self.truncate_len)
        print('output_dataset_path', self.output_dataset_path)
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir,exist_ok=True)

    def train_stage2_step(self, data):
        ''' Performs a training step. Stage 2 Model

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        _, _, info_dict = self.model.Stage1_quantize_get_index(c)   # inputs: B x C x reso x reso x reso
        nll_loss, nll_loss_xz, nll_loss_xy, nll_loss_yz = self.model.get_losses(info_dict)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), nll_loss_xz.item(), nll_loss_xy.item(), nll_loss_yz.item()
    
    def train_stage2_step_ignore(self, data):
        ''' Performs a training step. Stage 2 Model Ignore

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        _, _, info_dict = self.model.Stage1_quantize_get_index(c)   # inputs: B x C x reso x reso x reso
        nll_loss, nll_loss_xz, nll_loss_xy, nll_loss_yz = self.model.get_ignore_losses(info_dict)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), nll_loss_xz.item(), nll_loss_xy.item(), nll_loss_yz.item()
    
    def train_stage2_step_single(self, data):
        ''' Performs a training step. Stage 2 Model Single Transformer

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        _, _, info_dict = self.model.Stage1_quantize_get_index(c)   # inputs: B x C x reso x reso x reso
        nll_loss = self.model.get_losses(info_dict)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item()
    
    def train_stage2_focal_step(self, data):
        ''' Performs a training step. Stage 2 Model

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        _, _, info_dict = self.model.Stage1_quantize_get_index(c)   # inputs: B x C x reso x reso x reso
        nll_loss, nll_loss_xz, nll_loss_xy, nll_loss_yz = self.model.get_focal_losses(info_dict)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), nll_loss_xz.item(), nll_loss_xy.item(), nll_loss_yz.item()
    

    def train_stage2_joint_step(self, data):
        ''' Performs a training step. Stage 2 Model  Joint Distribution

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        nll_loss, loss_dict = self.model.get_joint_losses(data,device=self.device)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), loss_dict
    
    def train_stage2_joint_step_reso32(self, data):
        ''' Performs a training step. Stage 2 Model  Joint Distribution

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        nll_loss, loss_dict = self.model.get_joint_losses(data,device=self.device)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), loss_dict

    def train_stage2_joint_step_reso32_amp(self, data):
        ''' Performs a training step. Stage 2 Model  Joint Distribution

        Args:
            data (dict): data dictionary
        '''

        self.model.train()
        nll_loss, loss_dict = self.model.get_joint_losses(data,device=self.device)
        # self.model.backward(nll_loss)                # optimizer
        return nll_loss, loss_dict
    
    def train_stage2_joint_step_reso32_weight(self, data):
        ''' Performs a training step. Stage 2 Model  Joint Distribution

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        nll_loss, loss_dict = self.model.get_joint_weight_losses(data,device=self.device)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), loss_dict


    def train_stage2_joint_step_reso32_pc_sample(self, data):
        ''' Performs a training step. Stage 2 Model  Joint Distribution

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        nll_loss, loss_dict = self.model.get_joint_losses_pc_sample(data,device=self.device)
        # self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), loss_dict
    


    def train_stage2_joint_step_reso32_optim_class_weight(self, pointcloud, weight):
        ''' Performs a training step. Stage 2 Model  Joint Distribution

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        nll_loss, loss_dict = self.model.get_joint_losses_optim_weight(pointcloud, weight, device=self.device)
        return nll_loss, loss_dict
    

    def train_stage2_joint_3plane_step(self, data):
        ''' Performs a training step. Stage 2 Model  Joint Distribution 3Plane

        Args:
            data (dict): data dictionary  
        '''
        self.model.train()
        nll_loss, loss_dict = self.model.get_3plane_losses(data,device=self.device)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), loss_dict
    
    
    def train_stage2_joint_baseline_single_step(self, data):
        ''' Performs a training step. Stage 2 Model  Joint Distribution

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        nll_loss, loss_dict = self.model.get_joint_single_losses(data,device=self.device)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), loss_dict

    def train_stage2_weight_step(self, data):
        ''' Performs a training step. Stage 2 Model Weight

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        _, _, info_dict = self.model.Stage1_quantize_get_index(c)   # inputs: B x C x reso x reso x reso
        nll_loss, nll_loss_xz, nll_loss_xy, nll_loss_yz, xz_bg_loss, xz_fg_loss, xy_bg_loss, xy_fg_loss, yz_bg_loss, yz_fg_loss = self.model.get_weight_losses(info_dict)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), nll_loss_xz.item(), nll_loss_xy.item(), nll_loss_yz.item(), xz_bg_loss.item(), xz_fg_loss.item(), xy_bg_loss.item(), xy_fg_loss.item(), yz_bg_loss.item(), yz_fg_loss.item()

    
    def train_stage2_joint_baseline_step(self, data):
        ''' Performs a training step. Stage 2 Model  Joint Distribution

        Args:
            data (dict): data dictionary  
        '''

        self.model.train()
        nll_loss, loss_dict = self.model.get_joint_losses(data,device=self.device)
        self.model.backward(nll_loss)                # optimizer
        return nll_loss.item(), loss_dict
        

    def generate_dataset(self,data):
        self.model.eval()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        _, _, info_dict = self.model.Stage1_quantize_get_non_empty_index(c)   # inputs: B x C x reso x reso x reso
        
        assert c['xz'].shape[0] == 1 # Batch size = 1
        plane_index_xz = info_dict['xz'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_xy = info_dict['xy'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_yz = info_dict['yz'][1].reshape(-1)     # 1 x (reso x reso)  4096

        # Add End Token Padding & Truncated Sequences
        coord = torch.arange(0, plane_index_xz.shape[0])        # coordinate: 0 - 4095
        xz_mask = ~info_dict['mask_xz'].reshape(-1)  # 1 x 4096          non empty index
        xy_mask = ~info_dict['mask_xy'].reshape(-1)  # 1 x 4096          non empty index
        yz_mask = ~info_dict['mask_yz'].reshape(-1)  # 1 x 4096          non empty index

        non_empty_xz, coord_xz = plane_index_xz[xz_mask], coord[xz_mask]  # L1
        non_empty_xy, coord_xy = plane_index_xy[xy_mask], coord[xy_mask]  # L2
        non_empty_yz, coord_yz = plane_index_yz[yz_mask], coord[yz_mask]  # L3

        data_dict = {}
        xz_dict, xy_dict, yz_dict = {}, {}, {}
        xz_dict['index'], xz_dict['coord'] = non_empty_xz, coord_xz 
        xy_dict['index'], xy_dict['coord'] = non_empty_xy, coord_xy
        yz_dict['index'], yz_dict['coord'] = non_empty_yz, coord_yz
        
        xz_dict = self.align_seq(xz_dict) # unfinished
        xy_dict = self.align_seq(xy_dict) # unfinished
        yz_dict = self.align_seq(yz_dict) # unfinished
        data_dict['xz'], data_dict['xy'], data_dict['yz'] = xz_dict, xy_dict, yz_dict

        # data_dict['xz'] = {'index':non_empty_xz.cpu().numpy(), 'coord':coord_xz.cpu().numpy()}
        # data_dict['xy'] = {'index':non_empty_xy.cpu().numpy(), 'coord':coord_xy.cpu().numpy()}
        # data_dict['yz'] = {'index':non_empty_yz.cpu().numpy(), 'coord':coord_yz.cpu().numpy()}

        model_name = data['model'][0]
        category = data['category'][0]
        # print('model',model_name,'cateogry:',category)
        self.save_process_data(data_dict, category, model_name)
        return True
    
    

    def generate_dataset_quantize_single(self,data):
        self.model.eval()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        _, _, info_dict = self.model.Stage1_quantize_get_non_empty_index_quantize_single(c)   # inputs: B x C x reso x reso x reso
        
        assert c['xz'].shape[0] == 1 # Batch size = 1
        plane_index_xz = info_dict['xz'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_xy = info_dict['xy'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_yz = info_dict['yz'][1].reshape(-1)     # 1 x (reso x reso)  4096

        # Add End Token Padding & Truncated Sequences
        coord = torch.arange(0, plane_index_xz.shape[0])        # coordinate: 0 - 4095
        xz_mask = ~info_dict['mask_xz'].reshape(-1)  # 1 x 4096          non empty index
        xy_mask = ~info_dict['mask_xy'].reshape(-1)  # 1 x 4096          non empty index
        yz_mask = ~info_dict['mask_yz'].reshape(-1)  # 1 x 4096          non empty index

        non_empty_xz, coord_xz = plane_index_xz[xz_mask], coord[xz_mask]  # L1
        non_empty_xy, coord_xy = plane_index_xy[xy_mask], coord[xy_mask]  # L2
        non_empty_yz, coord_yz = plane_index_yz[yz_mask], coord[yz_mask]  # L3

        data_dict = {}
        xz_dict, xy_dict, yz_dict = {}, {}, {}
        xz_dict['index'], xz_dict['coord'] = non_empty_xz, coord_xz 
        xy_dict['index'], xy_dict['coord'] = non_empty_xy, coord_xy
        yz_dict['index'], yz_dict['coord'] = non_empty_yz, coord_yz
        
        xz_dict = self.align_seq(xz_dict) # unfinished
        xy_dict = self.align_seq(xy_dict) # unfinished
        yz_dict = self.align_seq(yz_dict) # unfinished
        data_dict['xz'], data_dict['xy'], data_dict['yz'] = xz_dict, xy_dict, yz_dict

        # data_dict['xz'] = {'index':non_empty_xz.cpu().numpy(), 'coord':coord_xz.cpu().numpy()}
        # data_dict['xy'] = {'index':non_empty_xy.cpu().numpy(), 'coord':coord_xy.cpu().numpy()}
        # data_dict['yz'] = {'index':non_empty_yz.cpu().numpy(), 'coord':coord_yz.cpu().numpy()}

        model_name = data['model'][0]
        category = data['category'][0]
        # print('model',model_name,'cateogry:',category)
        self.save_process_data(data_dict, category,model_name)
        return True
    
    # def generate_dataset_test(self,data):
    #     self.model.eval()
    #     inputs = data.get('inputs').to(self.device)   

    #     # Stage1 Encode ---> Get Quantize Index
    #     c = self.model.Stage1_encode_inputs(inputs)
    #     _, _, info_dict = self.model.Stage1_quantize_get_index(c)   # inputs: B x C x reso x reso x reso
        
    #     xz_index = info_dict['xz'][1].reshape(-1).cpu().numpy()   # 4096
    #     xy_index = info_dict['xy'][1].reshape(-1).cpu().numpy()
    #     yz_index = info_dict['yz'][1].reshape(-1).cpu().numpy()

    #     # data_dict['xz'], data_dict['xy'], data_dict['yz'] = xz_dict, xy_dict, yz_dict

    #     model_name = data['model'][0]
    #     output_path = os.path.join(self.output_dataset_path,model_name+'_direct_final.npz')
    #     np.savez(output_path,xz_index=xz_index,xy_index=xy_index,yz_index=yz_index)
    #     return True
    
    def align_seq(self, plane_dict):
        """ 
            Align Index Sequences & Add Padding & Truncated Sequences
            Change to Numpy Format
        """
        index_seq, coord_seq = plane_dict['index'].cpu(), plane_dict['coord'].cpu()  # L1 , L1
        # Truncated:
        assert index_seq.shape[0] == coord_seq.shape[0]
        if index_seq.shape[0] >= self.truncate_len: # maximum length: 1023   End Token
            index_seq = index_seq[:self.truncate_len-1]
            coord_seq = coord_seq[:self.truncate_len-1]

        seq_len = index_seq.shape[0]    # <= 1023
        # print('seq len',seq_len)
        # Add End Token and Padding Token:
        """
            Special Token:
            1024: Index End Token
            1025: Index Padding Token
            4096: Coord Padding Token
        """
        index_tensor = torch.full([self.truncate_len], 1025)  # 1025: Index Padding Token
        coord_tensor = torch.full([self.truncate_len], 4096)  # 4096: Coord Padding Token
        
        # Add End token First:
        index_tensor[:seq_len] = index_seq  
        index_tensor[seq_len] = 1024         # Index End Token
        coord_tensor[:seq_len] = coord_seq

        plane_dict = {}
        plane_dict['index'] = index_tensor.numpy()
        plane_dict['coord'] = coord_tensor.numpy()
        return plane_dict

    def save_process_data(self, data_dict,category, model_name,):
        '''Save Data Dict'''
        category_path = os.path.join(self.output_dataset_path,category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
        output_path = os.path.join(self.output_dataset_path,category,model_name+'.npz')
        np.savez(output_path,
                xz_index=data_dict['xz']['index'],
                xy_index=data_dict['xy']['index'],
                yz_index=data_dict['yz']['index'],
                xz_coord=data_dict['xz']['coord'],
                xy_coord=data_dict['xy']['coord'],
                yz_coord=data_dict['yz']['coord'])


    def eval_stage2_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        
        batch_size = points.size(0)

        kwargs = {}
        
        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs, 
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict
    
    def get_zero_index(self):
        zero_index_dict = self.model.Stage1_get_zero_index()
        return zero_index_dict
        

    def generate_dataset_auto_zero_index(self,data,zero_dict):
        self.model.eval()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        _, _, info_dict = self.model.Stage1_quantize_get_non_empty_index_auto(c, zero_dict)   # inputs: B x C x reso x reso x reso
        
        assert c['xz'].shape[0] == 1 # Batch size = 1
        plane_index_xz = info_dict['xz'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_xy = info_dict['xy'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_yz = info_dict['yz'][1].reshape(-1)     # 1 x (reso x reso)  4096

        # Add End Token Padding & Truncated Sequences
        coord = torch.arange(0, plane_index_xz.shape[0])        # coordinate: 0 - 4095
        xz_mask = ~info_dict['mask_xz'].reshape(-1)  # 1 x 4096          non empty index
        xy_mask = ~info_dict['mask_xy'].reshape(-1)  # 1 x 4096          non empty index
        yz_mask = ~info_dict['mask_yz'].reshape(-1)  # 1 x 4096          non empty index

        non_empty_xz, coord_xz = plane_index_xz[xz_mask], coord[xz_mask]  # L1
        non_empty_xy, coord_xy = plane_index_xy[xy_mask], coord[xy_mask]  # L2
        non_empty_yz, coord_yz = plane_index_yz[yz_mask], coord[yz_mask]  # L3

        data_dict = {}
        xz_dict, xy_dict, yz_dict = {}, {}, {}
        xz_dict['index'], xz_dict['coord'] = non_empty_xz, coord_xz 
        xy_dict['index'], xy_dict['coord'] = non_empty_xy, coord_xy
        yz_dict['index'], yz_dict['coord'] = non_empty_yz, coord_yz
        
        xz_dict = self.align_seq(xz_dict) # unfinished
        xy_dict = self.align_seq(xy_dict) # unfinished
        yz_dict = self.align_seq(yz_dict) # unfinished
        data_dict['xz'], data_dict['xy'], data_dict['yz'] = xz_dict, xy_dict, yz_dict

        # data_dict['xz'] = {'index':non_empty_xz.cpu().numpy(), 'coord':coord_xz.cpu().numpy()}
        # data_dict['xy'] = {'index':non_empty_xy.cpu().numpy(), 'coord':coord_xy.cpu().numpy()}
        # data_dict['yz'] = {'index':non_empty_yz.cpu().numpy(), 'coord':coord_yz.cpu().numpy()}

        model_name = data['model'][0]
        category = data['category'][0]
        # print('model',model_name,'cateogry:',category)
        self.save_process_data(data_dict, category, model_name)
        return True
    
    def generate_dataset_32_index(self,data):
        self.model.eval()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        # _, _, info_dict = self.model.Stage1_quantize_get_non_empty_index_auto(c, zero_dict)   # inputs: B x C x reso x reso x reso
        _, _, info_dict = self.model.Stage1_quantize_get_index(c)
        
        assert c['xz'].shape[0] == 1 # Batch size = 1
        plane_index_xz = info_dict['xz'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_xy = info_dict['xy'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_yz = info_dict['yz'][1].reshape(-1)     # 1 x (reso x reso)  4096
        data_dict = {}

        data_dict['xz_plane_index'] = plane_index_xz.cpu().numpy()
        data_dict['xy_plane_index'] = plane_index_xy.cpu().numpy()
        data_dict['yz_plane_index'] = plane_index_yz.cpu().numpy()

        model_name = data['model'][0]
        category = data['category'][0]
        # print('model',model_name,'cateogry:',category)
        self.save_process_data_32_index(data_dict, category, model_name)
        return True
    
    def save_process_data_32_index(self, data_dict,category, model_name,):
        '''Save Data Dict without coord / index'''
        category_path = os.path.join(self.output_dataset_path,category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
        output_path = os.path.join(self.output_dataset_path,category,model_name+'.npz')
        np.savez(output_path,
                xz_index=data_dict['xz_plane_index'],
                xy_index=data_dict['xy_plane_index'],
                yz_index=data_dict['yz_plane_index'])




    # Generate Quantize 1 index:
    def generate_dataset_32_index_single_quantize(self,data):
        self.model.eval()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        # _, _, info_dict = self.model.Stage1_quantize_get_non_empty_index_auto(c, zero_dict)   # inputs: B x C x reso x reso x reso
        _, _, info_dict = self.model.Stage1_quantize_get_index(c)
        assert c['xz'].shape[0] == 1 # Batch size = 1

        plane_index = info_dict['index'][1].reshape(-1)

        data_dict = {}
        data_dict['plane_index'] = plane_index.cpu().numpy()
        model_name = data['model'][0]
        category = data['category'][0]

        self.save_process_data_32_index_single_quantize(data_dict, category, model_name)
        return True

#wy_add
    def generate_diffusion_dataset_plane_feature(self, data):
        self.model.eval()
        inputs = data.get('inputs').to(self.device)
        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        plane_feature = self.model.Stage1_get_plane_feature(c)

        assert c['xz'].shape[0] == 1  # Batch size = 1
        data_dict = {}
        data_dict['plane_feature'] = plane_feature.cpu().numpy()
        model_name = data['model'][0]
        category = data['category'][0]

        self.save_process_diffusion_data(data_dict, category, model_name)
        return True

    def diffusion_dataset_to_mesh(self, data):
        self.model.eval()
        inputs = data.get('inputs').to(self.device)
        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        plane_feature = self.model.Stage1_get_plane_feature(c)

        assert c['xz'].shape[0] == 1  # Batch size = 1
        data_dict = {}
        data_dict['plane_feature'] = plane_feature.cpu().numpy()
        model_name = data['model'][0]
        category = data['category'][0]

        self.save_process_diffusion_data(data_dict, category, model_name)
        return True

    def save_process_data_32_index_single_quantize(self, data_dict,category, model_name,):
        '''Save Data Dict without coord / index'''
        category_path = os.path.join(self.output_dataset_path,category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
        output_path = os.path.join(self.output_dataset_path,category,model_name+'.npz')
        np.savez(output_path, index = data_dict['plane_index'])

#wy_add
    def save_process_diffusion_data(self, data_dict,category, model_name,):
        '''Save Data Dict without coord / index'''
        category_path = os.path.join(self.output_dataset_path,category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
        output_path = os.path.join(self.output_dataset_path,category,model_name+'.npz')
        np.savez(output_path, index = data_dict['plane_feature'])

    def generate_dataset_64_index(self,data,zero_dict):
        self.model.eval()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        _, _, info_dict = self.model.Stage1_quantize_get_non_empty_index_auto(c, zero_dict)   # inputs: B x C x reso x reso x reso
        
        assert c['xz'].shape[0] == 1 # Batch size = 1
        plane_index_xz = info_dict['xz'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_xy = info_dict['xy'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_yz = info_dict['yz'][1].reshape(-1)     # 1 x (reso x reso)  4096
        data_dict = {}

        data_dict['xz_plane_index'] = plane_index_xz.cpu().numpy()
        data_dict['xy_plane_index'] = plane_index_xy.cpu().numpy()
        data_dict['yz_plane_index'] = plane_index_yz.cpu().numpy()

        model_name = data['model'][0]
        category = data['category'][0]
        # print('model',model_name,'cateogry:',category)
        self.save_process_data_64_index(data_dict, category, model_name)
        return True
    

    def save_process_data_64_index(self, data_dict,category, model_name):
        '''Save Data Dict without coord / index'''
        category_path = os.path.join(self.output_dataset_path,category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
        output_path = os.path.join(self.output_dataset_path,category,model_name+'.npz')
        np.savez(output_path,
                xz_index=data_dict['xz_plane_index'],
                xy_index=data_dict['xy_plane_index'],
                yz_index=data_dict['yz_plane_index'])
    


    def generate_dataset_auto_zero_index_count(self,data,zero_dict):
        self.model.eval()
        inputs = data.get('inputs').to(self.device)   

        # Stage1 Encode ---> Get Quantize Index
        c = self.model.Stage1_encode_inputs(inputs)
        _, _, info_dict = self.model.Stage1_quantize_get_non_empty_index_auto(c, zero_dict)   # inputs: B x C x reso x reso x reso
        
        assert c['xz'].shape[0] == 1 # Batch size = 1
        plane_index_xz = info_dict['xz'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_xy = info_dict['xy'][1].reshape(-1)     # 1 x (reso x reso)  4096
        plane_index_yz = info_dict['yz'][1].reshape(-1)     # 1 x (reso x reso)  4096

        # Add End Token Padding & Truncated Sequences
        coord = torch.arange(0, plane_index_xz.shape[0])        # coordinate: 0 - 4095
        xz_mask = ~info_dict['mask_xz'].reshape(-1)  # 1 x 4096          non empty index
        xy_mask = ~info_dict['mask_xy'].reshape(-1)  # 1 x 4096          non empty index
        yz_mask = ~info_dict['mask_yz'].reshape(-1)  # 1 x 4096          non empty index

        non_empty_xz, coord_xz = plane_index_xz[xz_mask], coord[xz_mask]  # L1
        non_empty_xy, coord_xy = plane_index_xy[xy_mask], coord[xy_mask]  # L2
        non_empty_yz, coord_yz = plane_index_yz[yz_mask], coord[yz_mask]  # L3

        count_xz = non_empty_xz.shape[0]
        count_xy = non_empty_xy.shape[0]
        count_yz = non_empty_yz.shape[0]

        # data_dict = {}
        # xz_dict, xy_dict, yz_dict = {}, {}, {}
        # xz_dict['index'], xz_dict['coord'] = non_empty_xz, coord_xz 
        # xy_dict['index'], xy_dict['coord'] = non_empty_xy, coord_xy
        # yz_dict['index'], yz_dict['coord'] = non_empty_yz, coord_yz
        
        # xz_dict = self.align_seq(xz_dict) # unfinished
        # xy_dict = self.align_seq(xy_dict) # unfinished
        # yz_dict = self.align_seq(yz_dict) # unfinished
        # data_dict['xz'], data_dict['xy'], data_dict['yz'] = xz_dict, xy_dict, yz_dict

        # model_name = data['model'][0]
        # category = data['category'][0]
        # print('model',model_name,'cateogry:',category)

        return count_xz, count_xy, count_yz
