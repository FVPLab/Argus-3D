import torch
from torch import nn
import os 
from src.stage2_model import training
from src.stage2_model import generation
from src.stage2_model import generation_single
from src.stage2_model import generation_joint
from src.stage2_model import generation_3plane
from src.stage2_model.model_joint_baseline_reso32_cond_quant_single_cond_class import GET_FAST_Transformer_Joint_baseline_reso32_cond_quant_single_cond_class

from src import data
import numpy as np


def get_stage2_model_joint_baseline_reso32_cond_quant_single_cond_class(cfg, device=None,dataset=None, **kwargs):
    '''
    Returns the Stage2 Model: Single Transformer  3Plane Condition  Noise !!!  Single Transformer  Different Order
    Containing:
        1. 3D-VQVAE
        2. Fast-Transformer
    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    model = GET_FAST_Transformer_Joint_baseline_reso32_cond_quant_single_cond_class(cfg, device, dataset)

    return model

def get_stage2_trainer(model_stage2, cfg, device, **kwargs):
    '''
        Returns the trainer object.
    Args:
        stage2_model (nn.Module): GET_FAST_Transformer Model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Stage2_Trainer(
        model_stage2,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'], 
        cfg=cfg
    )

    return trainer

def get_stage2_generator(model_stage2, cfg, device, **kwargs):
    '''
       Return the Stage2 Generator: (Autoregressive) Generator
    
    Args:
        model (nn.Module): GET_FAST_Transformer Model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    vol_bound = None
    vol_info = None
    generator_stage2 = generation.Generator3D_Stage2(
        model_stage2,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info = vol_info,
        vol_bound = vol_bound,
    )
    return generator_stage2


def get_stage2_generator_single(model_stage2, cfg, device, **kwargs):
    '''
       Return the Stage2 Generator: (Autoregressive) Generator Single Transformer
    
    Args:
        model (nn.Module): GET_FAST_Transformer Model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    vol_bound = None
    vol_info = None
    generator_stage2 = generation_single.Generator3D_Stage2(
        model_stage2,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info = vol_info,
        vol_bound = vol_bound,
    )
    return generator_stage2

def get_stage2_generator_joint(model_stage2, cfg, device, **kwargs):
    '''
       Return the Stage2 Generator: (Autoregressive) Generator Joint Transformer
    
    Args:
        model (nn.Module): GET_FAST_Transformer Model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    vol_bound = None
    vol_info = None
    generator_stage2 = generation_joint.Generator3D_Stage2(
        model_stage2,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info = vol_info,
        vol_bound = vol_bound,
    )
    return generator_stage2

def get_stage2_generator_3plane_cond(model_stage2, cfg, device, **kwargs):
    '''
       Return the Stage2 Generator: (Autoregressive) Generator Joint Transformer
    
    Args:
        model (nn.Module): GET_FAST_Transformer Model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    vol_bound = None
    vol_info = None
    generator_stage2 = generation_3plane.Generator3D_Stage2(
        model_stage2,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info = vol_info,
        vol_bound = vol_bound,
    )
    return generator_stage2


def get_stage2_generator_3plane_cond_noise(model_stage2, cfg, device, **kwargs):
    '''
       Return the Stage2 Generator: (Autoregressive) Generator Joint Transformer Noise 
    
    Args:
        model (nn.Module): GET_FAST_Transformer Model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    vol_bound = None
    vol_info = None
    generator_stage2 = generation_3plane.Generator3D_Stage2(
        model_stage2,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info = vol_info,
        vol_bound = vol_bound,
    )
    return generator_stage2




def get_stage2_data_fields(mode, cfg):
    ''' Returns the data fields.
        Stage2 Add Shape Image Fields
    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        assert input_type == 'pointcloud'
        fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
        )

        # fields['shape_img'] = data.ShapeImageField(cfg['data']['img_folder_name'])
        # fields['class_id'] = data.

    
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            assert input_type == 'pointcloud'
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields



def get_stage2_data_quantize_fields(mode, cfg):
    ''' Returns the data fields.
        Stage2 Add Shape Image Fields
    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        assert input_type == 'pointcloud'
        fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
        )

        # fields['shape_img'] = data.ShapeImageField(cfg['data']['img_folder_name'])
        # fields['class_id'] = data.
    fields['quantize'] = data.ShapeQuantizeField(cfg['data']['quantize_dataset_path'])

    
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            assert input_type == 'pointcloud'
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields


def get_stage2_data_diffusion_fields(mode, cfg):
    ''' Returns the data fields.
        Stage2 Add Shape Image Fields
    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        assert input_type == 'pointcloud'
        fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
        )

        # fields['shape_img'] = data.ShapeImageField(cfg['data']['img_folder_name'])
        # fields['class_id'] = data.
    fields['quantize'] = data.ShapeQuantizeField_Diffusion(cfg['data']['quantize_dataset_path'])

    
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            assert input_type == 'pointcloud'
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields

def get_stage2_data_diffusion_fields_single(mode, cfg):
    ''' Returns the data fields.
        Stage2 Add Shape Image Fields
    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        assert input_type == 'pointcloud'
        fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
        )

        # fields['shape_img'] = data.ShapeImageField(cfg['data']['img_folder_name'])
        # fields['class_id'] = data.
    fields['quantize'] = data.ShapeQuantizeField_Diffusion_Single(cfg['data']['quantize_dataset_path'])

    
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            assert input_type == 'pointcloud'
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)
    import pdb
    # pdb.set_trace()

    return fields


def get_stage2_data_image_condition_fields(mode, cfg):
    ''' Returns the data fields.
        Stage2 Add Shape Image Fields
    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        assert input_type == 'pointcloud'
        fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
        )

        # fields['shape_img'] = data.ShapeImageField(cfg['data']['img_folder_name'])
        # fields['class_id'] = data.
    fields['quantize'] = data.ShapeQuantizeField_Diffusion(cfg['data']['quantize_dataset_path'])

    fields['shape_img'] = data.ShapeImageField(cfg['data']['img_folder_name'])
    
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            assert input_type == 'pointcloud'
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
