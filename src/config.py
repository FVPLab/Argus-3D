import yaml
from torchvision import transforms
from src import data
from src import conv_onet
from src import stage2_model

method_dict = {
    'conv_onet': conv_onet,
    'fast_transformer': stage2_model,
    # 'diffusion_model': stage2_diffusion,
    # 'maskgit': stage2_maskgit,
    # 'text2shape': stage2_text2shape,
    # # 'img_cond': stage2_img_cond,
    # 'partial_pointcloud': stage2_partial_point,
    # 'latent_diffusion': latent_diffusion
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)
    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model

def get_model_improve(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model_improve(
        cfg, device=device, dataset=dataset)
    return model

def get_stage2_model(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer
    model = method_dict[method].config.get_stage2_model(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_single(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer        # single transformer
    model = method_dict[method].config.get_stage2_model_single(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer        # single transformer
    model = method_dict[method].config.get_stage2_model_joint(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_3plane_cond(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer        # single transformer
    model = method_dict[method].config.get_stage2_model_joint_3plane_cond(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_3plane_cond_noise(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer        # single transformer
    model = method_dict[method].config.get_stage2_model_joint_3plane_cond_noise(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_3plane_cond_single(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer        # single transformer
    model = method_dict[method].config.get_stage2_model_joint_3plane_cond_single(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_3plane_cond_single_order(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer        # single transformer
    model = method_dict[method].config.get_stage2_model_joint_3plane_cond_single_order(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_3plane_cond_final(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer        # single transformer
    model = method_dict[method].config.get_stage2_model_joint_3plane_cond_final(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_3plane_cond_final_class_condition(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer        # single transformer
    model = method_dict[method].config.get_stage2_model_joint_3plane_cond_final_class_condition(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer        # single transformer
    model = method_dict[method].config.get_stage2_model_joint_baseline(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_single(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # fast_transformer        # single transformer GPt
    model = method_dict[method].config.get_stage2_model_joint_baseline_single(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_diffusion(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # diffusion_model  
    model = method_dict[method].config.get_stage2_model(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_diffusion_cond(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # diffusion_model  
    model = method_dict[method].config.get_stage2_model_cond(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_maskgit(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # MaskGIT 
    model = method_dict[method].config.get_stage2_model(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_maskgit_gpt(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # MaskGIT 
    model = method_dict[method].config.get_stage2_model_gpt(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_joint_baseline_reso32_cond_quant_single_text2shape(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model, img_transform = method_dict[method].config.get_stage2_model_quant_single_text2shape(
        cfg,device=device,dataset=dataset)
    return model, img_transform

# wy_add
def get_stage2_latent_diffusion(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # diffusion
    model = method_dict[method].config.get_latent_diffusion_joint_cond_quant_single(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_joint_baseline_reso32_cond_quant_single(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_quant_single(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_quant_single_pc_complete(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_quant_single_pc_complete(
        cfg,device=device,dataset=dataset)
    return model



def get_stage2_model_joint_baseline_reso32_cond_quant_single_pc_complete_pointnet(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_quant_single_pc_complete_pointnet(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_joint_baseline_reso32_cond_quant_single_pc_complete_cross(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_quant_single_pc_complete_cross(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_joint_baseline_reso32_cond_quant_single_cond_class(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_quant_single_cond_class(
        cfg,device=device,dataset=dataset)
    return model

#wy_add
def get_stage2_diffusion(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer
    model = method_dict[method].config.get_stage2_diffusion(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_quant_single_short_seq(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_quant_single_short_seq(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_short_seq(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_short_seq(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_short_seq_spatial(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_short_seq_spatial(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_xyz_cat(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_xyz_cat(
        cfg,device=device,dataset=dataset)
    return model



def get_stage2_model_joint_baseline_reso32_cond_xyz_independ(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_xyz_independ(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_xyz_2order(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_xyz_2order(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_joint_baseline_reso32_cond_xyz_2order_1(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_xyz_2order_1(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_xyz_2order_2(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_xyz_2order_2(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_joint_baseline_reso32_cond_3plane_cat(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_3plane_cat(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_joint_baseline_reso32_cond_sgd_resume(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_sgd_resume(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_concat(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_concat(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_joint_baseline_reso32_cond_concat_768(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_concat_768(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_att(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_att(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_maskgit_gpt_cond(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # MaskGIT 
    model = method_dict[method].config.get_stage2_model_gpt_cond(
        cfg,device=device,dataset=dataset)
    return model


def get_stage2_model_joint_baseline_reso32_img_cond(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_img_cond(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_class(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_class(
        cfg,device=device,dataset=dataset)
    return model

def get_stage2_model_joint_baseline_reso32_cond_class_ada(cfg,device=None,dataset=None):
    method = cfg['stage2_method']     # FAST Transformer 
    model = method_dict[method].config.get_stage2_model_joint_baseline_reso32_cond_class_ada(
        cfg,device=device,dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer

def get_stage2_trainer(model_stage2, cfg, device):
    ''' Returns Stage2 trainer 
    
    Args:
        model_stage2 (nn.Module): GET_FAST_Transformer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['stage2_method']
    trainer_stage2 = method_dict[method].config.get_stage2_trainer(
        model_stage2, cfg, device)
    return trainer_stage2


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator

# Stage2 Generator for final mesh extraction
def get_stage2_generator(model_stage2,cfg,device):
    ''' Returns a Stage2 Generator Instance 
    
    Args:
        model_stage2 (nn.Module): GET_FAST_Transformer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['stage2_method']   # fast_transformer
    generator_stage2 = method_dict[method].config.get_stage2_generator(model_stage2,cfg,device)
    return generator_stage2


def get_stage2_generator_single(model_stage2,cfg,device):
    ''' Returns a Stage2 Generator Instance  Single Transformer
    
    Args:
        model_stage2 (nn.Module): GET_FAST_Transformer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['stage2_method']   # fast_transformer
    generator_stage2 = method_dict[method].config.get_stage2_generator_single(model_stage2,cfg,device)
    return generator_stage2

def get_stage2_generator_joint(model_stage2,cfg,device):
    ''' Returns a Stage2 Generator Instance  Joint Transformer 
    
    Args:
        model_stage2 (nn.Module): GET_FAST_Transformer_joint
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['stage2_method']   # fast_transformer
    generator_stage2 = method_dict[method].config.get_stage2_generator_joint(model_stage2,cfg,device)
    return generator_stage2

def get_stage2_generator_3plane_cond(model_stage2,cfg,device):
    ''' Returns a Stage2 Generator Instance  Joint Transformer  3Plane Origin Transformer
    
    Args:
        model_stage2 (nn.Module): GET_FAST_Transformer_joint
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['stage2_method']   # fast_transformer
    generator_stage2 = method_dict[method].config.get_stage2_generator_3plane_cond(model_stage2,cfg,device)
    return generator_stage2

# Datasets
def get_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': 'train_val'
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
 
    return dataset


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'pointcloud':
        if cfg['training']['noisy']:
            transform = transforms.Compose([
                data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
                data.PointcloudNoise(cfg['data']['pointcloud_noise'])
            ])
            print('Using Noisy Pointcloud Input !!!!!')
        else:
            transform = transforms.Compose([
                data.SubsamplePointcloud(cfg['data']['pointcloud_n'])
            ])
            print('Without Using Noisy Pointcloud Input !!!!!')

        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], transform,
            multi_files= cfg['data']['multi_files']
        )
    elif input_type == 'partial_pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = data.PartialPointCloudField(
            cfg['data']['pointcloud_file'], transform,
            multi_files= cfg['data']['multi_files']
        )
    elif input_type == 'pointcloud_crop':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
    
        inputs_field = data.PatchPointCloudField(
            cfg['data']['pointcloud_file'], 
            transform,
            multi_files= cfg['data']['multi_files'],
        )
    
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field



'''
    Stage 2 Dataset: Add Shape Image Fields
'''

def get_stage2_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': "train_val"
    }

    split = splits[mode]
    print('Dataset Split Mode: {} !!!!!'.format(split))
    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_stage2_data_fields(mode, cfg)   # !!! Add Image Field
        # Input fields
        inputs_field = get_stage2_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
 
    return dataset

def get_stage2_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], transform,
            multi_files= cfg['data']['multi_files']
        )
    elif input_type == 'partial_pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = data.PartialPointCloudField(
            cfg['data']['pointcloud_file'], transform,
            multi_files= cfg['data']['multi_files']
        )
    elif input_type == 'pointcloud_crop':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
    
        inputs_field = data.PatchPointCloudField(
            cfg['data']['pointcloud_file'], 
            transform,
            multi_files= cfg['data']['multi_files'],
        )
    
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field

    

'''Quantize Index Dataset'''
def get_stage2_quantize_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': 'train_val'
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_stage2_data_quantize_fields(mode, cfg)   # !!! Add Image Field
        # Input fields
        inputs_field = get_stage2_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


'''Quantize Index Dataset Diffusion Index'''
def get_stage2_diffusion_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': 'train_val'
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_stage2_data_diffusion_fields(mode, cfg)   # !!! Add Image Field
        # Input fields
        inputs_field = get_stage2_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


'''Quantize Index Dataset Diffusion Index'''
def get_stage2_diffusion_dataset_single(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': 'train_val'
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_stage2_data_diffusion_fields_single(mode, cfg)   # !!! Add Image Field
        # Input fields
        inputs_field = get_stage2_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset

#wy_add
def get_stage2_latent_diffusion_dataset_single(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': 'train_val'
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':

        # Dataset fields
        # Method specific fields (usually correspond to output)

        fields = method_dict[method].config.get_stage2_data_latent_diffusion_fields_single(mode, cfg)   # !!! Add Image Field
        # Input fields
        inputs_field = get_stage2_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


'''Quantize Index Dataset Diffusion Index'''
def get_stage2_image_condition_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': 'train_val'
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_stage2_data_image_condition_fields(mode, cfg)   # !!! Add Image Field
        # Input fields
        inputs_field = get_stage2_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset



def get_stage2_partial_pointcloud_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': 'train_val'
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_stage2_data_partial_pointcloud_fields(mode, cfg)   # !!! Add Image Field
        # Input fields
        inputs_field = get_stage2_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset



def get_stage2_partial_pc_dataset_single(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': 'train_val'
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_stage2_data_partial_pointcloud_fields_quant_single(mode, cfg)   # !!! Add Image Field
        # Input fields
        inputs_field = get_stage2_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset




def get_stage2_partial_pointcloud_dataset_metric(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args: Metric
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': 'train_val'
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        # fields = method_dict[method].config.get_stage2_data_partial_pointcloud_fields(mode, cfg)   # !!! Add Image Field
        # # Input fields
        # inputs_field = get_stage2_inputs_field(mode, cfg)
        # if inputs_field is not None:
        #     fields['inputs'] = inputs_field

        # if return_idx:
        #     fields['idx'] = data.IndexField()

        # dataset = data.Shapes3dDataset(
        #     dataset_folder, fields,
        #     split=split,
        #     categories=categories,
        #     cfg = cfg
        # )

        fields = method_dict[method].config.get_stage2_partial_pointcloud_metric_fields(mode, cfg)

        dataset = data.Shapes3dDataset_subset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )

    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset




'''Text 2 Shape Dataset'''
def get_stage2_text2shape_single_dataset(mode, cfg, return_idx=False, img_transform=None):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'train_val': 'train_val'
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_stage2_data_text2shape_fields_single(mode, cfg, img_transform=img_transform)   # !!! Add Image Field
        # Input fields
        inputs_field = get_stage2_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset





# def get_stage2_image_condition_dataset(mode, cfg, return_idx=False):
#     ''' Returns the dataset.

#     Args:
#         model (nn.Module): the model which is used
#         cfg (dict): config dictionary
#         return_idx (bool): whether to include an ID field
#     '''
#     method = cfg['method']
#     dataset_type = cfg['data']['dataset']
#     dataset_folder = cfg['data']['path']
#     categories = cfg['data']['classes']

#     # Get split
#     splits = {
#         'train': cfg['data']['train_split'],
#         'val': cfg['data']['val_split'],
#         'test': cfg['data']['test_split'],
#         'train_val': 'train_val'
#     }

#     split = splits[mode]

#     # Create dataset
#     if dataset_type == 'Shapes3D':
#         # Dataset fields
#         # Method specific fields (usually correspond to output)
#         fields = method_dict[method].config.get_stage2_data_image_condition_fields(mode, cfg)   # !!! Add Image Field
#         # Input fields
#         inputs_field = get_stage2_inputs_field(mode, cfg)
#         if inputs_field is not None:
#             fields['inputs'] = inputs_field

#         if return_idx:
#             fields['idx'] = data.IndexField()

#         dataset = data.Shapes3dDataset(
#             dataset_folder, fields,
#             split=split,
#             categories=categories,
#             cfg = cfg
#         )
#     else:
#         raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

#     return dataset