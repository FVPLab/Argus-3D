import os
import torch
import torch.optim as optim
# from tensorboardX import SummaryWriter
import argparse
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
from tqdm import tqdm
import time
import numpy as np
import time, datetime
import random

# os.environ['CUDA_VISIBLE_DEVICES']='1'

# Arguments:
np.random.seed(41)
random.seed(41)

parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.p'
)
parser.add_argument('--config', default='configs/stage2.yaml')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')
parser.add_argument('--cate', type=str, default='chair')
parser.add_argument('--tags', type=str, default='transformer3072_24_32')
parser.add_argument('--plane_resolution', type=int, default=256)
parser.add_argument('--embedding_dim', type=int, default=512)
parser.add_argument('--embedding_num', type=int, default=8192)

# parser.add_argument('--sequence_length', type=int)
parser.add_argument('--transformer_embed_dim', type=int, default=3072)
parser.add_argument('--transformer_n_head', type=int, default=24)
parser.add_argument('--transformer_layer', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# Set t0
t0 = time.time()

cfg['model']['encoder_kwargs']['plane_resolution'] = args.plane_resolution
cfg['model']['quantizer_kwargs']['embedding_dim'] = args.embedding_dim
cfg['model']['quantizer_kwargs']['embedding_num'] = args.embedding_num

cfg['model']['stage2_model_kwargs']['sequence_length'] = (args.plane_resolution // 8) ** 2 + 1
cfg['model']['stage2_model_kwargs']['position_num'] = (args.plane_resolution // 8) ** 2
cfg['model']['stage2_model_kwargs']['transformer_embed_dim'] = args.transformer_embed_dim
cfg['model']['stage2_model_kwargs']['stage2_embed_dim'] = args.transformer_embed_dim
cfg['model']['stage2_model_kwargs']['transformer_n_head'] = args.transformer_n_head
cfg['model']['stage2_model_kwargs']['transformer_layer'] = args.transformer_layer

cfg['stage1_load_path'] = 'output/PR256_ED512_EN8192/model_stage1.pt'
out_dir = os.path.join('output/PR256_ED512_EN8192/class-guide', args.tags)
cfg['training']['out_dir'] = out_dir

batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
exit_after = args.exit_after
model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')


# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_counter = defaultdict(int)

model_full = config.get_stage2_model_joint_baseline_reso32_cond_quant_single_cond_class(cfg, device=device)

# Generator 
generator_stage2 = config.get_stage2_generator(model_full, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model_full)

try:
    load_dict = checkpoint_io.load('model_stage2.pt')#warning
    print("\nLoading Stage2 Model Success\n")
except FileExistsError:
    print("\nLoading Stage2 Fail !!!!!!\n")
    load_dict = dict()

epoch_it = load_dict.get('epoch_it',0)
it = load_dict.get('it',0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

# Print model
nparameters = sum(p.numel() for p in model_full.parameters())
print('Total number of parameters: %d' % nparameters)
print('output path:', cfg['training']['out_dir'])

count = 2000

class_dict = {0: 'plane', 1: 'ashcan', 2: 'bag', 3: 'basket', 4: 'bathtub', 5: 'bed', 6: 'bench', 7: 'birdhouse', 8: 'bookshelf', 9: 'bottle', 10: 'bowl', 11: 'bus', 12: 'cabinet', 13: 'camera', 14: 'can', 15: 'cap', 16: 'car', 17: 'cell_phone', 18: 'chair', 19: 'clock', 20: 'keyboard', 21: 'dishwasher', 22: 'display', 23: 'earphone', 24: 'faucet', 25: 'file_cabinet', 26: 'guitar', 27: 'helmet', 28: 'jar', 29: 'knife', 30: 'lamp', 31: 'laptop', 32: 'loudspeaker', 33: 'mailbox', 34: 'microphone', 35: 'microwave', 36: 'bike', 37: 'mug', 38: 'piano', 39: 'pillow', 40: 'pistol', 41: 'flowerpot', 42: 'printer', 43: 'remote', 44: 'rifle', 45: 'rocket', 46: 'skateboard', 47: 'sofa', 48: 'stove', 49: 'table', 50: 'telephone', 51: 'tower', 52: 'train', 53: 'vessel', 54: 'washer'}
class_reverse_dict = {'plane': 0, 'ashcan': 1, 'bag': 2, 'basket': 3, 'bathtub': 4, 'bed': 5, 'bench': 6, 'birdhouse': 7, 'bookshelf': 8, 'bottle': 9, 'bowl': 10, 'bus': 11, 'cabinet': 12, 'camera': 13, 'can': 14, 'cap': 15, 'car':16, 'cell_phone': 17, 'chair':18, 'clock': 19, 'keyboard': 20, 'dishwasher': 21, 'display': 22, 'earphone': 23, 'faucet': 24, 'file_cabinet': 25, 'guitar': 26, 'helmet': 27, 'jar': 28, 'knife': 29, 'lamp': 30, 'laptop': 31, 'loudspeaker': 32, 'mailbox': 33, 'microphone': 34, 'microwave': 35, 'bike': 36, 'mug': 37, 'piano': 38, 'pillow': 39, 'pistol': 40, 'flowerpot': 41, 'printer': 42, 'remote': 43, 'rifle':44, 'rocket': 45, 'skateboard': 46, 'sofa': 47, 'stove': 48, 'table': 49, 'telephone': 50, 'tower': 51, 'train': 52, 'vessel': 53, 'washer': 54}
# test_num_dict = {'plane':809,'bench':365,'table':1687,'sofa':634,'cabinet':314,'display':218,'chair':1355,'rifle':474,'vessel':387,'car':701,'loudspeaker':319,'lamp':463,'telephone':217,'bed':46}

category = args.cate
model_full.restore_from_stage1()

bs = args.batch_size
path_name = os.path.join(out_dir,'class_cond','stage2_{}_class_{}'.format(it,category))
os.makedirs(path_name,exist_ok=True)
class_label = class_reverse_dict[category]
# test_num = test_num_dict[category]

for i in tqdm(range(count // bs + 1),desc='Category {}'.format(category)):
    class_idx = torch.tensor(class_label, dtype=torch.long).repeat(bs).reshape(bs)
    mesh_list = generator_stage2.generate_mesh_stage2_reso32_cond_class(bs=bs,top_k=250,class_idx=class_idx)
    # Get statistics
    for j in range(len(mesh_list)):
        count += 1
        mesh = mesh_list[j]
        if not mesh.vertices.shape[0]:
            continue
        mesh.export(os.path.join(path_name, 'Stage2_{}_{}.obj'.format(it, count+1000)))


    
