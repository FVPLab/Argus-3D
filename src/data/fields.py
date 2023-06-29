import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from src.data.core import Field
from src.utils import binvox_rw
from src.common import coord2index, normalize_coord
from torchvision import transforms

import pdb

class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


#wy_add
class DiffusionNPZField(Field):
    ''' Shape Quantize Field Diffusion without coord

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, dataset_root_path, multi_files=None):
        self.dataset_root_path = dataset_root_path
        self.multi_files = multi_files  # default: None
        self.file_name = '.npz'

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        e.g: model_path: /home/hongyuxin/icml_3d/occupancy_networks-master/data/ShapeNet/02691156/1a04e3eab45ca15dd86060f189eb133
        '''
        file_path = '/'.join(model_path.split('/')[-2:])  # e.g: 02691156/1a04e3eab45ca15dd86060f189eb133
        file_path = os.path.join(self.dataset_root_path, file_path+self.file_name)

        plane_feature = np.load(file_path)['index'][0,:].transpose(2,0,1)#[1,reso,reso,c] to [c,reso,reso]

        return plane_feature         # data: {None:tensor(3000,3),normals:(3000,3)}


# 3D Fields
class PatchPointsField(Field):
    ''' Patch Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape and then split to patches.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files
        
    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        # acquire the crop
        ind_list = []
        for i in range(3):
            ind_list.append((points[:, i] >= vol['query_vol'][0][i])
                     & (points[:, i] <= vol['query_vol'][1][i]))
        ind = ind_list[0] & ind_list[1] & ind_list[2]
        data = {None: points[ind],
                    'occ': occupancies[ind],
            }
            
        if self.transform is not None:
            data = self.transform(data)

        # calculate normalized coordinate w.r.t. defined query volume
        p_n = {}
        for key in vol['plane_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(data[None].copy(), vol['input_vol'], plane=key)
        data['normalized'] = p_n

        return data

class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data     # data: {None:tensor(2048,3),occ:tensor(2048)}

class VoxelsField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PatchPointCloudField(Field):
    ''' Patch point cloud field.

    It provides the field used for patched point cloud data. These are the points
    randomly sampled on the mesh and then partitioned.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform=None, transform_add_noise=None, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files

    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        # add noise globally
        if self.transform is not None:
            data = {None: points, 
                    'normals': normals}
            data = self.transform(data)
            points = data[None]

        # acquire the crop index
        ind_list = []
        for i in range(3):
            ind_list.append((points[:, i] >= vol['input_vol'][0][i])
                    & (points[:, i] <= vol['input_vol'][1][i]))
        mask = ind_list[0] & ind_list[1] & ind_list[2]# points inside the input volume
        mask = ~mask # True means outside the boundary!!
        data['mask'] = mask
        points[mask] = 0.0
        
        # calculate index of each point w.r.t. defined resolution
        index = {}
        
        for key in vol['plane_type']:
            index[key] = coord2index(points.copy(), vol['input_vol'], reso=vol['reso'], plane=key)
            if key == 'grid':
                index[key][:, mask] = vol['reso']**3
            else:
                index[key][:, mask] = vol['reso']**2
        data['ind'] = index
        
        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete

class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform=None, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)
        
        data = {
            None: points,
            'normals': normals,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data         # data: {None:tensor(3000,3),normals:(3000,3)}

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PartialPointCloudField(Field):
    ''' Partial Point cloud field.

    It provides the field used for partial point cloud data. These are the points
    randomly sampled on the mesh and a bounding box with random size is applied.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
        part_ratio (float): max ratio for the remaining part
    '''
    def __init__(self, file_name, transform=None, multi_files=None, part_ratio=0.7):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files
        self.part_ratio = part_ratio

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        
        side = np.random.randint(3)
        xb = [points[:, side].min(), points[:, side].max()]
        length = np.random.uniform(self.part_ratio*(xb[1] - xb[0]), (xb[1] - xb[0]))
        ind = (points[:, side]-xb[0])<= length
        data = {
            None: points[ind],
            'normals': normals[ind],
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class ShapeImageField(Field):
    ''' ShapeImage Field.

    It provides the field to load shape image. 

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, img_transform):
        self.file_name = file_name       # 'img_choy2016'
        self.transform = img_transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''

        img_folder_path = os.path.join(model_path, self.file_name)
        file_list = os.listdir(img_folder_path)
        img_list = []
        for l1 in file_list:
            if os.path.splitext(l1)[1] == '.png' or os.path.splitext(l1)[1] == '.jpg':
                img_list.append(l1)
        # if "cameras.npz" in img_list:
        #     img_list.remove("cameras.npz")
        img_list.sort()
        idx = np.random.randint(0, len(img_list)-1)
        img_path = os.path.join(img_folder_path, img_list[idx])
        img = Image.open(img_path)
        img = self.transform(img)

        data = {None: img}
        return data


class ShapeQuantizeField(Field):
    ''' Shape Quantize Field

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, dataset_root_path, multi_files=None):
        self.dataset_root_path = dataset_root_path
        self.multi_files = multi_files  # default: None
        self.file_name = '.npz'

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        e.g: model_path: /home/hongyuxin/icml_3d/occupancy_networks-master/data/ShapeNet/02691156/1a04e3eab45ca15dd86060f189eb133
        '''
        file_path = '/'.join(model_path.split('/')[-2:])  # e.g: 02691156/1a04e3eab45ca15dd86060f189eb133
        file_path = os.path.join(self.dataset_root_path, file_path+self.file_name)

        quantize_dict = np.load(file_path)
        xz_index = quantize_dict['xz_index']    # 1024
        xy_index = quantize_dict['xy_index']
        yz_index = quantize_dict['yz_index']

        xz_coord = quantize_dict['xz_coord']
        xy_coord = quantize_dict['xy_coord']
        yz_coord = quantize_dict['yz_coord']

        data = {
            'xz_index': xz_index,
            'xy_index': xy_index,
            'yz_index': yz_index,
            'xz_coord': xz_coord,
            'xy_coord': xy_coord,
            'yz_coord': yz_coord,
        }

        return data         # data: {None:tensor(3000,3),normals:(3000,3)}
    

class ShapeQuantizeField_Diffusion(Field):
    ''' Shape Quantize Field Diffusion without coord

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, dataset_root_path, multi_files=None):
        self.dataset_root_path = dataset_root_path
        self.multi_files = multi_files  # default: None
        self.file_name = '.npz'

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = '/'.join(model_path.split('/')[-2:])  # e.g: 02691156/1a04e3eab45ca15dd86060f189eb133
        file_path = os.path.join(self.dataset_root_path, file_path+self.file_name)

        quantize_dict = np.load(file_path)
        xz_index = quantize_dict['xz_index']    # 1024
        xy_index = quantize_dict['xy_index']
        yz_index = quantize_dict['yz_index']

        data = {
            'xz_index': xz_index,
            'xy_index': xy_index,
            'yz_index': yz_index,
        }

        return data         # data: {None:tensor(3000,3),normals:(3000,3)}
    


class ShapeQuantizeField_Diffusion_Single(Field):
    ''' Shape Quantize Field Diffusion without coord

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, dataset_root_path, multi_files=None):
        self.dataset_root_path = dataset_root_path
        self.multi_files = multi_files  # default: None
        self.file_name = '.npz'

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        e.g: model_path: /home/hongyuxin/icml_3d/occupancy_networks-master/data/ShapeNet/02691156/1a04e3eab45ca15dd86060f189eb133
        '''
        file_path = '/'.join(model_path.split('/')[-2:])  # e.g: 02691156/1a04e3eab45ca15dd86060f189eb133
        file_path = os.path.join(self.dataset_root_path, file_path+self.file_name)

        quantize_dict = np.load(file_path)
        # import pdb
        # pdb.set_trace()
        index = quantize_dict['index']    # 1024

        data = {
            'index': index
        }
        return data         # data: {None:tensor(3000,3),normals:(3000,3)}


class Partial_PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform=None, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)
        
        data = {
            None: points,
            'normals': normals,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data         # data: {None:tensor(3000,3),normals:(3000,3)}

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class Partial_PointCloud_Metric_Field(Field):
    ''' Point cloud field. Metric

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, metric_test_folder):
        self.metric_test_folder = metric_test_folder     # /home/hongyuxin/icml_3d/c_occ_net_codebook_3plane/partial_pointcloud_data


    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        # model_path: "/home/hongyuxin/icml_3d/occupancy_networks-master/data/ShapeNet/02691156/1a04e3eab45ca15dd86060f189eb133"
        # file_path = os.path.join()
        name_list = model_path.split('/')[-2:]  # ['02691156', '1a04e3eab45ca15dd86060f189eb133']
        category_name, model_name = name_list[0], name_list[1]
        file_path = os.path.join(self.metric_test_folder, category_name, model_name)
        # print(file_path)
        pointcloud_dict = np.load(file_path)
        # print(pointcloud_dict.keys())
        partial_points = pointcloud_dict['partial_point'].astype(np.float32)
        gt_points = pointcloud_dict['gt_point'].astype(np.float32)
        data = {
            'partial': partial_points,
            'gt': gt_points
        }
        return data


    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete