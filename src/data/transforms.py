import numpy as np
import random
import open3d as o3d


# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        # self.stddev = stddev        # 0.005
        self.stddev = 0.0005
        print('std dev', self.stddev)

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        indices = np.random.randint(points.shape[0], size=self.N)
        # indices = np.random.choice(points.shape[0], size=self.N, replace=False)
        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]

        return data_out

class Subsample_Partial_Pointcloud(object):
    '''Sub sample Partial Point Cloud'''

    def __init__(self, N):
        self.N = N
        self.low = 1 / 4
        self.upp = 3 / 4 
        
    def __call__(self, data):
        ''' Calls the transformation.
        Args: 
            data (dict): data dictionary
        '''

        data_out = data.copy()
        points = data[None]             # N x 3
        normals = data['normals']

        num_points = points.shape[0]
        num_crop = random.randint(int(num_points * self.low), int(num_points * self.upp))

        center = np.random.randn(1,3)
        center_n = center / np.linalg.norm(center, ord=2)   # 1,3
        distance_matrix =  np.linalg.norm(center_n - points, ord=2, axis=-1)    # N
        idx = np.argsort(distance_matrix, axis=-1)      # N 

        partial_points = points.copy()[idx[num_crop:], :]
        partial_normal = normals[idx[num_crop:], :]

        # Subsample N Points from the partial point
        indices = np.random.randint(partial_points.shape[0], size=self.N)
        data_out[None] = partial_points[indices, :]
        data_out['normals'] = partial_normal[indices, :]

        return data_out



class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                None: points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out


