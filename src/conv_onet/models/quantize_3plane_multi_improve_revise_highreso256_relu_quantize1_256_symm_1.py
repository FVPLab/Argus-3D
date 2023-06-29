
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoder.unet import UNet
from src.conv_onet.models.quantize_2D_improve_revise import VectorQuantizer


class Quantize_multi_improve_revise_256_relu_quantize1_256_symm_1(nn.Module):
    '''
        Quantize 1 Feature Plane
        Quantize 3 Plane Features
        Returns: dict{'xz','xy','yz'}   Feature planes
        Inputs: 
             - n_e : number of embeddings
             - e_dim : dimension of embedding
             - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    '''

    def __init__(self, cfg, n_e, e_dim, c_dim, beta, unet_kwargs):
        super(Quantize_multi_improve_revise_256_relu_quantize1_256_symm_1, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim      # Codebook Embedding dim
        self.beta = beta

        try:
            self.project_dim = cfg['model']['quantizer_kwargs']['reduce_dim']
        except:
            self.project_dim = 8
        # self.quantize_mode = quantize_mode  # 3_Codebook, 1_Codebook
        print('Using Single Quantizer  !!!! Reduce {} High Reso {} Relu Quantize 256 Symm 1'.format(self.project_dim, 256))


        self.quantizer =  VectorQuantizer(n_e=n_e,e_dim=self.project_dim,beta=beta)

        # self.unet_2d = UNet(e_dim, in_channels=e_dim, **unet_kwargs)
        self.unet_after_quant = UNet(e_dim, in_channels=e_dim, **unet_kwargs)

        self.project1 =  nn.Linear(e_dim, self.project_dim)  # reduce the dimension of code
        self.project2 =  nn.Linear(self.project_dim, e_dim) # project to original dimension


        ## Downsampler & Upsampler:    256 dimension --> 32 dimension               # in: b x 32 x 256 x 256
        self.down =  nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=2),       # out: b x 64 x 128 x 128
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2,stride=2),      # out: b x 128 x 64 x 64
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=2, stride=2),    # out: b x 256 x 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=self.e_dim,kernel_size=1)               # out: b x 256 x 32 x 32
            #nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)  # out: b x 256 x 32 x 32
        )

        self.up =  nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),                                    # out: b x 256 x 64 x 64
            nn.Conv2d(in_channels=self.e_dim,out_channels=128,kernel_size=3,stride=1,padding=1),  # out: b x 128 x 64 x 64
            # nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1),  # out: b x 128 x 64 x 64
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2,mode='nearest'),                                    # out: b x 128 x 128 x 128
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1),   # out: b x 64  x 128 x 128
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2,mode='nearest'),                                    # out: b x 64  x 256 x 256
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1),    # out: b x 32  x 256 x 256
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1)                        # out: b x 32  x 256 x 256
        )
    

        c_dim = 32
        self.unet_2d = UNet(c_dim, in_channels=c_dim, **unet_kwargs)

        # Encoder: B x 3C x 32 x 32 --> B x C x 32 x 32
        self.encode_3plane_reso32 =  nn.Sequential(
            nn.Conv2d(in_channels=3*c_dim, out_channels=3*c_dim, kernel_size=3, stride=1, padding=1),   
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3*c_dim, out_channels=c_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c_dim, out_channels=c_dim, kernel_size=1)
        )

        # Decoder: B x e_dim x 32 x 32 --> B x 3 e_dim x 32 x 32
        self.decode_3plane_reso256 =  nn.Sequential(
            nn.Conv2d(in_channels=c_dim, out_channels=3*c_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3*c_dim,out_channels=3*c_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3*c_dim, out_channels=3*c_dim, kernel_size=1)
        )


    def forward(self, z):
        # print('--\n--\n--\n')
        """
            Merger 3 Plane: 3 x B x C x 32 x 32 ---> 1 Plane (B x C x 32 x 32)
        """
        concat_x = torch.cat([z['xz'], z['xy'], z['yz']], dim=1)                     # B x 96 x 256 x 256 #[B, 96, res, res]
        bs, c, h, w = z['xz'].shape#[16, 32, 512, 512]

        plane_feature = self.encode_3plane_reso32(concat_x)           # Encode: B x C x 256 x 256 #[B, 32, res, res] reduce channel C 1/3
        plane_feature = self.down(plane_feature)                      # Encode: B x C x 32 x 32  #[B, 256, res/8, res/8]

        plane_feature = plane_feature.permute(0,2,3,1).contiguous()   # B x reso x reso x C  #[B, res/8, res/8, 256]
        plane_feature = self.project1(plane_feature)                  #[B, res/8, res/8, 4]

        fea, loss, info = self.quantizer(plane_feature)                 #[B, res/8, res/8 ,4]
        fea = self.project2(fea)                                        #[B, res/8, res/8 ,e_dim]
        fea = fea.permute(0, 3, 1, 2).contiguous()                      #[B, e_dim, res/8, res/8]

        # unet:
        fea = self.unet_after_quant(fea)                                #[B, e_dim, res/8, res/8]
        # import pdb
        # pdb.set_trace()
        # Upsample
        fea = self.up(fea)

        # decoder:
        fea = self.decode_3plane_reso256(fea)
        fea_xz, fea_xy, fea_yz = torch.split(fea, c, dim=1)
        
        fea_xz = self.unet_2d(fea_xz)
        fea_xy = self.unet_2d(fea_xy)
        fea_yz = self.unet_2d(fea_yz)


        feature_plane =  {}        
        loss_dict = {}

        feature_plane['xz'] = fea_xz
        feature_plane['xy'] = fea_xy
        feature_plane['yz'] = fea_yz

        loss_dict['xz'] = loss
        loss_dict['xy'] = torch.tensor(0)
        loss_dict['yz'] = torch.tensor(0)

        return feature_plane, loss_dict, info
    

    def get_codebook_entry(self, indices, shape):

        z_q = self.quantizer.get_codebook_entry(indices, shape)  # B x reso x reso x 4
        z_q = self.project2(z_q)                                 # B x reso x reso x C
        z_q = z_q.permute(0, 3, 1, 2).contiguous()         # B x C x reso x reso
        
        z_q = self.unet_after_quant(z_q)

        # up:
        fea = self.up(z_q)

        fea = self.decode_3plane_reso256(fea)
        fea_xz, fea_xy, fea_yz = torch.split(fea, 32, dim=1)

        fea_xz = self.unet_2d(fea_xz)
        fea_xy = self.unet_2d(fea_xy)
        fea_yz = self.unet_2d(fea_yz)

        return fea_xz, fea_xy, fea_yz

    
    def get_quantize_index(self, z):
        
        concat_x = torch.cat([z['xz'], z['xy'], z['yz']], dim=1)                     # B x 96 x 256 x 256
        plane_feature = self.encode_3plane_reso32(concat_x)           # Encode: B x C x 256 x 256
        plane_feature = self.down(plane_feature)      

        bs, c, h, w = plane_feature.shape

        plane_feature = plane_feature.permute(0,2,3,1).contiguous()   # B x reso x reso x C
        plane_feature = self.project1(plane_feature)

        _, _, info = self.quantizer(plane_feature) 

        info_dict = {}

        info_dict['index'] = info
        return _, _, info_dict

    def get_plane_feature(self, z):
        concat_x = torch.cat([z['xz'], z['xy'], z['yz']], dim=1)  # B x 96 x 256 x 256
        plane_feature = self.encode_3plane_reso32(concat_x)  # Encode: B x C x 256 x 256
        plane_feature = self.down(plane_feature)
        plane_feature = plane_feature.permute(0, 2, 3, 1).contiguous()  # B x reso x reso x C => [B, res/8, res/8, 256]
        # plane_feature = self.project1(plane_feature) ##[B, res/8, res/8, 4]
        # _, _, info = self.qu
        # antizer(plane_feature)
        # info_dict = {}
        # info_dict['index'] = info
        return plane_feature
    

    def get_zero_index(self):
        zero_index_xz = self.quantizer_xz.get_zero_index()
        zero_index_xy = self.quantizer_xy.get_zero_index()
        zero_index_yz = self.quantizer_yz.get_zero_index()
        print('Zero Index: xz: {}, xy: {}, yz:{}'.format(zero_index_xz, zero_index_xy, zero_index_yz))
        index_dict = {}
        index_dict['xz_zero_index'] = zero_index_xz
        index_dict['xy_zero_index'] = zero_index_xy
        index_dict['yz_zero_index'] = zero_index_yz
        return index_dict
        

    

















