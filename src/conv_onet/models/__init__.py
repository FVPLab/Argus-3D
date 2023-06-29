import torch
import torch.nn as nn
from torch import distributions as dist

from src.conv_onet.models import quantize_3plane_multi_improve_revise_highreso256_relu_quantize1_256_symm_1


from src.conv_onet.models import decoder

# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder
}

# Quantizer Dictionary
quantizer_dict = {
    # 'quantize_grid': quantize_2D.VectorQuantizer,
    'quantize_3plane_multi_improve_revise_highreso256_relu_quantize1_256_symm_1': quantize_3plane_multi_improve_revise_highreso256_relu_quantize1_256_symm_1.Quantize_multi_improve_revise_256_relu_quantize1_256_symm_1,
}

class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder, quantizer=None, encoder=None, device=None):
        super().__init__()
        
        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None
        
        """Quantize Start"""
        self.quantizer = quantizer.to(device)
        """Quantize End"""

        self._device = device

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        c_quantize, _, _ = self.quantize(c)
        p_r = self.decode(p, c_quantize, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def quantize(self, inputs):
        """ 
            Quantize the Feature Grid:
        Args:
            input(tensor): the input
        """
        quantize_plane, loss_dict, info_dict = self.quantizer(inputs)
        return quantize_plane, loss_dict, info_dict
    
    def quantize_get_index(self,inputs):
        '''
         Get Quantize Index
        '''
        _, _, info_dict = self.quantizer.get_quantize_index(inputs)
        return _, _, info_dict

#wy_add
    def quantize_get_plane_feature(self,inputs):
        '''
         Get Quantize Index
        '''
        plane_feature = self.quantizer.get_plane_feature(inputs)
        return plane_feature
    
    def get_zero_index(self):
        '''Get zero Index'''
        zero_index_dict = self.quantizer.get_zero_index()
        return zero_index_dict
    
    def quantize_get_non_empty_index(self, inputs):
        '''Get Quantize Non Empty Index'''
        _, _, info_dict = self.quantizer.get_quantize_non_empty_index(inputs)
        return _, _, info_dict
        

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
