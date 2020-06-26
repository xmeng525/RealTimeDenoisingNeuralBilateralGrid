"""
description: network architecture

@author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)
@author: QZheng
"""

import tensorflow as tf
import numpy as np
import math
from network_units import conv_layer, lrelu
from bilateral_grid_ops import create_bi_grid, slice_bi_grid

class DenoiserGuideNet(object):
    def __init__(
        self, input_shape, target_shape, ss=4, sr=4):
        self.input_shape = input_shape
        self.target_shape = target_shape
        
        self.source = tf.placeholder(tf.float32, self.input_shape, name='source')
        self.target = tf.placeholder(tf.float32, self.target_shape, name='target')
        
        self.ss = ss
        self.sr = sr

    def inference(self):
        tmp = self.tone_mapping(self.source[:,:,:, 0:3] * self.source[:,:,:,3:6])
        ae_input = tf.concat([tmp, self.source[:,:,:, 3:10]], axis=3)

        with tf.variable_scope('GuideNet'):
            net_out = self._denseconnect_conv(ae_input, net_name="GuideNet")
            guide_1 = net_out[:,:,:, 0] * 255
            guide_2 = net_out[:,:,:, 1] * 255
            guide_3 = net_out[:,:,:, 2] * 255
            weight_grid_1 = tf.tile(net_out[:,:,:, 3:4], [1,1,1,3])
            weight_grid_2 = tf.tile(net_out[:,:,:, 4:5], [1,1,1,3])
            tf.add_to_collection('guide_1', guide_1)
            tf.summary.histogram('guide_1', guide_1)

        with tf.variable_scope('BilateralGrid'):
            grid_1 = self._create_grid(self.source[:,:,:,0:3], guide_1, self.ss, self.sr)
            tf.add_to_collection('grid', grid_1)
            tf.summary.histogram('grid', grid_1)

            from_grid_hdr_1 = self._slice_grid(grid_1, guide_1, self.ss, self.sr)
            tf.add_to_collection('from_grid_hdr_1', from_grid_hdr_1)
            tf.summary.histogram('from_grid_hdr_1', from_grid_hdr_1)

            grid_2 = self._create_grid(self.source[:,:,:,0:3], guide_2, self.ss * 2, self.sr * 2)
            tf.add_to_collection('grid_2', grid_2)
            tf.summary.histogram('grid_2', grid_2)

            from_grid_hdr_2 = self._slice_grid(grid_2, guide_2, self.ss * 2, self.sr * 2)
            tf.add_to_collection('from_grid_hdr_2', from_grid_hdr_2)
            tf.summary.histogram('from_grid_hdr_2', from_grid_hdr_2)

            grid_3 = self._create_grid(self.source[:,:,:,0:3], guide_3, self.ss * 4, self.sr * 4)
            tf.add_to_collection('grid_3', grid_3)
            tf.summary.histogram('grid_3', grid_3)

            from_grid_hdr_3 = self._slice_grid(grid_3, guide_3, self.ss * 4, self.sr * 4)
            tf.add_to_collection('from_grid_hdr_3', from_grid_hdr_3)
            tf.summary.histogram('from_grid_hdr_3', from_grid_hdr_3)

        with tf.name_scope("FinalLoss"):
            denoised_hdr = from_grid_hdr_1 * weight_grid_1 + from_grid_hdr_2 * weight_grid_2 + \
                from_grid_hdr_3 * (1.0 - weight_grid_1 - weight_grid_2)

            denoised_hdr_abd = denoised_hdr * self.source[:,:,:,3:6]
            denoised_abd = self.tone_mapping(denoised_hdr_abd)

            from_grid_hdr_1_abd = from_grid_hdr_1 * self.source[:,:,:,3:6]
            from_grid_1_abd = self.tone_mapping(from_grid_hdr_1_abd)

            from_grid_hdr_2_abd = from_grid_hdr_2 * self.source[:,:,:,3:6]
            from_grid_2_abd = self.tone_mapping(from_grid_hdr_2_abd)

            from_grid_hdr_3_abd = from_grid_hdr_3 * self.source[:,:,:,3:6]
            from_grid_3_abd = self.tone_mapping(from_grid_hdr_3_abd)

            target_tm = self.tone_mapping(self.target)
            
            loss_all_L1 = tf.reduce_mean(tf.losses.absolute_difference(denoised_abd, target_tm))
            loss_grid_L1 = tf.reduce_mean(tf.losses.absolute_difference(from_grid_1_abd, target_tm))

            tf.summary.scalar('loss_all_L1', loss_all_L1)
            tf.summary.scalar('loss_grid_L1', loss_grid_L1)
        
        return {'source': self.source, 'target': self.target, 
                'guide_1': net_out[:,:,:, 0], 
                'guide_2': net_out[:,:,:, 1], 
                'guide_3': net_out[:,:,:, 2], 
                'weight_1': net_out[:,:,:,3],
                'weight_2': net_out[:,:,:,4],
                'denoised_hdr': denoised_hdr_abd, 
                'from_grid_hdr_1': from_grid_hdr_1_abd, 
                'from_grid_hdr_2': from_grid_hdr_2_abd,
                'from_grid_hdr_3': from_grid_hdr_3_abd, 
                'grid_1': grid_1, 'grid_2': grid_2, 'grid_3': grid_3, 
                'loss_all_L1': loss_all_L1, 'loss_grid_L1': loss_grid_L1}

    def _denseconnect_conv(self, corrupt_input, net_name=''):
        print('Creating ' + net_name + ' ...')
        current_input = corrupt_input  # input 10 channels

        prep_kernel_size = 5
        prep_out_ch = 20
        prep_stride = [1, 1, 1, 1]
        with tf.variable_scope('prep_0'):
            prep_conv_output = conv_layer(current_input, prep_kernel_size, prep_stride, prep_out_ch)
            current_input = lrelu(prep_conv_output)
            print('prep layer input_shape: ',
                  corrupt_input.get_shape().as_list())
            print('prep layer output_shape: ',
                  current_input.get_shape().as_list())

        # layer_cout = 5
        # mid_layer_out_ch = 5
        # mid_filter_size = 5
        # mid_layer_stride = [1, 1, 1, 1]
        # for layer_i in range(layer_cout):
        #     with tf.variable_scope('dn_conv_%d' % (layer_i)):
        #         conv_out = conv_layer(current_input, mid_filter_size, mid_layer_stride, mid_layer_out_ch)
        #         result = tf.concat([current_input, conv_out], 3)
        #         current_input = lrelu(result)
        #         print('Mid layer %d' % (layer_i + 1), ' out_shape: ', current_input.get_shape().as_list())

        final_layer_out_ch = 5  # ouptut 5 channel
        final_filter_size = 5
        final_layer_stride = [1, 1, 1, 1]
        with tf.variable_scope('final_conv') as scope:
            output = conv_layer(current_input, final_filter_size, final_layer_stride, final_layer_out_ch)
            output = tf.nn.sigmoid(output)
            print('Final layer out_shape: ', output.get_shape().as_list())

        return output

    def _create_grid(self, input_image, input_guide, ss, sr):
        with tf.device('/gpu:0'):
            input_attrs = tf.zeros([ss, sr], dtype=tf.int32)
            output = create_bi_grid(input_image, input_guide, input_attrs)
        return output

    def _slice_grid(self, input_grid, input_guide, ss, sr):
        with tf.device('/gpu:0'):
            input_attrs = tf.zeros([ss, sr], dtype=tf.int32)
            output = slice_bi_grid(input_grid, input_guide, input_attrs)
        return output

    def tone_mapping(self, input_image):
        with tf.name_scope("tone_mapping"):
            tone_mapped_color = tf.clip_by_value(
                tf.math.pow(tf.math.maximum(0., input_image), 0.454545), 0., 1.)
            return tone_mapped_color