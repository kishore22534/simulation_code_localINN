#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Pytorch
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
#from torchviz import make_dot # for visualizing the architecture as a graph

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom, RNVPCouplingBlock

# cuda
from torch.cuda.amp import GradScaler, autocast

#import getpass
import os
import threading
#import json

device='cuda' if torch.cuda.is_available() else 'cpu'
print("device is", device)

BATCHSIZE = int(500)
N_DIM = int(60)
COND_DIM = 6
COND_OUT_DIM = 12

x_low = -7
x_high = 7
x_length = x_high- x_low
y_low = -7
y_high =7
y_length = y_high -y_low

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(270, 270)
        self.linear2 = nn.Linear(270, 54)
        self.linear3 = nn.Linear(270, 54)

    def forward(self, x, random_nums):
        x = torch.nn.functional.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * random_nums.to(device)
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(54, 270)
        self.linear2 = nn.Linear(270, 270)

    def forward(self, z):
        z = nn.functional.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder().to(device)
        self.decoder = Decoder().to(device)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Local_INN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.build_inn()

        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for param in self.trainable_parameters:
            param.data = 0.05 * torch.randn_like(param)
        self.cond_net = self.subnet_cond()
        self.vae = VariationalAutoencoder()

    def subnet_cond(self):
        return nn.Sequential(nn.Linear(COND_DIM, 256), nn.ReLU(),
                             nn.Linear(256, COND_OUT_DIM))

    def build_inn(self):

        def subnet_fc(dim_in, dim_out):
            return nn.Sequential(nn.Linear(dim_in, 1024), nn.ReLU(),
                                    nn.Linear(1024, dim_out))

        nodes = [InputNode(N_DIM, name='input')]
        cond = ConditionNode(COND_OUT_DIM, name='condition')
        for k in range(6):
            nodes.append(Node(nodes[-1],
                                GLOWCouplingBlock,
                                {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                                conditions=cond,
                                name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                                PermuteRandom,
                                {'seed': k},
                                name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))

        return ReversibleGraphNet(nodes + [cond], verbose=False).to(device)

    def forward(self, x, cond):
        return self.model(x, self.cond_net(cond))

    def reverse(self, y_rev, cond):
        return self.model(y_rev, self.cond_net(cond), rev=True)

class Local_INN_Reverse(nn.Module):
    def __init__(self, sample_num):
        super().__init__()
        self.model = Local_INN()
        self.sample_num = sample_num

    def forward(self, scan_t, encoded_cond_t, random_nums1, random_nums2):
        sample_num = self.sample_num
        encode_scan = torch.zeros((sample_num, N_DIM)).to(torch.device(device))
        encode_scan[:, :54] = self.model.vae.encoder.forward(scan_t, random_nums1)
        encode_scan[1:, 54:] = random_nums2
        # copying condition position 20 times i.e shape 20,6
        encoded_cond = encoded_cond_t[None].repeat(sample_num, 1).view(-1, COND_DIM)
        encoded_result = self.model.reverse(encode_scan.to(device), encoded_cond.to(device))[0]
        return encoded_result

class PositionalEncoding():
    def __init__(self, L):
        self.L = L
        self.val_list = []
        for l in range(L):
            self.val_list.append(2.0 ** l)
        self.val_list = np.array(self.val_list)

    def encode(self, x):
        return np.sin(self.val_list * np.pi * x), np.cos(self.val_list * np.pi * x)

    def encode_even(self, x):
        return np.sin(self.val_list * np.pi * 2 * x), np.cos(self.val_list * np.pi * 2 * x)

    def decode(self, sin_value, cos_value):
        atan2_value = np.arctan2(sin_value, cos_value) / (np.pi)
        if np.isscalar(atan2_value) == 1:
            if atan2_value > 0:
                return atan2_value
            else:
                return 1 + atan2_value
        else:
            atan2_value[np.where(atan2_value < 0)] = atan2_value[np.where(atan2_value < 0)] + 1
            return atan2_value

    def decode_even(self, sin_value, cos_value):
        atan2_value = np.arctan2(sin_value, cos_value) / np.pi/2
        if np.isscalar(atan2_value) == 1:
            if atan2_value < 0:
                atan2_value = 1 + atan2_value
            if np.abs(atan2_value - 1) < 0.001:
                atan2_value = 0
        else:
            atan2_value[np.where(atan2_value < 0)] = atan2_value[np.where(atan2_value < 0)] + 1
            atan2_value[np.where(np.abs(atan2_value - 1) < 0.001)] = 0
        return atan2_value

class DataProcessor():
    def __init__(self) -> None:
        pass

    def two_pi_warp(self, angles):
        twp_pi = 2 * np.pi
        return (angles + twp_pi) % (twp_pi)
        # twp_pi = 2 * np.pi
        # if angle > twp_pi:
        #     return angle - twp_pi
        # elif angle < 0:
        #     return angle + twp_pi
        # else:
        #     return angle

    def data_normalize(self, data):
        data_min = np.min(data)
        data = data - data_min
        data_max = np.max(data)
        data = data / data_max
        return data, [data_max, data_min]

    def runtime_normalize(self, data, params):
        return (data - params[1]) / params[0]

    def de_normalize(self, data, params):
        return data * params[0] + params[1]


class Local_INN_TRT_Runtime():
    def __init__(self, sample_num) -> None:
        self.data_proc = DataProcessor()

        self.p_encoding_c = PositionalEncoding(L = 1)
        self.p_encoding = PositionalEncoding(L = 10)
        # self.p_encoding_torch = PositionalEncoding_torch(L = 10)

        self.sample_num = sample_num

        self.device = device

        self.local_inn_reverse = Local_INN_Reverse(self.sample_num)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_directory, 'local_INN_model_best_nov1.pt')
        self.local_inn_reverse.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.local_inn_reverse.to(self.device)
        self.local_inn_reverse.eval()
        print('Model loaded.')

    def reverse(self, scan, prev_state):
        sample_num = self.sample_num
        ## normalize scan and sample the latent space
        scan_np = np.array(scan)

        ## normalize prev_state
        prev_state_normal = np.array(prev_state).copy()
        #prev_state_normal[0] = self.data_proc.runtime_normalize(prev_state[0], self.c.d['normalization_x'])
        #prev_state_normal[1] = self.data_proc.runtime_normalize(prev_state[1], self.c.d['normalization_y'])
        #prev_state_normal[2] = self.data_proc.runtime_normalize(prev_state[2], self.c.d['normalization_theta'])
        #prev_state_normal = np.round(prev_state_normal, decimals=1)

        ## encode the prev_state
        #encoded_cond = []
        #for k in range(3):
         #   if k == 2:
         #       sine_part, cosine_part = self.p_encoding_c.encode_even(prev_state_normal[k])
          #  else:
          #      sine_part, cosine_part = self.p_encoding_c.encode(prev_state_normal[k])
          #  encoded_cond.append(sine_part)
           # encoded_cond.append(cosine_part)
       # encoded_cond = np.array(encoded_cond)
       # encoded_cond = encoded_cond.flatten('F')

        random_nums1 = np.random.default_rng().normal(size=(sample_num, 54))
        random_nums2 = np.random.default_rng().normal(size=(sample_num-1, 6))



        scan_t = torch.from_numpy(np.array(scan_np)).type('torch.FloatTensor').to(self.device)
        encoded_cond = torch.from_numpy(prev_state_normal).type('torch.FloatTensor').to(self.device)
        random_nums1 = torch.from_numpy(random_nums1).type('torch.FloatTensor').to(self.device)
        random_nums2 = torch.from_numpy(random_nums2).type('torch.FloatTensor').to(self.device)

        encoded_result = self.local_inn_reverse(scan_t, encoded_cond, random_nums1, random_nums2)
        encoded_result = encoded_result.cpu().detach().numpy()

        ## decode and de-normalize the results
        results = np.zeros([sample_num, 3])
        results[:, 0] = self.p_encoding.decode(encoded_result[:, 0], encoded_result[:, 1])
        results[:, 1] = self.p_encoding.decode(encoded_result[:, 2], encoded_result[:, 3])
        results[:, 2] = self.p_encoding.decode_even(encoded_result[:, 4], encoded_result[:, 5])

        results[:, 0] = self.data_proc.de_normalize(results[:, 0], [ x_length, x_low])
        results[:, 1] = self.data_proc.de_normalize(results[:, 1], [ y_length, y_low])
        results[:, 2] = results[:, 2]*2*np.pi


        ## find the average
        #if 1:
            #result = np.zeros(3)
#             average_angle = np.arctan2(np.mean(np.sin(results[:, 2])), np.mean(np.cos(results[:, 2])))
#             if average_angle < 0:
#                 average_angle += np.pi * 2
#             result[2] = average_angle
            #angles =results[:, 2] # np.arctan2(np.sin(results[:, 2]), np.cos(results[:, 2]))
            #angles[angles<0] += np.pi * 2
            #result[2] = np.median(angles, axis=0)
            #if( result[2] <0):
                #result[2] += 2 * np.pi
            #result = np.median(results, axis=0)
        if 1:
            result = np.zeros(3)
            average_angle = np.arctan2(np.median(np.sin(results[:, 2])), np.median(np.cos(results[:, 2])))
            #average_angle = -average_angle  #temporary fix
            if average_angle < 0:
                average_angle += np.pi * 2
            result[2] = average_angle
            result[:2] = np.median(results[:, :2], axis=0)
        else:
            result = results[0]

        return result, results

local_inn = Local_INN_TRT_Runtime(40)

def predict_from_model( prev_pose, scan):
    result_arr = np.empty((0, 3))
    #local_inn = Local_INN_TRT_Runtime(40)
    #prev_pos = path_data[0,60+270:60+270+6]
    scan = np.array(scan)
    prev_pose = np.array(prev_pose)

    #UNCOMMENT THIS CODE
    #clip and normalize the scan
    scan = np.clip(scan, 0.2, 10)
    scan =  (scan - 0.2)/ (10-0.2)

    #normalize the conditional data
    prev_pose[0] = (prev_pose[0]-x_low)/x_length
    prev_pose[1] = (prev_pose[1]-y_low)/y_length
    prev_pose[2] = prev_pose[2]/(2* np.pi)

    # divide the condition input to 1000 regions
    prev_pose = np.floor(prev_pose* 10) / 10

    # encode the prev state
    position_data_cond = prev_pose
    p2 = PositionalEncoding(1)
    encode_cond_pose =[]
    for k in range(3):
      if k == 2:
        sine_part_cond, cosine_part_cond = p2.encode_even(position_data_cond[k])
      else:
        sine_part_cond, cosine_part_cond  = p2.encode(position_data_cond[k])

      encode_cond_pose.append(sine_part_cond)
      encode_cond_pose.append(cosine_part_cond)

    encode_cond_pose = np.array(encode_cond_pose)
    encode_cond_pose = encode_cond_pose.flatten('F')

    inferred_state, inferred_states = local_inn.reverse(scan, encode_cond_pose)
    return inferred_state


import rospy
from local_inn.srv import pose_communication, pose_communicationResponse
from std_msgs.msg import Float32MultiArray

def handle_lidar_request(req):
    #rospy.loginfo("Received LIDAR data for pose inference")

    prev_pose = req.lidar_data[0:3]
    scan = req.lidar_data[3:]
    inferred_position = predict_from_model( prev_pose, scan)
    
    # Return the response with the inferred pose
    return pose_communicationResponse(inferred_position.tolist())

def inferred_pose_server():
    rospy.init_node('local_inn_server', anonymous=True)

    # Create the service server
    rospy.Service('compute_inferred_pose_service', pose_communication, handle_lidar_request)

    rospy.loginfo("pose_communication Service Ready")
    rospy.spin()

if __name__ == '__main__':
    inferred_pose_server()

 