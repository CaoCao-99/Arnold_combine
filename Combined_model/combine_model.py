import os
import time
import hydra
import torch
import numpy as np
import torch.nn as nn
##########################################################################################
# import pyrender
# import trimesh
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# import open3d as o3d
# from math import floor
# from pyrender.trackball import Trackball
# from scipy.spatial.transform import Rotation as R
# from utils.transforms import create_pcd_hardcode
# from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('Agg') 
# TASK_OFFSET_BOUNDS = [-0.63, 0, -0.63, 0.63, 1.26, 0.63]
# CAMERAS = ['front', 'base', 'left', 'wrist_bottom', 'wrist']
# SINGLE_CAMERA = 'left'
# DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
# SAVE_IMG = 10000
##########################################################################################
from torch.utils.tensorboard import SummaryWriter
from dataset import ArnoldDataset, ArnoldMultiTaskDataset, InstructionEmbedding
from peract.agent import CLIP_encoder, T5_encoder, RoBERTa, PerceiverIO, PerceiverActorAgent
from peract.utils import point_to_voxel_index, normalize_quaternion, quaternion_to_discrete_euler




class CombinedModel(nn.Module):
    def __init__(self, cfg, device, grasp_model_path, manipulate_model_path):
        super(CombinedModel, self).__init__()
        # 모델의 구조 정의
        from train_peract import create_agent, create_lang_encoder
        self.grasp_model = create_agent(cfg, device=device)  # 집는 모델
        self.grasp_model.load_model(grasp_model_path)
        self.manipulate_model = create_agent(cfg, device=device)  # 조작하는 모델
        self.manipulate_model.load_model(manipulate_model_path)
        self.lang_encoder = create_lang_encoder(cfg, device=device)
        self.lang_embed_cache = InstructionEmbedding(self.lang_encoder)

    def forward(self, x, task_type):
        if task_type == 'Grasp':
            return self.grasp_model(x), self.lang_embed_cache
        elif task_type == 'Manipulate':
            return self.manipulate_model(x), self.lang_embed_cache
        else:
            raise ValueError("Unknown task type")
        
    def save_model(self, path, iteration):
        torch.save({
            'iteration': iteration,
            'grasp_model_state_dict': self.grasp_model._q.state_dict(),
            'grasp_optimizer_state_dict': self.grasp_model._optimizer.state_dict(),
            'manipulate_model_state_dict': self.manipulate_model._q.state_dict(),
            'manipulate_optimizer_state_dict': self.manipulate_model._optimizer.state_dict(),
            }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.grasp_model._q.load_state_dict(checkpoint['grasp_model_state_dict'])
        self.grasp_model._optimizer.load_state_dict(checkpoint['grasp_optimizer_state_dict'])
        self.manipulate_model._q.load_state_dict(checkpoint['manipulate_model_state_dict'])
        self.manipulate_model._optimizer.load_state_dict(checkpoint['manipulate_optimizer_state_dict'])
        return checkpoint['iteration']
        
    
# CombinedModel 저장 및 로드 예시
def save_combined_model(cfg, device, grasp_model_path, manipulate_model_path, combined_model_path, iteration):
    combined_model = CombinedModel(cfg, device, grasp_model_path, manipulate_model_path)
    combined_model.save_model(combined_model_path, iteration)

def load_combined_model(cfg, device, combined_model_path, grasp_model_path, manipulate_model_path):
    combined_model = CombinedModel(cfg, device, grasp_model_path, manipulate_model_path)
    iteration = combined_model.load_model(combined_model_path)
    combined_model.to(device)
    return combined_model, iteration
# # 모델 정의
# combined_model = CombinedModel()

# # 저장된 모델 로드
# grasp_checkpoint = torch.load('grasp_model.pth')
# manipulate_checkpoint = torch.load('manipulate_model.pth')

# # 두 모델의 파라미터를 CombinedModel에 할당
# combined_model.grasp_model.load_state_dict(grasp_checkpoint['model_state_dict'])
# combined_model.manipulate_model.load_state_dict(manipulate_checkpoint['model_state_dict'])

# # CombinedModel 저장
# torch.save({
#     'model_state_dict': combined_model.state_dict(),
# }, 'combined_model.pth')
