"""
For example, run:
    Single-task:
        python train_peract.py task=pickup_object model=peract lang_encoder=clip \
                               mode=train batch_size=8 steps=100000
    Multi-task:
        python train_peract.py task=multi model=peract lang_encoder=clip \
                               mode=train batch_size=8 steps=200000
"""

import os
import time
import hydra
import torch
import numpy as np
##########################################################################################
import pyrender
import trimesh
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import open3d as o3d
from math import floor
from pyrender.trackball import Trackball
from scipy.spatial.transform import Rotation as R
from utils.transforms import create_pcd_hardcode
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
TASK_OFFSET_BOUNDS = [-0.63, 0, -0.63, 0.63, 1.26, 0.63]
CAMERAS = ['front', 'base', 'left', 'wrist_bottom', 'wrist']
SINGLE_CAMERA = 'left'
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
SAVE_IMG = 5000
##########################################################################################
from torch.utils.tensorboard import SummaryWriter
from dataset import ArnoldDataset, ArnoldMultiTaskDataset, InstructionEmbedding
from peract.agent import CLIP_encoder, T5_encoder, RoBERTa, PerceiverIO, PerceiverActorAgent
from peract.utils import point_to_voxel_index, normalize_quaternion, quaternion_to_discrete_euler


"""
하단의 코드는 새로 추가한 것(inference.ipynb)




"""
#################
# visualization tools for peract

def _compute_initial_camera_pose(scene, view_point):
    # Adapted from:
    # https://github.com/mmatl/pyrender/blob/master/pyrender/viewer.py#L1032
    # cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
    # cp[:3, :3] = np.array([[-1, 0, 0], [0, -s2, -s2], [0, -s2, s2]])
    # cp[:3, :3] = np.array([[0, 0, 1.0], [0, -1.0, 0], [1.0, 0, 0]])
    # cp[:3, :3] = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    # cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
    # cp[:3, 3] = dist * np.array([0, -1, 1]) + centroid
    # cp[:3, 3] = dist * np.array([1.5, 0, 0.25]) + centroid
    # cp[:3, 3] = dist * np.array([0, 0, 1]) + centroid
    centroid = scene.centroid
    scale = scene.scale
    scale = 4.0
    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)

    if view_point == 'original':
        cp[:3, :3] = np.array([[0, 0, -1.0], [0, -1.0, 0], [-1.0, 0, 0]])
        hfov = np.pi / 6.0
        dist = scale / (2.0 * np.tan(hfov))
        cp[:3, 3] = dist * np.array([-1, 0, 0]) + centroid
    elif view_point == 'left':
        cp[:3, :3] = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        hfov = np.pi / 6.0
        dist = scale / (2.0 * np.tan(hfov))
        cp[:3, 3] = dist * np.array([0, 0, -1]) + centroid
    elif view_point == 'right':  # 추가된 부분
        cp[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])  # 우측면을 바라보는 방향으로 변경
        hfov = np.pi / 6.0
        dist = scale / (2.0 * np.tan(hfov))
        cp[:3, 3] = dist * np.array([-1, 0, 0]) + centroid  # 카메라 위치를 우측면에서 바라볼 수 있도록 조정
    return cp


def _from_trimesh_scene(
        trimesh_scene, bg_color=None, ambient_light=None):
    # convert trimesh geometries to pyrender geometries
    geometries = {name: pyrender.Mesh.from_trimesh(geom, smooth=False)
                  for name, geom in trimesh_scene.geometry.items()}
    # create the pyrender scene object
    scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    # add every node with geometry to the pyrender scene
    for node in trimesh_scene.graph.nodes_geometry:
        pose, geom_name = trimesh_scene.graph[node]
        scene_pr.add(geometries[geom_name], pose=pose)
    return scene_pr


def _create_bounding_box(scene, voxel_size, res):
    l = voxel_size * res
    T = np.eye(4)
    w = 0.01
    for trans in [[0, 0, l / 2], [0, l, l / 2], [l, l, l / 2], [l, 0, l / 2]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [w, w, l], T, face_colors=[0, 0, 0, 255]))
    for trans in [[l / 2, 0, 0], [l / 2, 0, l], [l / 2, l, 0], [l / 2, l, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [l, w, w], T, face_colors=[0, 0, 0, 255]))
    for trans in [[0, l / 2, 0], [0, l / 2, l], [l, l / 2, 0], [l, l / 2, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [w, l, w], T, face_colors=[0, 0, 0, 255]))


def create_voxel_scene(
        voxel_grid: np.ndarray,
        q_attention: np.ndarray = None,
        highlight_coordinate: np.ndarray = None,
        highlight_gt_coordinate: np.ndarray = None,
        highlight_alpha: float = 1.0,
        voxel_size: float = 0.1,
        show_bb: bool = False,
        alpha: float = 0.5):
    _, d, h, w = voxel_grid.shape
    v = voxel_grid.transpose((1, 2, 3, 0))
    occupancy = v[:, :, :, -1] != 0
    alpha = np.expand_dims(np.full_like(occupancy, alpha, dtype=np.float32), -1)
    rgb = np.concatenate([(v[:, :, :, 3:6] + 1)/ 2.0, alpha], axis=-1)

    if q_attention is not None:
        q = np.max(q_attention, 0)
        q = q / np.max(q)
        show_q = (q > 0.75)
        occupancy = (show_q + occupancy).astype(bool)
        q = np.expand_dims(q - 0.5, -1)  # Max q can be is 0.9
        q_rgb = np.concatenate([
            q, np.zeros_like(q), np.zeros_like(q),
            np.clip(q, 0, 1)], axis=-1)
        rgb = np.where(np.expand_dims(show_q, -1), q_rgb, rgb)

    if highlight_coordinate is not None:
        x, y, z = highlight_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [0.0, 1.0, 0.0, highlight_alpha] #예측 : 초록

    if highlight_gt_coordinate is not None:
        x, y, z = highlight_gt_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [1.0, 0.0, 0.0, highlight_alpha] #GT : 빨강

    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size, translate=(0.0, 0.0, 0.0))
    trimesh_voxel_grid = trimesh.voxel.VoxelGrid(
        encoding=occupancy, transform=transform)
    geometry = trimesh_voxel_grid.as_boxes(colors=rgb)
    scene = trimesh.Scene()
    scene.add_geometry(geometry)
    if show_bb:
        assert d == h == w
        _create_bounding_box(scene, voxel_size, d)
    return scene


def visualise_voxel(view_point:str,
                    save_path:str,
                    voxel_grid: np.ndarray,
                    q_attention: np.ndarray = None,
                    highlight_coordinate: np.ndarray = None,
                    highlight_gt_coordinate: np.ndarray = None,
                    highlight_alpha: float = 1.0,
                    rotation_amount: float = 0.0,
                    show: bool = False,
                    voxel_size: float = 0.1,
                    offscreen_renderer: pyrender.OffscreenRenderer = None,
                    show_bb: bool = True,                       #boundingbox
                    alpha: float = 0.5,
                    render_gripper=True,
                    gripper_pose=None,
                    gt_gripper_pose=None,
                    gripper_mesh_scale=1.0):
    scene = create_voxel_scene(
        voxel_grid, q_attention, highlight_coordinate, highlight_gt_coordinate,
        highlight_alpha, voxel_size,
        show_bb, alpha)
    print(scene)
    print(type(scene))
    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=1080, viewport_height=1080, point_size=1.0)
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8],
            bg_color=[1.0, 1.0, 1.0])
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width/r.viewport_height)
        p = _compute_initial_camera_pose(s, view_point)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)

        if render_gripper:
            gripper_trimesh = trimesh.load('C:\Users\USER\Documents\GitHub\Arnold_multistep\hand.dae', force='mesh')
            gripper_trimesh.vertices *= gripper_mesh_scale
            radii = np.linalg.norm(gripper_trimesh.vertices - gripper_trimesh.center_mass, axis=1)
            gripper_trimesh.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale, color_map='winter')
            gripper_mesh = pyrender.Mesh.from_trimesh(gripper_trimesh, poses=np.array([gripper_pose]), smooth=False)
            s.add(gripper_mesh)
            
            gripper_trimesh = trimesh.load('C:\Users\USER\Documents\GitHub\Arnold_multistep\hand2.dae', force='mesh')
            gripper_trimesh.vertices *= gripper_mesh_scale
            radii = np.linalg.norm(gripper_trimesh.vertices - gripper_trimesh.center_mass, axis=1)
            gripper_trimesh.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale, color_map='winter')
            gripper_mesh = pyrender.Mesh.from_trimesh(gripper_trimesh, poses=np.array([gt_gripper_pose]), smooth=False)
            s.add(gripper_mesh)
        color, depth = r.render(s)
        return color.copy()


def get_gripper_render_pose(voxel_scale, scene_bound_origin, continuous_trans, continuous_quat):
	# finger tip to gripper offset
	offset = np.array([[1, 0, 0, 0],
	                   [0, 1, 0, 0],
	                   [0, 0, 1, 0.1*voxel_scale],
	                   [0, 0, 0, 1]])


	# scale and translate by origin
	translation = (continuous_trans - (np.array(scene_bound_origin[:3]))) * voxel_scale
	mat = np.eye(4,4)
	mat[:3,:3] = R.from_quat([continuous_quat[0], continuous_quat[1], continuous_quat[2], continuous_quat[3]]).as_matrix()
	offset_mat = np.matmul(mat, offset)
	mat[:3,3] = translation - offset_mat[:3,3]
	return mat


def visualize_voxel_grid(language_instructions, img_name ,timestep, save_folder, voxel_grid, q_trans=None, pred_vox=None, gt_vox=None, render_gripper=False, pred_pos=None, pred_quat=None, gt_pos=None, gt_quat=None, rotation_amount=180):
    voxel_size = 0.02
    voxel_scale = voxel_size * 100
    if render_gripper:
        gripper_pose_mat = get_gripper_render_pose(
            voxel_scale, TASK_OFFSET_BOUNDS[:3], pred_pos, pred_quat
        )
        gt_gripper_pose_mat = get_gripper_render_pose(
            voxel_scale, TASK_OFFSET_BOUNDS[:3], gt_pos, gt_quat
        )
    # elif gt_render_gripper:
    #     gt_gripper_pose_mat = get_gripper_render_pose(
    #         voxel_scale, TASK_OFFSET_BOUNDS[:3], gt_pos, gt_quat
    #     )
    #     gripper_pose_mat = None
    # elif render_gripper:
    #     gripper_pose_mat = get_gripper_render_pose(
    #         voxel_scale, TASK_OFFSET_BOUNDS[:3], pred_pos, pred_quat
    #     )
    #     gt_gripper_pose_mat = None
    else:
        gt_gripper_pose_mat = None
        gripper_pose_mat = None
    view_points = ['original', 'left']#['front', 'side', 'back']
    for i in view_points:
        save_path = os.path.join(save_folder, f'{i}.png')

        rendered_img = visualise_voxel(view_point=i, save_path=save_path,
            voxel_grid=voxel_grid, q_attention=q_trans, highlight_coordinate=pred_vox, highlight_gt_coordinate=gt_vox, voxel_size=voxel_size, show_bb=True,
            rotation_amount=np.deg2rad(rotation_amount), render_gripper=render_gripper, gripper_pose=gripper_pose_mat, gt_gripper_pose=gt_gripper_pose_mat, gripper_mesh_scale=voxel_scale
        )
        # Ensure rendered_img has the correct data type
        # rendered_img = rendered_img.astype(np.float32)
        plt.figure(figsize=(15,15))
        plt.imshow(rendered_img)
        # 이미지에 텍스트 추가
        plt.text(
            10, 10, 
            f"Action: act{timestep+1}\n"
            f"language_instructions: {language_instructions}",
            color='black', 
            fontsize=12, 
            ha='left', 
            va='top'
        )
        plt.axis('off')
        # Save the image if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()



@torch.no_grad()
def predict(batch_number, inp, agent, points, colors, instruction, lang_embed_cache, timestep, camera):
    lang_goal_embs = lang_embed_cache.get_lang_embed(instruction)

    if timestep == 0:
        gripper_open = True
        gripper_joint_positions = [0.04, 0.04]
    elif timestep == 1:
        gripper_open = False
        gripper_joint_positions = [0, 0]
    low_dim_state = np.array([gripper_open, *gripper_joint_positions, timestep]).reshape(1, -1)

    input_dict = {
        'lang_goal_embs': lang_goal_embs,
        'low_dim_state': low_dim_state,
    }

    obs_dict = {}
    print(f'receive observation at shape {points.shape}')
    if camera == 'single':
        if points.ndim == 2:
            points = points[None, None, ...]
            colors = colors[None, None, ...]
        elif points.ndim == 3:
            points = points[None, ...]
            colors = colors[None, ...]
        for n in CAMERAS:
            if n == SINGLE_CAMERA:
                obs_dict[f'{n}_rgb'] = colors.transpose(0, 3, 1, 2)   # peract requires input as CHW
                obs_dict[f'{n}_point_cloud'] = points.transpose(0, 3, 1, 2)   # peract requires input as CHW
            else:
                obs_dict[f'{n}_rgb'] = np.zeros((1, 3, 0, 0))   # peract requires input as CHW
                obs_dict[f'{n}_point_cloud'] = np.zeros((1, 3, 0, 0))   # peract requires input as CHW
    else:
        for n, rgb, pcd in zip(CAMERAS, colors, points):
            rgb = rgb[None, ...]
            pcd = pcd[None, ...]
            obs_dict[f'{n}_rgb'] = rgb.transpose(0, 3, 1, 2)   # peract requires input as CHW
            obs_dict[f'{n}_point_cloud'] = pcd.transpose(0, 3, 1, 2)   # peract requires input as CHW

    input_dict.update(obs_dict)
    # import pdb
    # pdb.set_trace()
    for k, v in input_dict.items():
        if v is not None:
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(v)
            input_dict[k] = v.to(DEVICE)


    output_dict = agent.predict(inp)
    # import pdb
    # pdb.set_trace()
    #output_dict = output_dict[batch_number]
    target_voxel = output_dict['pred_action']['trans'][batch_number].detach().cpu().numpy()
    # import pdb
    # pdb.set_trace()
    # gt_voxel = output_dict['pred_action']['trans'][0].detach().cpu().numpy()
    position = output_dict['pred_action']['continuous_trans'][batch_number].detach().cpu().numpy()
    print(f"timestep {timestep}: {target_voxel}, {position}")
    rotation = output_dict['pred_action']['continuous_quat'][batch_number]
    # rotation = rotation[[3,0,1,2]]   # xyzw to wxyz
    return target_voxel, position, rotation, output_dict['voxel_grid'][batch_number].detach().cpu().numpy()


def visualize_pos(scene_pcd, pos,save_path):
    cube_points = [
        pos + np.array([-0.05, -0.05, -0.05]),
        pos + np.array([0.05, -0.05, -0.05]),
        pos + np.array([-0.05, 0.05, -0.05]),
        pos + np.array([0.05, 0.05, -0.05]),
        pos + np.array([-0.05, -0.05, 0.05]),
        pos + np.array([0.05, -0.05, 0.05]),
        pos + np.array([-0.05, 0.05, 0.05]),
        pos + np.array([0.05, 0.05, 0.05])
    ]
    cube_lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7]
    ]
    cube = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cube_points),
        lines=o3d.utility.Vector2iVector(cube_lines)
    )
    cube.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(cube_lines))])
    # o3d.visualization.draw_geometries([scene_pcd, cube])
    if save_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(scene_pcd)
        vis.add_geometry(cube)
        vis.update_renderer()
        vis.poll_events()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([scene_pcd, cube])

def act_on_demo_step(language_instructions, batch_number, inp, gt_vox, gt_pos, gt_quat, save_folder, step, agent, lang_embed_cache, timestep, full_obs=True):
    robot_forward_direction = R.from_quat(step['robot_base'][1][[1,2,3,0]]).as_matrix()[:, 0]
    robot_forward_direction[1] = 0
    robot_forward_direction = robot_forward_direction / np.linalg.norm(robot_forward_direction) * 0.5   # m
    bound_center = step['robot_base'][0] / 100 + robot_forward_direction

    instruction = step['instruction']
    print(f'Instruction: {instruction}')

    imgs = step['images']
    points = []
    colors = []
    i=0
    for camera_obs in imgs:
        camera = camera_obs['camera']
        color = camera_obs['rgb'][:, :, :3]   # int [0, 255]
        depth = np.clip(camera_obs['depthLinear'], 0, 10)
        point_cloud = create_pcd_hardcode(camera, depth, cm_to_m=True)
        # here point_cloud is y-up and in meter
        point_cloud = point_cloud - bound_center
        # supposed to be located in TASK_OFFSET_BOUNDS
        points.append(point_cloud)

        colors.append(color)
        plt.figure()
        # show each partial obs
        plt.subplot(1, 2, 1)
        plt.imshow(color)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(depth / 10)
        plt.axis('off')
        save_path = os.path.join(save_folder, f'{CAMERAS[i]}_rgbd.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        i+=1
    points = np.stack(points)
    points[..., 0::2] = np.clip(points[..., 0::2], -0.63, 0.63)
    points[..., 1] = np.clip(points[..., 1], 0, 1.26)
    colors = np.stack(colors)
    # Create scene for each camera
    # for camera_idx, camera_name in enumerate(CAMERAS):
    #     camera_points = points[camera_idx]
    #     camera_colors = colors[camera_idx]
    #save_path = []
    #save_path = os.path.join(save_folder, 'predict.png')
    #for i in range(batchsize):
    os.makedirs(save_folder, exist_ok=True)
    if full_obs:
        pred_vox, pred_pos, pred_quat, voxel_grid = predict(batch_number, inp, agent, points, colors, instruction, lang_embed_cache, timestep, camera='multi')
    else:
        camera_idx = CAMERAS.index(SINGLE_CAMERA)
        pred_vox, pred_pos, pred_quat, voxel_grid = predict(batch_number, inp, agent, points[camera_idx], colors[camera_idx], instruction, lang_embed_cache, timestep, camera='single')

    print(f'absolute position: {pred_pos + bound_center}')
    ############################################# gt_vox : 추가해야 함
    # img_name = 'original'
    # visualize_voxel_grid(timestep, save_folder, voxel_grid, q_trans=None, pred_vox=pred_vox, gt_vox = gt_vox, gt_render_gripper=False, render_gripper=True, pred_pos=pred_pos, pred_quat=pred_quat, gt_pos=gt_pos, gt_quat=gt_quat) #render_gripper=True
    
    
    
    img_name = 'change'
    visualize_voxel_grid(language_instructions, img_name, timestep, save_folder, voxel_grid, q_trans=None, pred_vox=pred_vox, gt_vox = gt_vox, render_gripper=True, pred_pos=pred_pos, pred_quat=pred_quat, gt_pos=gt_pos, gt_quat=gt_quat) #render_gripper=True
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    scene_pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3) / 255)
    save_path = os.path.join(save_folder, 'pos.png')
    # visualize_pos(scene_pcd, pred_pos, save_path)


def act_on_ply(ply_path, agent, lang_embed_cache, instruction, timestep, bound_center):
    scene_pcd = o3d.io.read_point_cloud(ply_path)
    # scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.004)

    points = np.asarray(scene_pcd.points)
    colors = np.asarray(scene_pcd.colors)

    # process with bound
    # bound_min = np.min(points, axis=0)
    # bound_max = np.max(points, axis=0)
    # bound_center = (bound_min + bound_max) / 2
    
    points -= np.array(bound_center)
    points[..., 0::2] = np.clip(points[..., 0::2], -0.63, 0.63)
    points[..., 1] = np.clip(points[..., 1], 0, 1.26)
    if colors.max() < 1.5:
        colors = np.clip(colors*255, 0, 255).astype(np.uint8)

    pred_vox, pred_pos, pred_quat, voxel_grid = predict(agent, points, colors, instruction, lang_embed_cache, timestep, camera='single')
    print(f'absolute position: {pred_pos + bound_center}')

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    scene_pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3) / 255)
    visualize_pos(scene_pcd, pred_pos)
    
    
    
    
    
    
    
#####################################


def create_lang_encoder(cfg, device):
    if cfg.lang_encoder == 'clip':
        return CLIP_encoder(device)
    elif cfg.lang_encoder == 't5':
        return T5_encoder(cfg.t5_cfg, device)
    elif cfg.lang_encoder == 'roberta':
        return RoBERTa(cfg.roberta_cfg, device)
    elif cfg.lang_encoder == 'none':
        return None
    else:
        raise ValueError('Language encoder key not supported')


def create_agent(cfg, device):
    perceiver_encoder = PerceiverIO(
        depth=6,
        iterations=1,
        voxel_size=cfg.voxel_size,
        initial_dim=3 + 3 + 1 + 3,
        low_dim_size=4,
        layer=0,
        num_rotation_classes=72,
        num_grip_classes=2,
        num_state_classes=2,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        activation='lrelu',
        input_dropout=0.1,
        attn_dropout=0.1,
        decoder_dropout=0.0,
        voxel_patch_size=5,
        voxel_patch_stride=5,
        final_dim=64,
        lang_embed_dim=cfg.lang_embed_dim[cfg.lang_encoder],
        with_language=(cfg.lang_encoder != 'none')
    )

    peract_agent = PerceiverActorAgent(
        coordinate_bounds=cfg.offset_bound,
        perceiver_encoder=perceiver_encoder,
        camera_names=cfg.cameras,
        batch_size=cfg.batch_size,
        voxel_size=cfg.voxel_size,
        voxel_feature_size=3,
        num_rotation_classes=72,
        rotation_resolution=5,
        lr=0.0001,
        image_resolution=[cfg.img_size, cfg.img_size],
        lambda_weight_l2=0.000001,
        transform_augmentation=False,
        optimizer_type='lamb',
        state_head=cfg.state_head
    )

    peract_agent.build(training=(cfg.mode == 'train'), device=device)

    return peract_agent


def prepare_batch(batch_data, cfg, lang_embed_cache, device):
    obs_dict = {}
    language_instructions = []
    target_points = []
    gripper_open = []
    low_dim_state = []
    current_states = []
    goal_states = []
    for data in batch_data:
        for k, v in data['obs_dict'].items():
            if k not in obs_dict:
                obs_dict[k] = [v]
            else:
                obs_dict[k].append(v)
        
        target_points.append(data['target_points'])
        gripper_open.append(data['target_gripper'])
        language_instructions.append(data['language'])
        low_dim_state.append(data['low_dim_state'])
        current_states.append(data['current_state'])
        goal_states.append(data['goal_state'])

    for k, v in obs_dict.items():
        v = np.stack(v, axis=0)
        obs_dict[k] = v.transpose(0, 3, 1, 2)   # peract requires input as [C, H, W]
    
    bs = len(language_instructions)
    target_points = np.stack(target_points, axis=0)
    gripper_open = np.array(gripper_open).reshape(bs, 1)
    low_dim_state = np.stack(low_dim_state, axis=0)

    current_states = np.array(current_states).reshape(bs, 1)
    goal_states = np.array(goal_states).reshape(bs, 1)
    states = np.concatenate([current_states, goal_states], axis=1)   # [bs, 2]

    trans_action_coords = target_points[:, :3]
    trans_action_indices = point_to_voxel_index(trans_action_coords, cfg.voxel_size, cfg.offset_bound)

    rot_action_quat = target_points[:, 3:]
    rot_action_quat = normalize_quaternion(rot_action_quat)
    rot_action_indices = quaternion_to_discrete_euler(rot_action_quat, cfg.rotation_resolution)
    rot_grip_action_indices = np.concatenate([rot_action_indices, gripper_open], axis=-1)

    lang_goal_embs = lang_embed_cache.get_lang_embed(language_instructions)

    inp = {}
    inp.update(obs_dict)
    inp.update({
        'trans_action_indices': trans_action_indices,
        'rot_grip_action_indices': rot_grip_action_indices,
        'states': states,
        'lang_goal_embs': lang_goal_embs,
        'low_dim_state': low_dim_state
    })
    # import pdb
    # pdb.set_trace()
    for k, v in inp.items():
        if v is not None:
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(v)
            inp[k] = v.to(device)
    
    return inp, language_instructions


@hydra.main(config_path='./configs', config_name='default')
def main(cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = DEVICE
    print("현재 설정된 디바이스:", device)

    if cfg.novel:
        train_dataset = ArnoldDataset(data_path=os.path.join(cfg.novel_state, cfg.task, 'novel_state'), task=cfg.task, obs_type=cfg.obs_type)
    else:
        if cfg.task == 'multi':
                train_dataset = ArnoldMultiTaskDataset(data_root=cfg.data_root, obs_type=cfg.obs_type)
        else:   # 기존의 task 하나에 해당하는 데이터 로드 함수
            train_dataset = ArnoldDataset(data_path=os.path.join(cfg.data_root, cfg.task, 'train'), task=cfg.task, obs_type=cfg.obs_type)
            
        
    writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tb_logs'))

    agent = create_agent(cfg, device=device)

    lang_encoder = create_lang_encoder(cfg, device)
    lang_embed_cache = InstructionEmbedding(lang_encoder)
    save_folder = cfg.save_folder               # 추가한 내용
    os.makedirs(save_folder, exist_ok=True)
    start_step = 0
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print(f'=> loading checkpoint {cfg.resume}')
            start_step = agent.load_model(cfg.resume)
            print(f'=> loaded checkpoint {cfg.resume} (step {start_step})')
        else:
            print(f'=> no checkpoint found at {cfg.resume}')

    # cfg.save_interval = floor(cfg.steps/20)
    cfg.save_interval = 10000
    
    print(f'Training {cfg.steps} steps, {len(train_dataset)} demos, {cfg.batch_size} batch_size')
    start_time = time.time()
    
    for iteration in range(start_step, cfg.steps):
        # train
        batch_data = train_dataset.sample(cfg.batch_size, cfg.steps)
        inp, language_instructions = prepare_batch(batch_data, cfg, lang_embed_cache, device)
        # import pdb
        # pdb.set_trace()
        update_dict = agent.update(iteration, inp)
        running_loss = update_dict['total_loss']
        trans_loss = update_dict['trans_loss']
        rot_grip_loss = update_dict['rot_grip_loss']
        # state_loss = update_dict['state_loss']
        if iteration % cfg.log_interval == 0:
            elapsed_time = (time.time() - start_time) / 60.0
            print(f'Iteration: {iteration} | Total Loss: {running_loss} | Elapsed Time: {elapsed_time} mins')
            writer.add_scalar('trans_loss', trans_loss, iteration)
            writer.add_scalar('rot_grip_loss', rot_grip_loss, iteration)
           #writer.add_scalar('state_loss', state_loss, iteration)
            writer.add_scalar('total_loss', running_loss, iteration)
        if iteration % SAVE_IMG == 0:
            for i in range(len(batch_data)):
                timestep = 1 if batch_data[i]['action_num'] == 2 else 0
                new_save_folder = os.path.join(save_folder, f'{iteration}', f'batch_{i}')
                os.makedirs(new_save_folder, exist_ok=True)
                # import pdb
                # pdb.set_trace()
                gt_vox = update_dict['expert_action']['action_trans'][i].detach().cpu().numpy()
                gt_pos = update_dict['expert_action']['continuous_trans'][i].detach().cpu().numpy()
                gt_quat = update_dict['expert_action']['continuous_quat'][i]
                act_on_demo_step(language_instructions[i], i, inp, gt_vox, gt_pos, gt_quat, new_save_folder, batch_data[i]['step'], agent, lang_embed_cache, timestep, full_obs=True)
        if (iteration+1) % cfg.save_interval == 0:
            # save for model selection
            ckpt_path = os.path.join(cfg.checkpoint_dir, f'peract_{cfg.task}_{cfg.obs_type}_{cfg.lang_encoder}_{iteration+1}.pth')
            print('Saving checkpoint')
            agent.save_model(ckpt_path, iteration)
    
    writer.close()


if __name__ == '__main__':
    main()
