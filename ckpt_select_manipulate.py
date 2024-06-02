"""
Script for model selection. For example, run:
    Single-task:
        python ckpt_selection.py task=pickup_object model=peract lang_encoder=clip \
                                 mode=eval visualize=0
    Multi-task:
        python ckpt_selection.py task=multi model=peract lang_encoder=clip \
                                 mode=eval visualize=0
"""
import hydra
import json
import logging
import numpy as np
import os
import re
import shutil
import torch
import copy
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from environment.runner_utils import get_simulation
simulation_app, simulation_context, _ = get_simulation(headless=True, gpu_id=0)

from dataset import InstructionEmbedding
from tasks import load_task
from utils.env import get_action

logger = logging.getLogger(__name__)


def load_data(data_path):
    demos = list(Path(data_path).iterdir())
    demo_path = sorted([str(item) for item in demos if not item.is_dir()])
    data = []
    fnames = []
    for npz_path in demo_path:
        data.append(np.load(npz_path, allow_pickle=True))
        fnames.append(npz_path)
    return data, fnames


def make_agent(cfg, device):
    lang_embed_cache = None
    if 'peract' in cfg.model:
        from train_peract import create_agent, create_lang_encoder
        agent = create_agent(cfg, device=device)

        lang_encoder = create_lang_encoder(cfg, device=device)
        lang_embed_cache = InstructionEmbedding(lang_encoder)
    
    else:
        raise ValueError(f'{cfg.model} agent not supported')
    
    return agent, lang_embed_cache


def load_ckpt(cfg, agent, ckpt_path, device):
    if cfg.model == 'peract':
        agent.module.load_model(ckpt_path)
    else:
        raise ValueError(f'{cfg.model} agent not supported')
    
    logger.info(f"Loaded {cfg.model} from {ckpt_path}")
    return agent


@hydra.main(config_path='./configs', config_name='default')
def main(cfg):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    #추가 추가
    if not cfg.use_gt[0]:
        agent1, lang_embed_cache1 = make_agent(cfg,device=device)
        agent1 = load_ckpt(cfg, agent1, cfg.checkpoint_dir_grasp, device)
        print(f'cfg.checkpoint_dir_grasp : {cfg.checkpoint_dir_grasp}\n\n')
#    else:
 #       agent1, lang_embed_cache = make_agent(cfg,device=device)
  #      agent1 = load_ckpt(cfg, agent1, cfg.checkpoint_dir_grasp, device)

    ########
    cfg.checkpoint_dir = cfg.checkpoint_dir.split(os.path.sep)
    cfg.checkpoint_dir[-2] = cfg.checkpoint_dir[-2].replace('eval', 'train')
    cfg.checkpoint_dir = os.path.sep.join(cfg.checkpoint_dir)

#    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    render = cfg.visualize
    tasks_task = cfg.task       #New Task add
    offset = cfg.offset_bound
    use_gt = cfg.use_gt
    agent2, lang_embed_cache2 = make_agent(cfg, device=device)
    
    if cfg.task != 'multi':
        task_list = [cfg.task]
    else:
        task_list = [
            'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
            'open_cabinet', 'close_cabinet', 'pour_water', 'transfer_water'
        ]
    #Modify below code
     # ckpts_list = [f for f in os.listdir(cfg.checkpoint_dir) if f in ['peract_multi_rgb_t5_160000.pth','peract_multi_rgb_t5_170000.pth', 'peract_multi_rgb_t5_180000.pth',
     #     'peract_multi_rgb_t5_190000.pth','peract_multi_rgb_t5_200000.pth' ]]

    # cfg.checkpoint_dir에 존재하는 모든 파일을 리스트에 담습니다.
    file_list = os.listdir(cfg.checkpoint_dir)

    # 파일을 숫자로 변환하여 정렬합니다.
    sorted_files = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # 뒤에서 4개의 파일을 선택합니다.
    selected_files = sorted_files[-cfg.num_ckpt:]

    print(selected_files)

#    pattern = re.compile(r'.*(160000|170000|180000|190000|200000)\.pth$')
 #   ckpts_list = [f for f in os.listdir(cfg.checkpoint_dir) if pattern.match(f)]
    ckpts_list = selected_files
    print(ckpts_list)
    for ckpt_name in ckpts_list:
        if 'best' in ckpt_name:
            logger.info('Best checkpoint already recognized')
            simulation_app.close()
            return 1
    ckpts_list = sorted(ckpts_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    log_path = os.path.join(cfg.exp_dir, 'select_log.json')
    """
    val log structure:
    {
        'ckpt_name': {
            'task': {
                # 'stats': {
                #     'fname': int (1, 0, -1)
                # },
                'score': float
            }
        }
    }
    """

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            val_log = json.load(f)
    else:
        val_log = {}
    
    for ckpt_name in ckpts_list:
        print(f'ckpt_name : {ckpt_name}\n\n')
        agent2 = load_ckpt(cfg, agent2, os.path.join(cfg.checkpoint_dir, ckpt_name), device)
        if ckpt_name not in val_log:
            val_log[ckpt_name] = {}
        for task_name in task_list:
            if task_name not in val_log[ckpt_name]:
                val_log[ckpt_name][task_name] = {}
            elif 'score' in val_log[ckpt_name][task_name]:
                continue
            number = ckpt_name.split("_")[-1].replace(".pth", "")
            file_path = os.path.join(cfg.output_folder, number, f'{task_name}.txt')
            # 폴더가 존재하지 않는 경우 폴더 생성
            folder_path = os.path.dirname(file_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(file_path, 'w') as file:
                file.write(f'Task 명: {task_name}\n')
                file.write('_' * 70 + '\n')  # 구분선 추가
            print(f'Plus Information : {file_path}\n')
            logger.info(f'Evaluating {ckpt_name} {task_name}')
            data, fnames = load_data(data_path=os.path.join(cfg.data_root, task_name, 'val'))
            correct = 0
            total = 0
            with open(file_path, 'a') as file:
                file.write(f'Sampling Scenes num: {len(fnames)}\n')
                for i in fnames:
                    file.write(f'Sampling Scene 명: {i}\n')

            while len(data) > 0:
                fff = fnames.pop(0)
                print(f'Now Scene : {fff} \n')
                with open(file_path, 'a') as file:
                    file.write(f'Now Scene : {fff} \n')
                anno = data.pop(0)
                gt_frames = anno['gt']
                robot_base = gt_frames[0]['robot_base']
                gt_actions = [
                    gt_frames[1]['position_rotation_world'], gt_frames[2]['position_rotation_world'],
                    gt_frames[3]['position_rotation_world'] if 'water' not in task_name \
                    else (gt_frames[3]['position_rotation_world'][0], gt_frames[4]['position_rotation_world'][1])
                ]

                env, object_parameters, robot_parameters, scene_parameters = load_task(cfg.asset_root, npz=anno, cfg=cfg)

                obs = env.reset(robot_parameters, scene_parameters, object_parameters, 
                                robot_base=robot_base, gt_actions=gt_actions)

                logger.info(f'Instruction: {gt_frames[0]["instruction"]}')
                logger.info('Ground truth action:')
                for gt_action, grip_open in zip(gt_actions, cfg.gripper_open[task_name]):
                    act_pos, act_rot = gt_action
                    act_rot = R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)
                    logger.info(f'trans={act_pos}, orient(euler XYZ)={act_rot}, gripper_open={grip_open}')

                try:
                    for i in range(2):
                        if use_gt[i]:
                           # import pdb
                            #pdb.set_trace()
                            obs, suc = env.step(act_pos=None, act_rot=None, render=render, use_gt=True, file_path=file_path)
                            import pdb
                            #pdb.set_trace()
                        else:
                            #select_agent = copy.copy(agent1)
                            #select_lang_cache = copy.copy(lang_embed_cache1)
                            #print(f'ckpt_name : {ckpt_name}\n\n')
                            #if i == 0:
                               # select_agent = copy.copy(agent1)
                               # select_lang_cache = copy.copy(lang_embed_cache1)
                           # else:
                               # select_agnet = copy.copy(agent2)
                               # select_lang_cache = copy.copy(lang_embed_cache2)
                            import pdb
                            #pdb.set_trace()
                            if i == 0:
                                act_pos, act_rot = get_action(
                                    gt=obs, agent=agent1, franka=env.robot, c_controller=env.c_controller, npz_file=anno, offset=offset, timestep=i,
                                    device=device, agent_type=cfg.model, obs_type=cfg.obs_type, lang_embed_cache=lang_embed_cache1,   file_path=file_path
                                )
                            else:
                                act_pos, act_rot = get_action(
                                    gt=obs, agent=agent2, franka=env.robot, c_controller=env.c_controller, npz_file=anno, offset=offset, timestep=i,
                                    device=device, agent_type=cfg.model, obs_type=cfg.obs_type, lang_embed_cache=lang_embed_cache2,   file_path=file_path
                                )

                            logger.info(
                                f"Prediction action {i}: trans={act_pos}, orient(euler XYZ)={R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)}"
                            )

                            obs, suc = env.step(act_pos=act_pos, act_rot=act_rot, render=render, use_gt=False, file_path=file_path)
                            import pdb
                            #pdb.set_trace()
                        if suc == -1:
                            break
                
                except:
                    suc = -1
                    with open(file_path, 'a') as file:
                        file.write(f"성공 여부: {True if suc == 1  else False}\n")

                env.stop()
                if suc == 1:
                    correct += 1
                total += 1
                log_str = f'correct: {correct} | total: {total} | remaining: {len(data)}'
                logger.info(f'{log_str}\n')
                print(f"성공 여부: {True if suc == 1  else False}\n")
                with open(file_path, 'a') as file:
                        #                file.write(f"scene : {fname}\n") # scene 이름
                    file.write(f"성공 여부: {True if suc == 1  else False}\n")

            logger.info(f'{ckpt_name} {task_name}: {correct/total*100:.2f}\n\n')
            val_log[ckpt_name][task_name]['score'] = correct / total
            with open(file_path, 'a') as file:
                file.write(f"score: {correct/total*100:.2f}\n\n")    #Score

            with open(log_path, 'w') as f:
                json.dump(val_log, f, indent=2)

    ckpt_scores = [np.mean([val_log[ckpt_name][task_name]['score'] for task_name in task_list]) for ckpt_name in ckpts_list]
    import pdb
    #pdb.set_trace()
   # selected_idx = ckpt_scores.argmax()
   # selected_name = ckpts_list[selected_idx]
    selected_score = max(ckpt_scores)
    selected_idx = ckpt_scores.index(selected_score)
    selected_name = ckpts_list[selected_idx]
   # file_path = os.path.join(cfg.output_folder, number, f'{task_name}.txt')
    best_file_path = os.path.join(cfg.best_dir,tasks_task,'best.txt')
    folder_path2 = os.path.dirname(best_file_path)
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    with open(best_file_path, 'w') as file:
        file.write(f'Test for {tasks_task}\n')    #Score
        file.write('_' * 70 + '\n')  # 구분선 추가
    i=0
    for ckpt_name in ckpts_list:
        with open(best_file_path, 'a') as file:
            file.write(f'Test for {ckpt_name} : {ckpt_scores[i]} \n')    #Score
            i+=1
#            file.write('_' * 70 + '\n')  # 구분선 추가

        if ckpt_name == selected_name:
            new_name = ckpt_name.split('_')
            new_name[-1] = 'best.pth'
            new_name = '_'.join(new_name)
            shutil.copy(os.path.join(cfg.checkpoint_dir, ckpt_name), os.path.join(cfg.best_dir,tasks_task, 'best.pth'))
            #shutil.move(os.path.join(cfg.checkpoint_dir, ckpt_name), os.path.join(cfg.checkpoint_dir, new_name))        #Change name to best
            logger.info(f'Select {selected_name} as best')
	

        # else:
        #     os.remove(os.path.join(cfg.checkpoint_dir, ckpt_name))
        #     logger.info(f'Remove {ckpt_name}')

    simulation_app.close()


if __name__ == '__main__':
    main()