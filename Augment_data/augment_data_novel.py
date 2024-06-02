import os
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
num_dict = {
    1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
    11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
    16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty',
    21: 'twenty-one', 22: 'twenty-two', 23: 'twenty-three', 24: 'twenty-four', 25: 'twenty-five',
    26: 'twenty-six', 27: 'twenty-seven', 28: 'twenty-eight', 29: 'twenty-nine', 30: 'thirty',
    31: 'thirty-one', 32: 'thirty-two', 33: 'thirty-three', 34: 'thirty-four', 35: 'thirty-five',
    36: 'thirty-six', 37: 'thirty-seven', 38: 'thirty-eight', 39: 'thirty-nine', 40: 'forty',
    45: 'forty-five', 50: 'fifty', 55: 'fifty-five', 60: 'sixty', 65: 'sixty-five',
    70: 'seventy', 75: 'seventy-five', 80: 'eighty', 85: 'eighty-five', 90: 'ninety',
    95: 'ninety-five', 100: 'one hundred'
}

def parse_state(filename):
    units = filename.split('-')
    task_type = ''
    if 'pickup_object' in filename:
        init_state = 0
        goal_state = round(int(units[-3]) / 40, 2)
        task_type = 'pickup_object'
    elif 'reorient_object' in filename:
        init_state = 0.5                    # 90 도 init_state 는 90 도
        goal_state = round(int(filename.split(')')[0].split('_')[-1]) / 180, 2)
        task_type = 'reorient_object'
    elif 'pour_water' in filename:
        init_state = 0
        goal_state = round((100 - int(units[-3])) / 100, 2)
        task_type = 'pour_water'
    elif 'transfer_water' in filename:
        init_state = 0
        goal_state = round(int(units[-3]) / 100, 2)
        task_type = 'transfer_water'
    elif 'open_drawer' in filename:
        init_state = float(units[-4])
        goal_state = float(units[-3])
        task_type = 'open_drawer'  # 예를 들어, 'drawer' 작업으로 가정
    elif 'close_drawer' in filename:
        init_state = float(units[-4])
        goal_state = float(units[-3])
        task_type = 'close_drawer'  # 예를 들어, 'drawer' 작업으로 가정
    return init_state, goal_state, task_type

def num_to_words(num):
    # 숫자를 단어로 변환


    return num_dict.get(num, str(num))

def augment_data(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_file_path = os.path.join(output_folder, 'augmentation_log.txt')
    with open(log_file_path, 'a') as log_file:
        for fname in os.listdir(input_folder):
            if fname.endswith('.npz'):
                init_state, goal_state, task_type = parse_state(fname)
                data = np.load(os.path.join(input_folder, fname), allow_pickle=True)
                gt_frames = data['gt']
                language_instructions = gt_frames[0]['instruction']
                if task_type == 'pickup_object':
                    # 10부터 100까지 10단위로 새로운 goal state 생성
                    for new_goal in [40]:
                        if new_goal == int(goal_state * 40):
                            continue

                        # 새로운 goal state의 파일이 이미 존재하는지 확인
                        goal_state_str = str(int(goal_state * 40))
                        new_filename = fname.replace(f'-{int(goal_state * 40)}-', f'-{new_goal}-')
                        base_filename = '-'.join(new_filename.split('-')[:-1])
                        if any(f.startswith(base_filename) for f in os.listdir(output_folder)):
                            continue

                        new_goal_state = new_goal / 40.0
                        modified_gt_frames = copy.deepcopy(gt_frames)
                        new_file_path = os.path.join(output_folder, new_filename)
                        if os.path.exists(new_file_path):
                            continue
                        # 새로운 goal state에 맞게 act_pos 조정
                        act_pos = np.array(modified_gt_frames[3]['position_rotation_world'][0])
                        cur_pos = np.array(modified_gt_frames[2]['position_rotation_world'][0])
                        delta = (new_goal - int(goal_state * 40))

                        original_act_pos = act_pos.copy()

                        # 기본적으로 y 좌표 변경
                        act_pos[1] += delta

                        # 튜플을 딕셔너리로 변환하여 수정 가능하도록 함
                        for i in range(len(modified_gt_frames)):
                            if isinstance(modified_gt_frames[i], tuple):
                                modified_gt_frames[i] = dict(modified_gt_frames[i])

                        # position_rotation_world를 리스트로 변환하여 수정
                        position_rotation_world = list(modified_gt_frames[3]['position_rotation_world'])
                        position_rotation_world[0] = act_pos
                        modified_gt_frames[3]['position_rotation_world'] = tuple(position_rotation_world)
                        
                        # import pdb
                        # pdb.set_trace()
                        # language_instructions 수정
                            # centimeters 앞의 단어를 찾아서 바꿈
                        words = language_instructions.split()
                        for i in range(len(words)):
                            if words[i] == 'centimeters' and i > 0:
                                words[i - 1] = num_to_words(new_goal)
                        new_language_instructions = ' '.join(words)
                        
                            
                        modified_gt_frames[0]['instruction'] = new_language_instructions

                        # 수정된 npz 파일 저장
                        np.savez_compressed(new_file_path, gt=modified_gt_frames, instruction=new_language_instructions)
                        # import pdb
                        # pdb.set_trace()
                        # 로그 파일에 정보 기록
                        log_file.write(f'Original file: {fname}\n')
                        log_file.write(f'New file: {new_filename}\n')
                        log_file.write(f'Task type: {task_type}\n')
                        log_file.write(f'Initial state: {init_state}\n')
                        log_file.write(f'Original goal state: {goal_state}\n')
                        log_file.write(f'New goal state: {new_goal_state}\n')
                        log_file.write(f'cur_pos: {cur_pos.tolist()}\n')
                        log_file.write(f'Original act_pos: {original_act_pos.tolist()}\n')
                        log_file.write(f'New act_pos: {act_pos.tolist()}\n')
                        log_file.write(f'Original instruction: {language_instructions}\n')
                        log_file.write(f'New instruction: {new_language_instructions}\n')
                        log_file.write('\n')
                        
                elif task_type == 'open_drawer':
                    # 0.05 단위로 새로운 goal state 생성
                    for new_goal in [0.75]:
                        new_goal = round(new_goal, 2)
                        if np.isclose(new_goal, goal_state, atol=1e-2) or new_goal <= init_state:
                            continue

                        # 새로운 goal state의 파일이 이미 존재하는지 확인
                        new_filename = fname.replace(f'-{goal_state}-', f'-{new_goal}-')
                        base_filename = '-'.join(new_filename.split('-')[:-1])
                        if any(f.startswith(base_filename) for f in os.listdir(output_folder)):
                            continue

                        new_goal_state = new_goal
                        modified_gt_frames = copy.deepcopy(gt_frames)
                        new_file_path = os.path.join(output_folder, new_filename)
                        if os.path.exists(new_file_path):
                            continue
                        # 새로운 goal state에 맞게 act_pos 조정
                        act_pos = np.array(modified_gt_frames[3]['position_rotation_world'][0])
                        cur_pos = np.array(modified_gt_frames[2]['position_rotation_world'][0])
                        # import pdb
                        # pdb.set_trace()
                        original_act_pos = act_pos.copy()

                        # x, z 좌표 변경
                        diff_x = (act_pos[0] - cur_pos[0]) / ((goal_state - init_state) * 100)  # 1%당 움직이는 x 값
                        diff_z = (act_pos[2] - cur_pos[2]) / ((goal_state - init_state) * 100)  # 1%당 움직이는 z 값

                        # 새로운 goal state에 맞게 x, z 좌표 조정
                        act_pos[0] = cur_pos[0] + (new_goal_state - init_state) * 100 * diff_x
                        act_pos[2] = cur_pos[2] + (new_goal_state - init_state) * 100 * diff_z

                        # 튜플을 딕셔너리로 변환하여 수정 가능하도록 함
                        for i in range(len(modified_gt_frames)):
                            if isinstance(modified_gt_frames[i], tuple):
                                modified_gt_frames[i] = dict(modified_gt_frames[i])

                        # position_rotation_world를 리스트로 변환하여 수정
                        position_rotation_world = list(modified_gt_frames[3]['position_rotation_world'])
                        position_rotation_world[0] = act_pos
                        modified_gt_frames[3]['position_rotation_world'] = tuple(position_rotation_world)

                        # # language_instructions 수정
                        # old_word = num_to_words(int(goal_state * 100))
                        # new_word = num_to_words(int(new_goal * 100))
                        # new_language_instructions = language_instructions.replace(old_word, new_word)
                        words = language_instructions.split()
                        for i in range(len(words)):
                            if words[i] == 'percent':
                                if words[i - 2] in num_dict.values():
                                    words[i - 2] = ''
                                    words[i - 1] = num_to_words(int(new_goal * 100))
                                else:
                                    words[i - 1] = num_to_words(int(new_goal * 100))
                        new_language_instructions = ' '.join(words)
                        modified_gt_frames[0]['instruction'] = new_language_instructions

                        # 수정된 npz 파일 저장
                        np.savez_compressed(new_file_path, gt=modified_gt_frames, instruction=new_language_instructions)

                        # 로그 파일에 정보 기록
                        log_file.write(f'Original file: {fname}\n')
                        log_file.write(f'New file: {new_filename}\n')
                        log_file.write(f'Task type: {task_type}\n')
                        log_file.write(f'Initial state: {init_state}\n')
                        log_file.write(f'Original goal state: {goal_state}\n')
                        log_file.write(f'New goal state: {new_goal_state}\n')
                        log_file.write(f'cur_pos: {cur_pos.tolist()}\n')
                        log_file.write(f'Original act_pos: {original_act_pos.tolist()}\n')
                        log_file.write(f'New act_pos: {act_pos.tolist()}\n')
                        log_file.write(f'Original instruction: {language_instructions}\n')
                        log_file.write(f'New instruction: {new_language_instructions}\n')
                        log_file.write('\n')
                elif task_type == 'close_drawer':
                    # 0.05 단위로 새로운 goal state 생성
                    for new_goal in [0.25]:
                        new_goal = round(new_goal, 2)
                        if np.isclose(new_goal, goal_state, atol=1e-2) or new_goal >= init_state:
                            continue

                        # 새로운 goal state의 파일이 이미 존재하는지 확인
                        new_filename = fname.replace(f'-{goal_state}-', f'-{new_goal}-')
                        base_filename = '-'.join(new_filename.split('-')[:-1])
                        if any(f.startswith(base_filename) for f in os.listdir(output_folder)):
                            continue
                        new_file_path = os.path.join(output_folder, new_filename)
                        if os.path.exists(new_file_path):
                            continue
                        new_goal_state = new_goal
                        modified_gt_frames = copy.deepcopy(gt_frames)

                        # 새로운 goal state에 맞게 act_pos 조정
                        act_pos = np.array(modified_gt_frames[3]['position_rotation_world'][0])
                        cur_pos = np.array(modified_gt_frames[2]['position_rotation_world'][0])
                        
                        original_act_pos = act_pos.copy()

                        # x, z 좌표 변경
                        diff_x = (act_pos[0] - cur_pos[0]) / ((init_state - goal_state) * 100)  # 1%당 움직이는 x 값
                        diff_z = (act_pos[2] - cur_pos[2]) / ((init_state - goal_state) * 100)  # 1%당 움직이는 z 값

                        # 새로운 goal state에 맞게 x, z 좌표 조정
                        act_pos[0] = cur_pos[0] + (init_state - new_goal_state) * 100 * diff_x
                        act_pos[2] = cur_pos[2] + (init_state - new_goal_state) * 100 * diff_z

                        # 튜플을 딕셔너리로 변환하여 수정 가능하도록 함
                        for i in range(len(modified_gt_frames)):
                            if isinstance(modified_gt_frames[i], tuple):
                                modified_gt_frames[i] = dict(modified_gt_frames[i])

                        # position_rotation_world를 리스트로 변환하여 수정
                        position_rotation_world = list(modified_gt_frames[3]['position_rotation_world'])
                        position_rotation_world[0] = act_pos
                        modified_gt_frames[3]['position_rotation_world'] = tuple(position_rotation_world)

                        # language_instructions 수정
                        # old_word = num_to_words(int(goal_state * 100))
                        # new_word = num_to_words(int(new_goal * 100))
                        # new_language_instructions = language_instructions.replace(old_word, new_word)
                        words = language_instructions.split()
                        if len(words) > 1:
                            if 'open' in words[-1]:
                                words[-2] = f'{num_to_words(int(new_goal * 100))} percent'
                            elif 'close' in words[-1]:
                                words[-2] = f'{num_to_words(100 - int(new_goal * 100))} percent'
                                
                        new_language_instructions = ' '.join(words)
                        modified_gt_frames[0]['instruction'] = new_language_instructions

                        # 수정된 npz 파일 저장
                        np.savez_compressed(new_file_path, gt=modified_gt_frames, instruction=new_language_instructions)

                        # 로그 파일에 정보 기록
                        log_file.write(f'Original file: {fname}\n')
                        log_file.write(f'New file: {new_filename}\n')
                        log_file.write(f'Task type: {task_type}\n')
                        log_file.write(f'Initial state: {init_state}\n')
                        log_file.write(f'Original goal state: {goal_state}\n')
                        log_file.write(f'New goal state: {new_goal_state}\n')
                        log_file.write(f'cur_pos: {cur_pos.tolist()}\n')
                        log_file.write(f'Original act_pos: {original_act_pos.tolist()}\n')
                        log_file.write(f'New act_pos: {act_pos.tolist()}\n')
                        log_file.write(f'Original instruction: {language_instructions}\n')
                        log_file.write(f'New instruction: {new_language_instructions}\n')
                        log_file.write('\n')
                elif task_type == 'reorient_object':
                    for new_goal in [135]:
                        if new_goal == goal_state:
                            continue

                        # 새로운 goal state의 파일이 이미 존재하는지 확인
                        new_filename = fname.replace(f'_{goal_state})', f'_{new_goal})')
                        base_filename = '-'.join(new_filename.split('-')[:-1])
                        if any(f.startswith(base_filename) for f in os.listdir(output_folder)):
                            continue

                        modified_gt_frames = copy.deepcopy(gt_frames)
                        new_file_path = os.path.join(output_folder, new_filename)
                        if os.path.exists(new_file_path):
                            continue

                        # 새로운 goal state에 맞게 act_pos 조정
                        cur_pos_quat = np.array(modified_gt_frames[2]['position_rotation_world'][1])
                        act_pos_quat = np.array(modified_gt_frames[3]['position_rotation_world'][1])

                        cur_pos_euler = R.from_quat(cur_pos_quat).as_euler('xyz', degrees=True)
                        act_pos_euler = R.from_quat(act_pos_quat).as_euler('xyz', degrees=True)

                        # 각도를 새로운 goal state에 맞게 조정
                        act_pos_euler[2] = cur_pos_euler[2] + (new_goal - init_state)
                        new_act_pos_quat = R.from_euler('xyz', act_pos_euler, degrees=True).as_quat()

                        # position_rotation_world를 리스트로 변환하여 수정
                        position_rotation_world = list(modified_gt_frames[3]['position_rotation_world'])
                        position_rotation_world[1] = new_act_pos_quat
                        modified_gt_frames[3]['position_rotation_world'] = tuple(position_rotation_world)

                        # language_instructions 수정
                        words = language_instructions.split()
                        for i in range(len(words)):
                            if words[i] == 'degrees' and i > 0:
                                words[i - 1] = num_to_words(new_goal)
                        new_language_instructions = ' '.join(words)

                        modified_gt_frames[0]['instruction'] = new_language_instructions

                        # 수정된 npz 파일 저장
                        np.savez_compressed(new_file_path, gt=modified_gt_frames, instruction=new_language_instructions)

                        # 로그 파일에 정보 기록
                        log_file.write(f'Original file: {fname}\n')
                        log_file.write(f'New file: {new_filename}\n')
                        log_file.write(f'Task type: {task_type}\n')
                        log_file.write(f'Initial state: {init_state}\n')
                        log_file.write(f'Original goal state: {goal_state}\n')
                        log_file.write(f'New goal state: {new_goal}\n')
                        log_file.write(f'cur_pos_quat: {cur_pos_quat.tolist()}\n')
                        log_file.write(f'Original act_pos_quat: {act_pos_quat.tolist()}\n')
                        log_file.write(f'New act_pos_quat: {new_act_pos_quat.tolist()}\n')
                        log_file.write(f'Original instruction: {language_instructions}\n')
                        log_file.write(f'New instruction: {new_language_instructions}\n')
                        log_file.write('\n')
    print(f'데이터 증강이 완료되었습니다. 새로운 파일들과 로그 파일은 {output_folder}에 저장되었습니다.')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Augment data by generating new goal states')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing the original .npz files')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the augmented .npz files')

    args = parser.parse_args()

    augment_data(args.input_folder, args.output_folder)
