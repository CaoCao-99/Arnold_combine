# check if an argument is provided

base_output_root=${1:-"/root/arnold/output_dev"}
output_folder=${2:-"/root/arnold/PERACTMODEL/ZZAMTONG"}

# Define an array of tasks
tasks=('pickup_object' 'reorient_object' 'open_drawer' 'close_drawer' 'open_cabinet' 'close_cabinet' 'pour_water' 'transfer_water')

# Iterate three times for each set of tasks
for i in {1..3}; do
#    output_root="/root/arnold/output_dev2/output_iteration_${i}"
    output_root="${base_output_root}/output_iteration_${i}"
    echo "Iteration: $i, Output root: $output_root"

    # Iterate over the tasks
    for task in "${tasks[@]}"; do
        echo "Running task: $task  and output_root: $output_root"
        /isaac-sim/python.sh eval_combine.py task="$task" model=peract lang_encoder=clip mode=eval use_gt=[0,0] visualize=1 record=True output_root="$output_root" data_root=/vagrant/data_for_challenge_final output_folder="$output_folder" state_head=1 checkpoint_dir_grasp=/root/arnold/PERACTMODEL/STEP_2/Best/grasp_gt_GT/multi/best.pth checkpoint_dir_manipulate=/root/arnold/PERACTMODEL/STEP_2/Best/manipulate_gt/multi/best.pth
    done
done
