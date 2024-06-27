# ARNOLD Challenge: Continual Learning for Grasping and Manipulation
<div style="display: flex; flex-direction: row;">
    <img src="images/3rd.png" alt="3rd Place" width="300"/>
    <img src="images/Arnold.png" alt="Arnold" width="600"/>
</div>
We participated in the ARNOLD Challenge and won 3rd place. 

Our model incorporates two key innovative ideas: Phase Specific Agents and State Interpolation for data augmentation, which significantly improved our performance metrics.


## Pipeline Overview
Our model pipeline involves two major components: Phase Specific Agents and State Interpolation for Data Augmentation

## Phase Specific Agents
<img src="images/PhaseSpecificAgents" alt="Phase Specific Agents" width="800"/>
We introduced Phase Specific Agents to our model, which resulted in a significant performance increase on the validation set from 32% to 45%. This approach involves creating specialized agents for different phases of the task, allowing for more precise and efficient learning.

### Grasp Model
Train model using PERACT

<p><strong>Grasp Model Training Progress</strong></p>
<p>This section shows the progress of the grasp model at different training batches. The left image is at batch 0, and the right image is at batch 140,000. <br> <strong>GT: Brown head, Predict: Red head</strong></p>
<p>For more images and details, visit the <a href="https://github.com/CaoCao-99/Arnold_combine/tree/main/Grasp_model/images">Grasp Model Image Directory</a>.</p>

<table>
  <tr>
    <td><img src="https://github.com/CaoCao-99/Arnold_combine/assets/88222336/9f328341-6518-40c9-9e96-3a51fa78e788" alt="Grasp Model Batch 0" width="400"/><br><p align="center">Batch 0</p></td>
    <td><img src="https://github.com/CaoCao-99/Arnold_combine/assets/88222336/7ceca743-b66f-430c-861e-e2fb4bc7bec8" alt="Grasp Model Batch 140,000" width="400"/><br><p align="center">Batch 140,000</p></td>
  </tr>
</table>

### Manipulate Model
<p><strong>Manipulate Model Training Progress</strong></p>
<p>This section displays the progress of the manipulation model at different training batches. The left image is at batch 0, and the right image is at batch 180,000. <br> <strong>GT: Brown head, Predict: Red head</strong></p>
<p>For more images and details, visit the <a href="https://github.com/CaoCao-99/Arnold_combine/tree/main/Manipulate_model/images">Manipulate Model Image Directory</a>.</p>

<table>
  <tr>
    <td><img src="https://github.com/CaoCao-99/Arnold_combine/assets/88222336/c4606648-ace0-4532-9aed-7bb9ef4901c1" alt="Manipulate Model Batch 0" width="400"/><br><p align="center">Batch 0</p></td>
    <td><img src="https://github.com/CaoCao-99/Arnold_combine/assets/88222336/b3e7304d-ae34-40f1-a7cd-02543e06e74e" alt="Manipulate Model Batch 180,000" width="400"/><br><p align="center">Batch 180,000</p></td>
  </tr>
</table>

## State Interpolation for Data Augmentation
<img src="images/StateInterpolation.png" alt="State Interpolation" width="800"/>
Our second innovation was the application of State Interpolation for data augmentation. This technique enhanced our model's ability to generalize, improving the test set performance from 22% to 31%. State Interpolation involves generating intermediate states between actual data points, effectively increasing the training data diversity.


