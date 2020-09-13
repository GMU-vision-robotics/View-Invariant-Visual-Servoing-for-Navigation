# View-Invariant-Visual-Servoing-for-Navigation
code for paper '[Learning View and Target Invariant Visual Servoing for Navigation](https://arxiv.org/abs/2003.02327)'

![Title_image](https://github.com/GMU-vision-robotics/View-Invariant-Visual-Servoing-for-Navigation/blob/master/feature_map.gif)

**Setup GibsonEnv**
Download and install gibson environment from https://github.com/StanfordVL/GibsonEnv.

    git checkout d3aa0a1

**Generate Training and Testing data**

    python sample_image_pairs_with_common_area.py --scene_idx=0
    python visualize_sampled_image_pairs.py

**Learned Visual Servoing Model**
To train a learned-vs model through DQN,

    python vs_controller/train_DQN_vs_overlap.py
To evaluate the performance of the trained model,

    python vs_controller/evaluate_DQN_vs.py
 **Classical VS Model**
Test the vs model using sift and ground-truth depth,

    python visual_servoing/test_IBVS_SIFT_interMatrix_gtDepth_Vz_OmegaY.py  --scene_idx=$i

**Generate Occupancy Map**
Used some code from https://github.com/tenther/cs685-project.
To generate nice occupancy maps for the used Gibson environments,

    sh rrt/run_make_rrt.sh
