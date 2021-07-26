# LCRCA
The implementation of "LCRCA: Image Super-Resolution Using Lightweight Concatenated Residual Channel Attention Networks"


Dependencies：
  pytorch1.2.0  or 1.1.0 
  opencv-python（installed using pip）
  scikit-learn
  scikit-image
  imageio
  tqdm


Train：
    1. open a terminal and cd to src
    2. type and run: python main.py --model (model name） --scale scale --batch_size batch_size --patch_size patch_size --save "name_of_file_foder"
    3. data will be saved to:  ../experiment/"name_of_file_foder"

Test：
    1. 
    1. type and run: python --main.py --model xxx --scale x --data_test Set5+Set14+B100+Urban100 --load ../experiment/"name_of_file_foder"/model/model_best.pt --test_only --self_ensemble --save_results --save  --self_ensemble 


file description：
    src/model/grnn.py——Network of LCRCA
    option.py——Configuration file 
    
Thanks sanghyun-son et.al. for their excellent open source project at: https://github.com/sanghyun-son/EDSR-PyTorch.


