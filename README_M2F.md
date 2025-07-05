##VoldetectNet
这是一个分阶段体积检测模型，主要由多序列二维检测模型M2F-DETR和深度预测模型ADP-Net构成，我们首先开源了其中二维检测部分模型M2F-DETR的代码，该代码是在DINO的代码的基础上进行修改，
This is a staged volume detection model, mainly composed of the multi-sequence 2D detection model M2F-DETR and the depth prediction model ADP-Net. We first open-sourced the code of the 2D detection part model M2F-DETR, which is modified based on the code of DINO.
## Project Introduction
<details>
M2F-DETR is in [DINO code] (https://github.com/IDEACVR/DINO) on the basis of the code changes, major changes points include the following: 
（1）多序列数据的读取,相关修改在datasets，我们提供一种数据的读入方式，但可能并不适合于你的数据，可以作为参考，相关代码在coco.py和transfom.py中。
  (1) Reading multi-sequence data, the relevant modifications are in datasets. We provide a way to read data in, but it may not be suitable for your data. It can be used as a reference. The relevant code is in coco.py and Transformom.py.
大致思路为建立一个映射关系，寻找到每一张中心切片所对应的多序列切片读入--及多张图像，由于需要处理多张图像，原本的transform变换不再设配，需要能够同时处理多张图像，所以重构transfom.py中的代码
（2）多序列特征提取及预处理,相关修改在models/dino/dino.py
（3）重点修改encoder部分，引入MSeqFusion和MscaleFusion，相关修改在models/dino/deformable_transformer中
Author: Majiajie
</details>


## Installation

<details>
  <summary>Installation</summary>
  
  We use the environment same to DAB-DETR and DN-DETR to run DINO. If you have run DN-DETR or DAB-DETR, you can skip this step. 
  We test our models under ```python=3.7.3,pytorch=1.9.0,cuda=11.1```. Other versions might be available as well. Click the `Details` below for more details.

   1. Clone this repo
   ```sh
   git clone https://github.com/IDEA-Research/DINO.git
   cd DINO
   ```

   2. Install Pytorch and torchvision

   Follow the instruction on https://pytorch.org/get-started/locally/.
   ```sh
   # an example:
   conda install -c pytorch pytorch torchvision
   ```

   3. Install other needed packages
   ```sh
   pip install -r requirements.txt
   ```

   4. Compiling CUDA operators
   ```sh
   cd models/dino/ops
   python setup.py build install
   # unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```
</details>




## Data

<details>
  <summary>Data</summary>

Please download [COCO 2017](https://cocodataset.org/) dataset and organize them as following:
```
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

</details>


## Run

<details>
  <summary>1. Eval our pretrianed models</summary>

  <!-- ### Eval our pretrianed model -->
  Download our DINO model checkpoint "checkpoint0011_4scale.pth" from [this link](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_?usp=sharing) and perform the command below. You can expect to get the final AP about 49.0.
  ```sh
  bash scripts/DINO_eval.sh /path/to/your/COCODIR /path/to/your/checkpoint
  ```

</details>



<details>
  <summary>2. Inference and Visualizations</summary>

For inference and visualizations, we provide a [notebook](inference_and_visualization.ipynb) as an example.

</details>



<details>
  <summary>3. Train a 4-scale model for 12 epochs</summary>

We use the DINO 4-scale model trained for 12 epochs as an example to demonstrate how to evaluate and train our model.

You can also train our model on a single process:
```sh
bash scripts/DINO_train.sh /path/to/your/COCODIR
```

</details>



<details>
  <summary>4. Supports for Swin Transformer</summary>

To train Swin-L model, you need to first download the checkpoint of Swin-L backbone from [link](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth) and specify the dir of the pre-trained backbone when running the scripts. Here is an example.
```
bash scripts/DINO_train_submitit_swin.sh /path/to/your/COCODIR /path/to/your/pretrained_backbone 
```

</details>


<details>
  <summary>5. Distributed Run</summary>

As the training is time consuming, we suggest to train the model on multi-device.

If you plan to train the models **on a cluster with Slurm**, here is an example command for training:
```sh
# for DINO-4scale: 49.0
bash scripts/DINO_train_submitit.sh /path/to/your/COCODIR

# for DINO-5scale: 49.4
bash scripts/DINO_train_submitit_5scale.sh /path/to/your/COCODIR
```
Notes:
The results are sensitive to the batch size. We use 16(2 images each GPU x 8 GPUs for DINO-4scale and 1 images each GPU x 16 GPUs for DINO-5scale) by default.

Or run with **multi-processes on a single node**:
```sh
# for DINO-4scale: 49.0
bash scripts/DINO_train_dist.sh /path/to/your/COCODIR
```

</details>

</details>


<details>
  <summary>6. Training/Fine-tuning a DINO on your custom dataset</summary>

To train a DINO on a custom dataset **from scratch**, you need to tune two parameters in a config file:
- Tuning the `num_classes` to the number of classes to detect in your dataset.
- Tuning the parameter `dn_labebook_size` to ensure that `dn_labebook_size >= num_classes + 1`

To **leverage our pre-trained models** for model fine-tuning, we suggest add two more commands in a bash:
- `--pretrain_model_path /path/to/a/pretrianed/model`. specify a pre-trained model.
- `--finetune_ignore label_enc.weight class_embed`. ignore some inconsistent parameters.


</details>




# Links
Our model is based on [DAB-DETR](https://arxiv.org/abs/2201.12329) and [DN-DETR](https://arxiv.org/abs/2203.01305).
<p>
<font size=3><b>DN-DETR: Accelerate DETR Training by Introducing Query DeNoising.</b></font>
<br>
<font size=2>Feng Li*, Hao Zhang*, Shilong Liu, Jian Guo, Lionel M. Ni, Lei Zhang.</font>
<br>
<font size=2>IEEE Conference on Computer Vision and Pattern Recognition (<b>CVPR</b>) 2022.</font>
<br>
<a href='https://arxiv.org/abs/2203.01305'>[paper]</a> <a href='https://github.com/FengLi-ust/DN-DETR'>[code]</a> <a href='https://www.zhihu.com/question/517340666/answer/2381304399'>[中文解读]</a>
</p>

<p>
<font size=3><b>DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR.</b></font>
<br>
<font size=2>Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, Lei Zhang.</font>
<br>
<font size=2>International Conference on Learning Representations (<b>ICLR</b>) 2022.</font>
<br>
<a href='https://arxiv.org/abs/2201.12329'>[paper]</a> <a href='https://github.com/SlongLiu/DAB-DETR'>[code]</a>    
</p>

We also thank great previous work including DETR, Deformable DETR, SMCA, Conditional DETR, Anchor DETR, Dynamic DETR, etc. More related work are available at [Awesome Detection Transformer](https://github.com/IDEACVR/awesome-detection-transformer).

## LICNESE
DINO is released under the Apache 2.0 license. Please see the [LICENSE](LICNESE) file for more information.

Copyright (c) IDEA. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Bibtex
If you find our work helpful for your research, please consider citing the following BibTeX entry.   
```bibtex
@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{li2022dn,
      title={Dn-detr: Accelerate detr training by introducing query denoising},
      author={Li, Feng and Zhang, Hao and Liu, Shilong and Guo, Jian and Ni, Lionel M and Zhang, Lei},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={13619--13627},
      year={2022}
}

@inproceedings{
      liu2022dabdetr,
      title={{DAB}-{DETR}: Dynamic Anchor Boxes are Better Queries for {DETR}},
      author={Shilong Liu and Feng Li and Hao Zhang and Xiao Yang and Xianbiao Qi and Hang Su and Jun Zhu and Lei Zhang},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=oMI9PjOb9Jl}
}
```
