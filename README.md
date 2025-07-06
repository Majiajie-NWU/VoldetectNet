## VoldetectNet

This is a staged volume detection model, mainly composed of the multi-sequence 2D detection model M2F-DETR and the depth prediction model ADP-Net. We first open-sourced the code of the 2D detection part model M2F-DETR, which is modified based on the code of DINO.

## Project Introduction

<details>
M2F-DETR is in [DINO code] (https://github.com/IDEACVR/DINO) on the basis of the code changes, major changes points include the following: 
  (1) Reading multi-sequence data, the relevant modifications are in datasets. We provide a way to read data in, but it may not be suitable for your data. It can be used as a reference. The relevant code is in coco.py and Transformom.py.
  The general idea is to establish a mapping relationship to find the multi-sequence slice read in corresponding to each central slice - and multiple images. Since multiple images need to be processed, the original transform transformation is no longer configured. It is necessary to be able to process multiple images simultaneously, so the code in Transformom.py is restructured
（2）Multi-sequence feature extraction and preprocessing, related modifications are available in models/dino/dino.py
（3）The key modification of the encoder part is to introduce MSeqFusion and MscaleFusion. The relevant modifications are in models/dino/deformable_In the transformer
Author: Majiajie
</details>



## Installation

<details>
  <summary>Installation</summary>


  We use the environment same to DINO. 
  We test our models under ```python=3.7.3,pytorch=1.9.0,cuda=11.1```. Other versions might be available as well. 
  We present the environment configuration tutorial for DINO:

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


Please prepare youre dataset and organize them as following:

```
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

notes: The dataset you prepared is different from the one I used. You need to write the mapping rules yourself to complete the data reading
</details>




## Run

<details>
  The model can be trained through main.py
  <summary>1. Trian models</summary>
    <!-- ### Train model -->
  ```
 python.py main.py
  ```
The model can be evaluated through commands
  <!-- ### Eval model -->
  ```sh
  bash scripts/DINO_eval.sh /path/to/your/COCODIR /path/to/your/checkpoint
  ```
</details>



# Links

Our model is based on [DINO](https://arxiv.org/abs/2203.03605)

<p>
<font size=3><b>DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection</b></font>
<br>


We thank great previous work including DETR, Deformable DETR, SMCA, Conditional DETR, Anchor DETR, Dynamic DETR, etc. More related work are available at [Awesome Detection Transformer](https://github.com/IDEACVR/awesome-detection-transformer).

## LICNESE

DINO is released under the Apache 2.0 license. Please see the [LICENSE](LICNESE) file for more information.

Copyright (c) IDEA. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
