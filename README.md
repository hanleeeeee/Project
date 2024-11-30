# Enhancing Low-light Images Using CycleGAN and Near-Infrared Dual Blending

## 1. Project Results and Overview
  The visibility of images is often compromised by low-light conditions, backlighting, and low contrast.  Techniques such as histogram equalization and near-infrared (NIR)-visible fusion are commonly employed to mitigate these challenges . However, histogram equalization frequently results in detail loss and oversaturation, while pairing images in NIR-visible fusion remains complex and error-prone.  This study presents a novel method to address these limitations effectively.  The proposed algorithm leverages CycleGAN to generate synthetic NIR images, blended twice with visible images, to achieve tone-compression effects, substantially minimizing detail loss and oversaturation . This innovative approach enhances image quality while overcoming the inherent drawbacks of traditional methods . The results demonstrate that images generated using our method outperform conventional algorithms in terms of quality. This advancement holds significant potential for applications in various domains, particularly self-driving vehicles and CCTV surveillance systems, where reliable image clarity is paramount.

## 2 Source Code
## Run & Test

#### Structure of the project :

* src
  * train.py
    * to train for transforming between NIR and RGB image
  * test.py
    * to test for transforming between NIR and RGB image
  * datasets folder
    * Where 3 versions of the dataset is present, called trainA, trainB and test
    * trainA have RGB images, trainB have NIR images, test have RGB images for testing
  * input folder
    * Where 2 versions of the dataset is present, called near-infrared and visible dataset
  * output folder
    * where for result folder for main.py
  * main.py
    * Originally, **`test.py`** should be executed first to perform the necessary pre-processing before running **`main.py`**. However, since all the required settings have already been configured in the **`input`** folder, you can directly run **`main.py`** to check the post-processing results.

#### file and folder explanation :
* train.py
    * purpose
      *generate NIR-image from visible image 
* test.py
   * purpose
      *generate NIR-image from visible image 
* folders
    * input, output: for implementing main.py
    * datasets: for train.py and test.py
    *result: for storing images from test.py 
      *generate NIR-image from visible image 


#### Prerequisites

Start by cloning the project on your machine

```
git clone 
```
You will need a recent version of Python 3.8 with multiple dependencies :

* torch>=1.4.0
* torchvision>=0.5.0
* dominate>=2.4.0
* visdom>=0.1.8.8
* Torch
* CUDA toolkit (If using Nvidia GPU)

  * try `nvcc --version` in a terminal to ensure that CUDA toolkit is installed
  * also try running `torch.cuda.is_available()` to ensure it is available, and thus ensure maximum possible speed on your Nvidia GPU

  ```python
  >>> import torch
  >>> torch.cuda.is_available()
  True
  ```


## 3 Performance Metrics
### results and compare
![ex1_result](https://github.com/user-attachments/assets/66b32406-89fc-4d08-a159-aa94e91a3703)
![ex2_result](https://github.com/user-attachments/assets/caa7a44c-4c07-44e9-bafa-6bd15cc31efb)
![ex3_result](https://github.com/user-attachments/assets/cfee4b41-57d0-45a1-b424-d7d51b4a4f26)

### performance Metrics







## 4 Installation and Usage
This project has been implemented using own GPU. The size of the dataset is too large to upload and include it in this project, therefore instructions will be given on how to download and use it will be provided in the following. NIR dataset url: https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/  RGB(visible) dataset url :https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

### train steps:
Download the dataset from above URL and then store each of images to "./datasets/trainA" , "./datasets/trainB"
run train.py

### test steps:
test.py
move results components to input near-infrared folder(./results/images->./input/near-infrared)
extract extra RGB images from RGB(visible) here-> url :https://github.com/cs-chan/
main.py
Evaluate performance based on displayed metrics and visualizations

## 5 References and Documentation
References
J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2242–2251. https://arxiv.org/abs/1703.10593

Y. Wang, A. Mousavian, Y. Xiang, and D. Fox, "DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 3343–3352. https://arxiv.org/abs/1901.04780

A. Vanmali, S. Raval, P. Borse, and S. Chaudhuri, "NIR-Visible Image Fusion for Enhancing Visibility," IEEE Trans. Comput. Imaging, vol. 5, no. 4, pp. 539–551, Dec. 2019. https://doi.org/10.1109/TCI.2019.2919999

E. Reinhard, M. Ashikhmin, B. Gooch, and P. Shirley, "Color Transfer between Images," IEEE Comput. Graph. Appl., vol. 21, no. 5, pp. 34–41, Sep.-Oct. 2001. https://doi.org/10.1109/38.946629

G. M. Johnson and M. D. Fairchild, "iCAM06: A Refined Image Appearance Model for HDR Image Rendering," J. Vis. Commun. Image Represent., vol. 18, no. 5, pp. 406–414, Oct. 2007. https://doi.org/10.1016/j.jvcir.2006.11.002

L. Xu, Q. Yan, Y. Xia, and J. Jia, "Image Smoothing via L0 Gradient Minimization," ACM Trans. Graph., vol. 30, no. 6, Dec. 2011. https://doi.org/10.1145/2070781.2024176

C. Wei, W. Wang, W. Yang, and J. Liu, "RetinexNet: Deep Retinex Decomposition for Low-Light Enhancement," in Proc. BMVC, 2018, pp. 1–12. https://arxiv.org/abs/1808.04560


**Explanation of Key Algorithm (Support Fusion)**  

I executed CycleGAN training both in an unpaired and paired sequential manner, enabling extensive learning to generate NIR images from general RGB images. Subsequently, I created a sharp image by fusing the generated fake NIR image with the real visible image. The key algorithms used in this process include alpha blending, gamma correction, CLAHE, and color compensation.

## 6 Issues and Contributions
### Issues:

#### Data Quality and Variability:
One of the challenges encountered during the project was the variability in the quality of input data. The performance of the algorithms, particularly in generating NIR images from RGB inputs, was sometimes affected by noise or inconsistencies in the data. Future work could involve using more controlled datasets or enhancing the data preprocessing pipeline to minimize such issues.

#### Computational Complexity:
The training of CycleGAN on large datasets, especially when running in a paired and unpaired sequential manner, was computationally expensive and time-consuming. Although the results were promising, there were limitations in processing speed. Optimizing the model for faster training or exploring alternative architectures might help improve efficiency.

#### Fusion Artifacts:
When blending the generated fake NIR images with the real visible images, there were occasional artifacts that impacted the image quality. The fusion process, although successful in many cases, sometimes resulted in color inconsistencies or visible seams in the final image. Further refinement of the blending techniques, such as enhancing the alpha blending process or experimenting with different fusion algorithms, could address this issue.

### Contributions:

#### Innovative Approach for NIR-Visible Fusion:
The primary contribution of this project lies in the innovative use of CycleGAN for generating NIR images from RGB inputs, followed by a novel fusion process with real visible images to enhance image clarity. This method provides a new pathway for improving image quality in challenging environments, such as low-light or infrared imaging.

#### Application of Advanced Image Processing Techniques:
This project successfully integrated advanced image processing techniques like gamma correction, CLAHE, and color compensation, which significantly improved the final output. These algorithms played a crucial role in enhancing the contrast, brightness, and overall visual quality of the fused images.

#### Framework for Low-Light Image Enhancement:
The project contributes to the field of low-light image enhancement by proposing a framework that combines deep learning-based NIR generation with traditional image enhancement techniques. This hybrid approach offers a promising solution for applications where visibility enhancement in difficult lighting conditions is essential.

#### Potential for Further Development and Optimization:
The project sets the stage for future improvements and optimizations. The algorithms used can be further tuned for better performance, and the fusion techniques can be expanded to handle more complex image types or to improve real-time performance. Additionally, the methodology can be applied to other areas such as medical imaging, remote sensing, and surveillance.

## 7 Future work


### 1. Optimization of Computational Efficiency:
   One of the key areas for improvement is the optimization of the training process for CycleGAN. The current model requires significant computational resources and time, especially when processing large datasets. Future work could involve investigating more efficient architectures or utilizing techniques like model pruning or knowledge distillation to speed up the training process without compromising on performance.

### 2. Improvement of Fusion Quality:
   While the fusion of fake NIR images with real visible images showed promising results, there are still occasional artifacts and color inconsistencies. Exploring advanced fusion techniques, such as multi-scale fusion or deep learning-based fusion methods, could further improve the visual quality of the final images. Additionally, more sophisticated methods for handling the alpha blending process could lead to smoother transitions between the NIR and visible images.


### 3. **Real-Time Implementation:**  
   One potential direction for future development is the implementation of this fusion system in real-time applications. Optimizing the model for real-time image enhancement, particularly for use in systems like autonomous vehicles or surveillance, would be valuable. Real-time processing would require both model optimization and hardware acceleration, such as using GPUs or edge computing devices.

