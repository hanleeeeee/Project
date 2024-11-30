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


## 3 Performance Metrices







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

Explanation of Key Algorithm (Support Vector Machine SVM)
A Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification tasks. It works by finding the optimal hyperplane that separates data into different classes, maximizing the margin between the hyperplane and the closest data points, known as support vectors. The SVM can handle both linear and non-linear classification problems. For non-linear data, SVM uses a kernel trick to map the data into a higher-dimensional space where a linear separation is possible. Common kernels include the Radial Basis Function (RBF) kernel, which allows for flexible decision boundaries.

The key parameters in SVM are C and gamma. The parameter C controls the trade-off between maximizing the margin and minimizing classification errors, while gamma defines the influence of individual data points. SVM is effective in high-dimensional spaces and is known for its ability to generalize well, but it can be computationally expensive and requires careful tuning of parameters to achieve optimal performance.

## 6 Issues and Contributions
One of the primary challenges observed in this project is the model's limited generalization to extreme lighting conditions. While the SVM performs well under most scenarios, it struggles when faced with very bright or heavily shadowed images, where facial features become significantly obscured or distorted. Another limitation is the relatively small size of the Extended Yale B dataset, which, despite its controlled lighting variations, does not capture real-world complexities such as facial expressions, occlusions, or diverse backgrounds. The scalability of the model is another concern. SVMs are computationally intensive, especially when handling larger datasets, due to their quadratic training complexity. This may pose challenges for scaling the project to datasets with a significantly higher number of samples. Additionally, the model’s accuracy depends heavily on preprocessing steps such as resizing and grayscale conversion. Any inconsistencies or errors during these steps can negatively impact performance. Although the Extended Yale B dataset is not very large, the time required to load and preprocess the data, as well as to train the SVM model, is significant—taking approximately 30 minutes in some cases. This highlights the need for more efficient data handling and processing strategies.

Contributions:

Optimization of Data Loading and Preprocessing:
Efforts were made to streamline the loading and preprocessing of the Extended Yale B dataset to reduce runtime without sacrificing data quality. This included experimenting with optimized file handling techniques and reducing redundancy in preprocessing steps
Runtime Enhancements:
Adjustments were implemented to improve the efficiency of the training process. This included tuning SVM hyperparameters and using parallel processing where possible to speed up computations
Evaluation of Alternative Models:
The project incorporated a framework to test and compare the performance of different machine learning algorithms, such as k-Nearest Neighbors (k-NN) and Random Forest, against SVM. This provided insights into alternative approaches to address scalability and runtime issues
Model Scalability Exploration:
Experiments were conducted to explore the performance of the SVM model when trained on subsets of larger datasets, enabling an assessment of its scalability potential
Integration of Advanced Preprocessing Techniques:
Advanced preprocessing methods, such as histogram equalization and contrast adjustment, were explored to improve the robustness of the model under extreme lighting conditions
Benchmarking Against Real-World Datasets:
A plan was developed to benchmark the current implementation against larger, real-world datasets to identify further areas for improvement and validate the model’s performance in more diverse scenarios
7 Future work
To build on the current project, several potential improvements can be explored. One promising direction is the incorporation of neural networks, particularly Convolutional Neural Networks (CNNs), which can provide superior performance on larger and more diverse datasets by learning hierarchical feature representations. Data augmentation is another avenue worth pursuing, as it can enrich the dataset with synthetic variations, including different lighting angles, occlusions, and facial expressions, to improve the model’s robustness.

Future iterations of this project could also focus on enabling real-time facial recognition by integrating a webcam or camera feed. This would require optimizing the SVM implementation or exploring alternative algorithms better suited for real-time performance. Another area for enhancement is the combination of SVM with feature extraction techniques such as Principal Component Analysis (PCA) or Histogram of Oriented Gradients (HOG), which could boost both accuracy and speed.

Finally, cross-domain testing on other datasets would help evaluate the model's ability to generalize beyond the Extended Yale B dataset. Adding explainability tools to visualize the SVM decision boundaries could also provide valuable insights into the model's behavior, increasing its interpretability and trustworthiness.
"# Enhancing-Low-light-Images-Using-CycleGAN-and-Near-Infrared-Dual-Blending" 
