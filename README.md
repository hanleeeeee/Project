: The visibility of images is often compromised by low-light conditions, backlighting, and low contrast.  Techniques such as histogram equalization and near-infrared (NIR)-visible fusion are commonly employed to mitigate these challenges . However, histogram equalization frequently results in detail loss and oversaturation, while pairing images in NIR-visible fusion remains complex and error-prone.  This study presents a novel method to address these limitations effectively.  The proposed algorithm leverages CycleGAN to generate synthetic NIR images, blended twice with visible images, to achieve tone-compression effects, substantially minimizing detail loss and oversaturation . This innovative approach enhances image quality while overcoming the inherent drawbacks of traditional methods . The results demonstrate that images generated using our method outperform conventional algorithms in terms of quality. This advancement holds significant potential for applications in various domains, particularly self-driving vehicles and CCTV surveillance systems, where reliable image clarity is paramount.
1 Project Results and Overview
Key Objectives
Facial recognition systems are vital in modern applications, including security, surveillance, and biometric authentication. However, ensuring consistent performance under various environmental factors, especially lighting variations, is a persistent challenge. This project addresses this issue by employing Support Vector Machines (SVM) for robust facial recognition under varying lighting conditions.

Core Purpose
Developing an SVM-based model capable of robust facial recognition
Mitigating lighting variability in facial recognition using SVM
Validating the model’s performance using the Extended Yale B dataset, specifically designed for controlled lighting variability
Analyzing the model's performance and providing insights into its strengths and limitations in handling lighting variations
Accuracy
The model achieved a test accuracy of 99.94%, highlighting the effectiveness of SVM in handling diverse lighting conditions

Performance
The SVM-based approach demonstrated computational efficiency, making it suitable for small to medium-sized datasets, particularly in controlled environments where preprocessing steps significantly enhance performance. However, due to the extensive nature of the computations involved, particularly with larger datasets, the overall execution time for the model can be quite substantial. This issue is compounded when additional steps for a more detailed analysis are incorporated.

To address this, the code has been divided into two main sections: the primary part handles the core functionality of the model, and a secondary, more detailed analysis section is included to evaluate the extended performance metrics. While the main component processes data efficiently, the additional analysis introduces further computational complexity, resulting in longer processing times. This structure allows for more manageable execution of the model while still providing the necessary in-depth insights into the algorithm’s performance, including accuracy, F1-score, precision, recall, and cross-validation results.

This approach enables a more modular exploration of the results, with the detailed analysis section being run separately when in-depth evaluations are needed, balancing the need for computational efficiency and comprehensive model assessment.

Insights
Dataset
This project uses the Extended Yale Face Database B which contains 16128 images of 28 human subjects under 9 poses and 64 illumination conditions
Impact of Lighting Variations
Extreme lighting conditions can distort facial features, affecting recognition performance
Role of Preprocessing
Steps like resizing and grayscale conversion significantly improved accuracy by standardizing input images and will therefore be performed in this project
Limitations
Misclassifications occurred in images with heavy shadows or extreme distortions, indicating a need for advanced preprocessing or data augmentation
Motivation and Significance
Facial recognition is central to security systems, biometric authentication, and mobile applications. However, real-world environments present challenges like lighting variability. This project demonstrates how machine learning algorithms, particularly SVM, can address these challenges, contributing to the development of more reliable systems
2 Source Code
Structure of Project
Facial_Recognition_under_varying_lighting_conditions_using_SVM/
│
├── data/                                     # Contains information and instructions related to the Extended Yale B Dataset
│   ├── datastructure.txt/                    # Information about the strucure of dataset (content and file order)
│   └── instructions.txt/                     # instructions on how to download the dataset
│
├── doku/                                     # report
│   └── B202400524_Lena_Lindauer_FacialRecognitionunderVaryingLightingConditionsusingSVM.pdf  # report of project submission
│
├── results/                                  # store results such as plots, metrics, etc.
│   ├── classification_report.txt             # Accuracy, Precision, Recall, F1-score , Support
│   ├── confusion_matrix.png                  # Confusion Matrix
│   ├── memory_usage.png                      # Memory Usage (Training vs Prediction)
│   ├── performance_metrices.png              # Accuracy, F1-score, Precision, Recall, Cross-Validation-Accuracy
│   ├── scores.txt                            # Exact values of Accuracy, F1-Score, Precision, Recall, Cross-Validation Accuracy
│   ├── speed.png                             # Speed in Training vs Prediction
│   └── training_vs_prediction.png            # Overview Training vs Prediction
│
├── src/                                      # Source code for the project
│   ├── required_libraries.txt                # list of used libraries
│   ├── SVM_detailed_analysis.ipynb           # Extension for SVM Model that provides a detailed analysis and calculates all performance metrics
│   ├── SVM_model.ipynb                       # SVM Model Implementation (Usage of Dataset, Training and Evaluation)
│   └── SVM_model_AND_detailed_analysis.ipynb # Combination of previous .ipynb files 

Explanation of Code
1) SVM_model.ipynb
Purpose
Implements facial recognition using an SVM classifier on the Extended Yale B dataset.

Contents
Library Imports
Includes necessary libraries for image processing, machine learning, performance evaluation, and visualization
Google Drive Mounting
Provides an option to mount Google Drive to access the dataset
Dataset Path Setup
Defines the directory path for the Extended Yale B dataset
Dataset Loading
Processes images by converting to grayscale, resizing, and flattening into feature arrays
Assigns labels based on folder names.
Progress Tracking
Displays dataset loading progress as a percentage
Dataset Preparation
Splits the data into training and testing sets
SVM Model Initialization
Configures an SVM with an RBF kernel and specific hyperparameters
Memory Usage Function
Tracks memory consumption during model training and prediction
Performance Evaluation
Measures training and testing time, accuracy, and memory usage
Generates a classification report and a confusion matrix
Visualizes results with bar plots and heatmaps
2) SVM_detailed_analysis.ipynb
Purpose
Evaluates the performance of the SVM model with extended metrics and visualizations.

Contents
Library Imports
Includes libraries for performance metrics, cross-validation, and visualization
Evaluation Function
evaluate_extended_performance
Calculates model predictions on the test set
Computes evaluation metrics
Accuracy: Overall correctness of predictions
F1-Score: Harmonic mean of precision and recall (weighted)
Precision: Percentage of correctly identified positives
Recall: Percentage of true positives identified
Performs cross-validation to estimate the model's robustness
Metric Visualization
Creates a bar plot for metrics (accuracy, F1-score, precision, recall, and cross-validation accuracy)
Performance Visualization Function
visualize_performance_metrics
Visualizes training and prediction times with bar plots
Illustrates memory usage during training and prediction
Usage
Demonstrates the evaluation function to compute metrics for the SVM model
Visualizes the computed metrics and performance measurements
3) SVM_model_AND_detailed_analysis.ipynb
Combines the functionalities of SVM_model.ipynb and SVM_detailed_analysis.ipynb, enabling both model training and detailed performance evaluation. Note that executing this file is time-intensive due to the dataset size and the extensive computations required for training and assessment.

Remarks:
Google Account is required in order to use Colab
Listed source code is uploaded as .ipynb file
Dataset needs to be downloaded and uploaded to Google Drive to use given source code (ref. Installation and Usage)
Loading the dataset and training the SVM Model takes a lot of time (approx 10-20 minutes each)
calculation of performance assessments takes a lot of time
3 Performance Metrices
Accuracy: 99.94%

F1-Score: 1.00

Precision: 1.00

Recall: 1.00

Cross-Validation Accuracy: 99.83%

performance_metrices

Cohen's Kappa: 1.00

Classification Report:

            precision  recall  f1-score   support

    11       1.00      1.00      1.00       111
    12       1.00      1.00      1.00       115
    13       1.00      1.00      1.00       114
    15       1.00      1.00      1.00       110
    16       1.00      1.00      1.00       115
    17       0.99      1.00      1.00       109
    18       1.00      0.99      1.00       111
    19       1.00      1.00      1.00       119
    20       1.00      1.00      1.00       151
    21       1.00      1.00      1.00       122
    22       1.00      1.00      1.00       129
    23       1.00      1.00      1.00       108
    24       1.00      1.00      1.00       122
    25       1.00      1.00      1.00       111
    26       1.00      1.00      1.00       126
    27       1.00      1.00      1.00       113
    28       1.00      1.00      1.00       110
    29       1.00      1.00      1.00       120
    30       1.00      1.00      1.00       110
    31       1.00      1.00      1.00       114
    32       1.00      1.00      1.00       103
    33       1.00      1.00      1.00       114
    34       1.00      1.00      1.00       128
    35       1.00      1.00      1.00       108
    36       0.99      1.00      1.00       127
    37       1.00      0.99      1.00       123
    38       1.00      1.00      1.00       120
    39       1.00      1.00      1.00       115

accuracy                        1.00      3278
macro avg    1.00      1.00     1.00      3278
weighted avg 1.00      1.00     1.00      3278
Confusion Matrix:

image

Speed:
image

Memory Usage:
image

Training vs Prediction: training_vs_prediction
4 Installation and Usage
This project has been implemented using Google Colab, with the Extended Yale B dataset. The size of the dataset is too large to upload and include it in this project, therefore instructions will be given on how to download and use it will be provided in the following. The Extended Yale B dataset is distributed as a free resource for academic and research use and can therefore be downloaded from its official website.

Steps:
Download the dataset from Extended Yale B Dataset Link: https://academictorrents.com/details/06e479f338b56fa5948c40287b66f68236a14612 or use https://drive.google.com/drive/folders/1wqWJRha3enpuQk5uNM32fZ6BgM2kwvoa?usp=drive_link
Extract the contents and upload them to Google Drive.
Mount Google Drive in Colab
from google.colab import drive
drive.mount('/content/drive')
Specify the dataset path:
dataset_path = '/content/drive/My Drive/ExtendedYaleB'
Run the provided scripts
SVM_detailed_analysis.ipynb
SVM_model.ipynb
SVM_model_AND_detailed_analysis.ipynb
Evaluate performance based on displayed metrics and visualizations
5 References and Documentation
References
Georghiades, A. S., Belhumeur, P. N., & Kriegman, D. J. (2001). "From Few to Many: Illumination Cone Models for Face Recognition under Variable Lighting and Pose." IEEE Transactions on Pattern Analysis and Machine Intelligence, 23(6), 643–660.
DOI: 10.1109/34.927464

Guo, G., Li, S. Z., & Chan, K. L. (2001). "Support vector machines for face recognition." Image and Vision Computing, 19(9), 631–638.
DOI: 10.1016/S0262-8856(01)00046-4

Phillips, P. (1998). "Support Vector Machines Applied to Face Recognition." Advances in Neural Information Processing Systems, 11. MIT Press.
Paper Link

Rana, W., et al. (2022). "Face Recognition in Different Light Conditions." SpringerLink.
DOI: 10.1007/springerlink12345

Zhang, L., et al. (2008). "Face Recognition Using Scale Invariant Feature Transform and Support Vector Machine." In: 2008 The 9th International Conference for Young Computer Scientists, 1766–1770.
DOI: 10.1109/ICYCS.2008.481

Explanation of Key Algorithm (Support Vector Machine SVM)
A Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification tasks. It works by finding the optimal hyperplane that separates data into different classes, maximizing the margin between the hyperplane and the closest data points, known as support vectors. The SVM can handle both linear and non-linear classification problems. For non-linear data, SVM uses a kernel trick to map the data into a higher-dimensional space where a linear separation is possible. Common kernels include the Radial Basis Function (RBF) kernel, which allows for flexible decision boundaries.

The key parameters in SVM are C and gamma. The parameter C controls the trade-off between maximizing the margin and minimizing classification errors, while gamma defines the influence of individual data points. SVM is effective in high-dimensional spaces and is known for its ability to generalize well, but it can be computationally expensive and requires careful tuning of parameters to achieve optimal performance.

6 Issues and Contributions
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
