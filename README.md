# PneumoniaCXR: AI-Enabled Pneumonia Detection

## Introduction

This proof-of-concept project explores the use of advanced machine learning techniques, including convolutional neural networks and radiomics, to enhance the classification of pneumonia from chest X-ray images. By leveraging state-of-the-art AI methods, the system dynamically adapts its analysis to accurately distinguish between COVID-19, Non-COVID pneumonia, and normal cases, ensuring high diagnostic accuracy and supporting rapid medical response.

## Goals

- To develop and demonstrate a proof-of-concept for an AI system that accurately classifies different types of pneumonia from chest X-ray images.
- To assess the impact of various configurations of deep learning architectures and feature extraction methods on the classification accuracy.
- To optimize system performance through extensive testing and refinement of model parameters.

## Dataset

This project utilizes the COVID-QU-Ex Dataset, which contains over 30,000 chest X-ray images labeled as COVID-19 positive, non-COVID infections, and normal cases. Gold standard validation data is provided to benchmark the system's classification quality against expert radiological assessments.

### Source:
**COVID-QU-Ex Dataset**: Anas M. Tahir, Muhammad E. H. Chowdhury, Yazan Qiblawey, Amith Khandakar, Tawsifur Rahman, Serkan Kiranyaz, Uzair Khurshid, Nabil Ibtehaz, Sakib Mahmud, and Maymouna Ezeddin, “COVID-QU-Ex .” Kaggle, 2021, [DOI: 10.34740/kaggle/dsv/3122958](https://doi.org/10.34740/kaggle/dsv/3122958). [Dataset available on Kaggle](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu).

## Methodology

1. **EDA and Data Preprocessing**:
   - Split data into train, validation, and test sets.
   - Resampled data to enable balanced classes.
   - Applied Z-scale normalization to images.
   - Adjusted cropping and alignment of images for uniformity.

2. **Feature Engineering**:
   - Extracted Histogram of Oriented Gradients (HOG), radiomics, and ResNet features.
   - Tuned HOG and radiomics extraction parameters for optimal performance.
   - Applied Principal Component Analysis (PCA) to reduce dimensionality and improve model efficiency.

3. **Modeling and Evaluation**:
   - Conducted hyperparameter tuning for models including SVM, logistic regression, random forest, and gradient boost.
   - Trained and tested models to evaluate their performance using metrics such as accuracy, precision, recall, and F1-score.

## Results

The results confirm that the PneumoniaCXR system effectively classifies chest X-ray images into COVID-19, non-COVID pneumonia, and normal cases with a high level of accuracy. The project achieved:
- 90% accuracy in model performance across different validation datasets, demonstrating good generalizability.
- Balanced importance among the different feature sets (HOG, radiomics, and ResNet), indicating that no single feature set dominated the predictive power.

## Usage

To utilize this project:
1. **Set up your Google Colab environment** to work with Google Drive by mounting your Google Drive in the Colab notebook. This will allow you to access and save files directly to your Google Drive.
2. **Clone the repository** into a specific folder within your Google Drive. This can be done using Git commands within a Colab notebook or by manually cloning the repository to your drive and then syncing with Google Drive.
3. **Download the data** from [Google Drive](https://drive.google.com/drive/folders/1Q44uT-VRO5vdfUt8FB4HovXdvRvj8SBd?usp=sharing) and ensure it is placed within the same directory as your cloned repository in Google Drive.
4. **Run the Jupyter notebooks** in the `notebooks/` directory from the cloned repository within Google Colab. All necessary dependencies are included in the notebooks and can be installed directly within Colab using `!pip install` commands.

Note: Detailed instructions and code examples are provided in the Jupyter notebooks within the repository. Ensure that the notebook paths correspond to the locations of your files in Google Drive for seamless execution.

## Future Work

While the current model demonstrates strong performance, future developments could include:
- Cross-validating generalizability across different CXR platforms and exploring CXR machine-specific models to tailor the system further for different imaging technologies and settings.
- Experimentation with additional model architectures and hybrid approaches to potentially enhance diagnostic accuracy.
- Extending the dataset to include more diverse platform, demographic and geographic data to improve model robustness and applicability in varied clinical environments.

## Contributors
- Gary Kong: Feature Engineering, Data-preprocessing (Support), PCA (Support)
- Drew Piispanen: Data Pre-processing (Lead)
- Diqing Wu: PCA (Lead), Model Training and Evaluation

## Project Organization

    ├── LICENSE
    ├── README.md                                                       <- The top-level README for developers using this project
    ├── data
    │   ├── raw                                                         <- The original, immutable raw dataset
    │   ├── preprocessed                                                <- Data after initial preprocessing
    │   ├── features_extracted                                          <- Data after feature extraction and selection
    │   └── features_PCA                                                <- Data after dimensionality reduction (PCA applied)
    ├── notebooks
    │   ├── 1.0-eda_data_preprocessing.ipynb                            <- Notebook for preprocessing images 
    │   ├── 2.0-feature_engineering.ipynb                               <- Notebook for feature extraction, selection and PCA
    │   └── 3.0-modelling                                               <- Notebook for model training and evaluation
    └── reports
        ├── PneumoniaCXR-AIDetect_Presentation_vF.pdf                   <- Final project presentation
        └── PneumoniaCXR-AIDetect_Report_vF.pdf                         <- Final project report

Note: Due to file size limitations on GitHub, the dataset is hosted on Google Drive. Please download the data from the provided Google Drive link before running the notebooks.
