# Famous-Personality-Image-Classification

**Project Overview**

This project focuses on building a machine learning model to classify sports celebrities from images. A dataset of 17 celebrities was used, sourced from Kaggle. The primary goal was to preprocess the images effectively, extract meaningful features, and train a robust classification model.

**Data Preprocessing**

Face Detection: Used OpenCV’s Haar Cascade Classifier to detect faces and crop images. Only images with both eyes visible were considered to improve training quality.

Feature Engineering: Applied Wavelet Transform (using pywt) to extract high-frequency details from images. Combined these features with raw pixel data to create a comprehensive feature set.

**Model Training**

-> Explored multiple algorithms: SVM, Random Forest, and Logistic Regression.

-> Used GridSearchCV for hyperparameter tuning to find the best model configuration.

-> SVM was selected as the final model due to superior performance on evaluation metrics.

**Deployment**

-> Trained model saved using joblib.

-> Class mapping saved in JSON for easy integration.

-> The project is structured to support deployment in a web application.

**Key Skills Demonstrated**

-> Computer Vision with OpenCV

-> Feature Engineering using Wavelet Transforms

Model Selection and Hyperparameter Tuning with GridSearchCV

Deployment-Ready Machine Learning Pipelines
