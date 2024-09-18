
# TensorFlow MNIST Example with MLflow (CUDA Enabled)

This repository contains two examples demonstrating how to train a simple neural network on the MNIST dataset using TensorFlow with GPU acceleration (CUDA) and track the experiments and models using MLflow. The two Jupyter notebooks illustrate how to:

- Train a model and register it in the MLflow Model Registry.
- Load the registered model from MLflow and use it for predictions.

## Structure

```plaintext
.
├── tensorflow-mnist-train-mlflow.ipynb   # Notebook to train and register the model with MLflow
└── tensorflow-mnist-predict-mlflow.ipynb # Notebook to load the registered model and run predictions
```

### 1. `tensorflow-mnist-train-mlflow.ipynb`

This notebook demonstrates the following:
- Loading the MNIST dataset and building a simple neural network model with TensorFlow.
- Training the model with CUDA (GPU acceleration).
- Tracking the training process (metrics, parameters, and artifacts) using MLflow's autologging feature.
- Registering the trained model in the MLflow Model Registry.

### 2. `tensorflow-mnist-predict-mlflow.ipynb`

This notebook demonstrates how to:
- Load the registered model from the MLflow Model Registry.
- Use the loaded model to run predictions on new data (MNIST test dataset).
- Compare the predicted results with actual labels.

## Requirements

- TensorFlow (with CUDA support)
- MLflow
- Python 3.x
- Jupyter Notebook

To install the required packages, you can run:

```bash
pip install tensorflow mlflow jupyter
```

Make sure your environment is set up with CUDA to enable GPU acceleration.

## Tested Environment

These examples were tested using the **Ubuntu Data Science Stack** tool running MLflow 2.1.1 and the `kubeflownotebookswg/jupyter-tensorflow-cuda-full:v1.8.0` TensorFlow image. You can refer to the full documentation for the Ubuntu Data Science Stack [here](https://documentation.ubuntu.com/data-science-stack/en/latest/).

The Ubuntu Data Science Stack provides a ready-to-use environment for data science workflows, including GPU-enabled TensorFlow, Jupyter notebooks, and MLflow for experiment tracking.

## Running the Examples

1. **Training and Registering the Model**: 
   Open and run `tensorflow-mnist-train-mlflow.ipynb`. This will train the model on the MNIST dataset and automatically log the model and training metrics in MLflow. The trained model will also be registered in the MLflow Model Registry.

2. **Loading and Predicting with the Model**: 
   After training, open and run `tensorflow-mnist-predict-mlflow.ipynb`. This will load the registered model from MLflow and use it to predict the classes of the MNIST test dataset.

## MLflow UI

To learn how to deploy and access the MLflow UI, refer to the [MLflow section in the Ubuntu Data Science Stack documentation](https://documentation.ubuntu.com/data-science-stack/en/latest/how-to/mlflow/), which outlines the steps for deployment and accessing MLflow in your environment.
