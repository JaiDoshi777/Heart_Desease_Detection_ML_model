# Heart Disease Prediction using PyTorch

An end-to-end Deep Learning project that predicts the probability of heart disease in patients based on medical attributes. This project utilizes a custom-built Neural Network in **PyTorch**, featuring data preprocessing, early stopping mechanisms, and comprehensive performance visualization.

## 📌 Project Overview
The goal of this project is to build a binary classifier capable of distinguishing between the presence and absence of heart disease. It processes 13 clinical features (such as Age, Sex, Cholesterol, BP, etc.) to output a diagnosis prediction.

**Key Highlights:**
* **Deep Learning:** Implemented a Feedforward Neural Network using PyTorch.
* **Optimization:** Utilized **Early Stopping** to prevent overfitting and ensure optimal generalization.
* **GPU Acceleration:** Code is optimized to run on CUDA-enabled GPUs for faster training.
* **Visualization:** Includes Confusion Matrix, ROC Curves, and Loss History graphs.

## 📂 Dataset
The dataset used for this project is the **Heart Disease Prediction Dataset** sourced from Kaggle.
* **Source:** [Kaggle - Heart Disease Prediction](https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction)
* **Attributes:** 14 columns (13 Features + 1 Target)
* **Target Variable:** `Heart Disease` (Presence/Absence)

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Data Manipulation:** Pandas, NumPy
* **Preprocessing:** Scikit-Learn (StandardScaler, LabelEncoder)
* **Visualization:** Matplotlib, Seaborn

## ⚙️ Model Architecture
The model is a fully connected neural network (Multi-Layer Perceptron) with the following structure:
1.  **Input Layer:** 13 Neurons (matching input features)
2.  **Hidden Layer 1:** 16 Neurons + ReLU Activation
3.  **Hidden Layer 2:** 8 Neurons + ReLU Activation
4.  **Output Layer:** 1 Neuron + Sigmoid Activation (Probability output)

## 📊 Results & Performance
The model achieves a test accuracy of more than 90% (depending on the random seed).
* **GRAPH OF MODEL TRAINING AND VALIDATION LOSS OVER THE NUMBER OF EPOCHS:**
<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/67daf419-a885-48d4-88d8-a2f713701af7" />
  
* **CONFUSION MATRIX:**
<img width="485" height="451" alt="image" src="https://github.com/user-attachments/assets/01b60350-06ca-44b6-bb24-c18ea0c59562" />

* **ROC-AUC:** Demonstrates strong separability between classes.
<img width="485" height="451" alt="image" src="https://github.com/user-attachments/assets/c5dd65f3-318e-4a8c-8d3c-980dae3311c2" />
  

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/heart-disease-prediction.git](https://github.com/yourusername/heart-disease-prediction.git)
    cd heart-disease-prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Run the training script:**
    ```bash
    python train_model.py
    ```

## 📈 Future Improvements
* Implement Hyperparameter Tuning (GridSearch) to further improve accuracy.
* Deploy the model as a REST API using Flask or FastAPI.
* Test with other architectures (e.g., Random Forest, XGBoost) for performance comparison.
