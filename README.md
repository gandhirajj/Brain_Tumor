# 🧠 AI-Powered Brain Tumor Identification  
*Using Machine Learning to Improve CNN Diagnostic Precision*  

## 📌 Project Overview  
This project leverages **Convolutional Neural Networks (CNNs)** to identify brain tumors from MRI scans with high accuracy.  
By combining **deep learning techniques** with **medical imaging**, the system provides an automated diagnostic tool that assists healthcare professionals in early detection and classification of brain tumors.  

Our model achieves **~99% accuracy** in distinguishing between tumor and non-tumor brain MRI scans, outperforming traditional machine learning classifiers like SVM.  

---

## 👨‍💻 Authors  
- **P. Charanya**, Assistant Professor, Sri Eshwar College of Engineering  
- **Gandhiraj J**, Department of AI & ML, Sri Eshwar College of Engineering  
- **Buvanes E**, Department of AI & ML, Sri Eshwar College of Engineering  
- **Prateeksha GV**, Department of AI & ML, Sri Eshwar College of Engineering  
- **Theertha R**, Department of AI & ML, Sri Eshwar College of Engineering  

Paper Link: https://liberteresearch.org/volume-13-issue-2-2025/
---

## ⚙️ Methodology  
### 🔹 Data Preprocessing  
- Extract MRI images from dataset (ZIP format).  
- Normalize pixel values (0–1 scaling).  
- Data Augmentation using `ImageDataGenerator` (rotation, zoom, flipping).  

### 🔹 Model Architecture (CNN)  
1. Input Layer: MRI scans resized to **64x64x3**.  
2. **Conv2D Layers**: Extract spatial features using 3x3 filters.  
3. **MaxPooling2D Layers**: Reduce dimensions & prevent overfitting.  
4. **Flatten Layer**: Convert feature maps into a 1D vector.  
5. **Dense Layers**:  
   - Dense-1: 128 neurons, ReLU activation  
   - Dense-2: 128 neurons, ReLU activation  
6. **Output Layer**: Single neuron with **Sigmoid activation** for binary classification.  

### 🔹 Model Training  
- Optimizer: **Adam**  
- Loss Function: **Binary Crossentropy**  
- Evaluation Metric: **Accuracy**  
- Epochs: **20** (adjustable)  

### 🔹 Model Evaluation  
- Metrics: **Accuracy, Precision, Recall, F1-Score**  
- Tools: **Confusion Matrix, Classification Report**  
- Achieved **99% accuracy** compared to **83% with SVM**.  

---

## 📊 Results  
- **Training Accuracy**: ~99%  
- **Validation Accuracy**: ~97%  
- **Confusion Matrix**: High precision in detecting tumor cases.  
- **F1-Score**: 0.90 for tumor detection.  

---

## 🚀 Future Work  
- Incorporate advanced CNN architectures (**ResNet, Inception, EfficientNet**).  
- Expand dataset for better generalization.  
- Develop **real-time tumor detection system** for clinical use.  
- Integrate with **explainable AI** methods for better medical interpretation.  

---

## 🛠️ Tech Stack  
- **Python 3.10+**  
- **TensorFlow / Keras**  
- **NumPy, Pandas, Matplotlib**  
- **scikit-learn**  

---

## 📂 Repository Structure  

├── data/                 # MRI dataset (Tumor / No\_Tumor)
├── models/               # Saved trained CNN models
├── notebooks/            # Jupyter notebooks for experiments
├── results/              # Evaluation reports & plots
└── README.md             # Project documentation

## 📄 License  
This project is licensed under the **MIT License** – feel free to use and modify for research and educational purposes.  

---

