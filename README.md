# 📊 Bankruptcy Prediction System Using Machine Learning

![Bankruptcy Prediction System image](https://github.com/user-attachments/assets/b1f7cc2e-8e78-4935-9017-62b25b4ce1b3)


This project is a full machine learning pipeline for predicting company bankruptcy based on financial and operational features. It uses multiple classification algorithms and is deployed via Gradio and Hugging Face Spaces.

## 🚀 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/AdMub/bankruptcy-prediction-app)

---

## 🧠 Models Implemented

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Multilayer Perceptron (MLP)**

---

## 🔧 Features

- **Data Preprocessing**
  - Feature selection using `SelectKBest` with ANOVA F-test
  - Balancing with SMOTE
- **Model Training and Evaluation**
  - Accuracy, Precision, Recall, F1-score, ROC AUC
  - Confusion Matrix visualization
- **Hyperparameter Tuning**
- **Interactive Prediction UI via Gradio**
- **Cloud Deployment using Hugging Face Spaces and AWS**

---

## 📁 Project Structure

```plain
bankruptcy_prediction/
├── bankruptcy_model.py # Main ML pipeline script
├── app.py # Gradio app for UI
├── requirements.txt # Python dependencies
├── model.pkl # Trained model
└── README.md # Project documentation
```


---

## 🖥️ Technologies Used

- **Python**
- **scikit-learn**
- **xgboost**
- **imbalanced-learn**
- **Gradio**
- **Google Colab**
- **AWS (S3 & SageMaker)**
- **Hugging Face Spaces**

---

## 📊 Dataset

- Financial dataset of companies with binary labels for bankruptcy.
- Feature selection reduced dimensions from 96 to 20 using statistical relevance.

---

## 📉 Results

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.87     | 0.88      | 0.85   | 0.86     | 0.92    |
| Random Forest       | 0.91     | 0.92      | 0.90   | 0.91     | 0.95    |
| XGBoost             | **0.93** | **0.94**  | 0.92   | 0.93     | **0.97**|
| MLP Classifier      | 0.89     | 0.90      | 0.87   | 0.88     | 0.94    |

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/bankruptcy-prediction.git
cd bankruptcy-prediction
```

#### Install the dependencies:
```bash
pip install -r requirements.txt
```

#### Run the app locally:
```bash
python app.py
```

## 🌐 Deployment
- Gradio App: Built for local or web-based UI.
- Hugging Face: Easily hosted with app.py and requirements.txt.
- AWS S3/SageMaker: Model files and training logs available for cloud workflows.

## 🤔 Future Improvements
- Add more interpretability (e.g., SHAP or LIME)
- Improve feature engineering
- Incorporate financial ratios or temporal data
- Deploy with containerized solutions (Docker + AWS ECS)

## 👨‍💻 Author
**AdMub**
- 📚 University of the People & University of Ibadan
- 🔗 LinkedIn | Twitter

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
