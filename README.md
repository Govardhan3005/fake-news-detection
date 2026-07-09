# 🔍 Fake News Detection using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=for-the-badge&logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**An intelligent AI-powered application that detects whether a news article is REAL or FAKE using Natural Language Processing and Machine Learning.**

[🚀 Live Demo](#how-to-run) • [📊 Model Performance](#model-performance) • [📁 Project Structure](#project-structure) • [👤 Author](#author)

</div>

---

## 📌 Project Overview

Fake news has become one of the most serious challenges in the digital age — spreading misinformation rapidly across social media and online platforms. This project builds a complete **end-to-end Machine Learning pipeline** to automatically classify news articles as **REAL** or **FAKE**.

The project includes:
- ✅ A complete **ML pipeline** from raw data to predictions
- ✅ A **beautiful web chatbot UI** built with Streamlit
- ✅ **Image-based detection** via OCR (upload newspaper photos or screenshots)
- ✅ **6 ML models** trained and compared
- ✅ **TF-IDF vectorization** with NLP preprocessing
- ✅ **Production-ready** modular code following PEP8

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Chat Interface** | Type or paste any news article and get instant predictions |
| 📸 **Image OCR** | Upload newspaper clippings or screenshots — AI extracts and analyzes text |
| 🤖 **6 ML Models** | Logistic Regression, Naive Bayes, Random Forest, Decision Tree, SVM, Passive Aggressive |
| 📊 **Confidence Score** | Probability breakdown for REAL vs FAKE with visual confidence bar |
| 🧹 **NLP Preprocessing** | URL removal, HTML stripping, stopword removal, lemmatization |
| 📈 **EDA Visualizations** | Word clouds, bigrams, trigrams, class distribution, ROC curves |
| 💾 **Model Persistence** | Trained model saved with Joblib for instant reloading |
| 📱 **Responsive UI** | Dark glassmorphism design, works on all screen sizes |
| 🕓 **Chat History** | Full session history with image analysis log |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **ML Library** | Scikit-learn |
| **NLP** | NLTK, Regex |
| **Vectorization** | TF-IDF (Scikit-learn) |
| **OCR** | EasyOCR |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Web App** | Streamlit |
| **Data** | Pandas, NumPy |
| **Model Saving** | Joblib |
| **Version Control** | Git & GitHub |

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── 📁 dataset/                  # Place your dataset CSV files here
│
├── 📁 models/
│   ├── model.pkl                # Trained ML model (generated after training)
│   └── vectorizer.pkl           # Fitted TF-IDF vectorizer
│
├── 📁 notebook/
│   └── FakeNewsDetection.ipynb  # Jupyter notebook with full EDA + training
│
├── 📁 screenshots/              # App screenshots for README
│
├── 🐍 app.py                    # Streamlit chatbot web application
├── 🐍 train.py                  # Model training script
├── 🐍 predict.py                # Prediction module (CLI + importable)
├── 🐍 preprocess.py             # Text preprocessing pipeline
│
├── 📄 requirements.txt          # All Python dependencies
├── 📄 README.md                 # Project documentation
├── 📄 LICENSE                   # MIT License
└── 📄 .gitignore                # Git ignore rules
```

---

## 📊 Dataset

This project supports any fake news dataset. Recommended datasets:

| Dataset | Source | Size |
|---------|--------|------|
| **Fake and Real News Dataset** | [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) | 44,000+ articles |
| **LIAR Dataset** | [UCSB](https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip) | 12,800+ statements |
| **WELFake Dataset** | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) | 72,000+ articles |

> ⚠️ **Note:** Place your CSV dataset in the `dataset/` folder before running `train.py`

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 📋 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
nltk>=3.8.1
regex>=2023.8.8
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.2
streamlit>=1.28.0
easyocr>=1.7.0
Pillow>=10.0.0
tqdm>=4.66.0
```

---

## 🚀 How to Run

### Step 1 — Train the Model
```bash
python train.py
```
This will:
- Load and preprocess the dataset
- Train 6 ML models
- Compare model performance
- Save the best model as `models/model.pkl`

### Step 2 — Launch the Web App
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

### Step 3 — (Optional) CLI Prediction
```bash
python predict.py "Breaking: Scientists discover new planet in solar system"
```

---

## 🖥️ App Screenshots

### 💬 Text Chat Analysis
> Paste any news article and get an instant REAL / FAKE verdict with confidence score

### 📸 Image OCR Analysis
> Upload a newspaper clipping or screenshot — AI extracts text via OCR and detects fake news

---

## 📈 Model Performance

> Results will be updated after training with the actual dataset

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | - | - | - | - |
| Passive Aggressive | - | - | - | - |
| Linear SVM | - | - | - | - |
| Random Forest | - | - | - | - |
| Naive Bayes | - | - | - | - |
| Decision Tree | - | - | - | - |

---

## 🔮 Future Improvements

- [ ] Integrate BERT / RoBERTa transformer models
- [ ] Add multi-language support (Hindi, Tamil, etc.)
- [ ] Real-time news URL scraping and fact-checking
- [ ] Browser extension for instant news verification
- [ ] Mobile app version
- [ ] API endpoint using FastAPI
- [ ] Integrate FactCheck.org / Snopes database
- [ ] Add explainability (SHAP / LIME)

---

## 🎓 Interview Preparation

<details>
<summary>📚 Click to expand 20 Interview Questions</summary>

1. What is TF-IDF and how does it work?
2. Why is text preprocessing important in NLP?
3. How does Logistic Regression work for text classification?
4. What is the difference between stemming and lemmatization?
5. Why did you choose TF-IDF over Word2Vec?
6. How do you handle class imbalance in fake news detection?
7. What is the Passive Aggressive Classifier?
8. Explain Precision, Recall, and F1-Score with examples.
9. What is a Confusion Matrix?
10. How does the Naive Bayes algorithm work for text?
11. What is n-gram? How do bigrams and trigrams help?
12. How do you prevent overfitting in ML models?
13. What is cross-validation and why is it important?
14. How does EasyOCR extract text from images?
15. What is the ROC curve and AUC score?
16. How would you deploy this app in production?
17. Why is Random Forest better than a single Decision Tree?
18. What are stopwords and why do we remove them?
19. How does the Streamlit app communicate with the ML model?
20. What are the limitations of ML-based fake news detection?

</details>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Govardhan N**

---

<div align="center">

⭐ **Star this repository if you found it helpful!** ⭐

Made with ❤️ using Python, Scikit-learn & Streamlit

</div>
