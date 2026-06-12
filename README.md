# 📰 TruthCheck AI

A full-stack Fake News Detection System that combines Machine Learning (NLP) and Google Fact Check API to analyze the credibility of news articles and online claims.

---

## 🚀 Features

- 🔍 Real-time fake news analysis
- 🧠 NLP-based machine learning classification
- 🌐 Google Fact Check API integration
- 📊 Confidence scoring system
- ⚡ Response caching for faster results
- 🛡️ Rate limiting protection
- 🔒 Secure URL validation and SSRF protection
- 🎨 Modern responsive frontend

  ![Uploading Screenshot 2026-06-12 101708.png…]()


---

## 🛠️ Tech Stack

### Backend
- Python
- Flask
- Scikit-Learn
- NLTK
- Requests
- BeautifulSoup4

### Frontend
- HTML5
- CSS3
- JavaScript

### Machine Learning
- TF-IDF Vectorization
- SGDClassifier

### APIs
- Google Fact Check Tools API

---

## 📂 Project Structure

```bash
truthcheck-ai/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── model/
│   │   ├── model.pkl
│   │   └── vectorizer.pkl
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── README.md
```

---

## 🧠 How It Works

1. User enters a news article or claim.
2. Text is cleaned and preprocessed.
3. TF-IDF converts text into numerical vectors.
4. Machine Learning model predicts:
   - Real
   - Fake
   - Uncertain
5. Google Fact Check API verifies known claims.
6. Results are combined and displayed with confidence scores.

---

## 🔒 Security Features

- URL Validation
- SSRF Protection
- Rate Limiting
- Environment Variable API Keys
- Response Caching
- Input Sanitization

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/pranjal6657/truthcheck-ai.git
cd truthcheck-ai
```

### Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### Configure API Key

Create a `.env` file:

```env
FACT_CHECK_API_KEY=your_api_key_here
```

### Run Application

```bash
python backend/app.py
```

---

## 📊 Machine Learning Model

| Component | Technology |
|------------|------------|
| Vectorization | TF-IDF |
| Classifier | SGDClassifier |
| NLP | NLTK |
| Verification | Google Fact Check API |

---

