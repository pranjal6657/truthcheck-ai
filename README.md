📰 TruthCheck AI

A full-stack fake news detection system that combines Machine Learning (NLP) and Google Fact Check API to analyze the credibility of news articles and claims.

🚀 Features
Real-time fake news analysis
NLP-based text classification
Google Fact Check API integration
Confidence scoring
Rate limiting protection
Response caching
Secure URL validation
Modern responsive UI
🛠️ Tech Stack
Backend
Python
Flask
Scikit-learn
NLTK
Requests
BeautifulSoup4
Frontend
HTML
CSS
JavaScript
Machine Learning
TF-IDF Vectorization
SGDClassifier
External Services
Google Fact Check Tools API
📂 Project Structure
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
🧠 How It Works
User enters news text or article URL.
Backend cleans and preprocesses the text.
TF-IDF converts text into numerical features.
Machine Learning model predicts:
Real
Fake
Uncertain
Google Fact Check API verifies known claims.
Results are combined and displayed with confidence scores.
🔒 Security Features
URL validation
SSRF protection
Request rate limiting
API key protection through environment variables
Response caching
⚙️ Installation
Clone Repository
git clone https://github.com/pranjal6657/truthcheck-ai.git
cd truthcheck-ai
Install Dependencies
pip install -r backend/requirements.txt
Set Environment Variable
set FACT_CHECK_API_KEY=YOUR_API_KEY
Run Application
python backend/app.py
📊 Machine Learning Model

Algorithm:

SGDClassifier

Text Processing:

TF-IDF Vectorization
Stopword Removal
Text Cleaning
🎯 Future Improvements
BERT-based classification
Multi-language support
User authentication
Analysis history
Advanced fact-check aggregation
News source credibility scoring
