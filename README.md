# 🏠 Real Estate AI Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/Alaashamel/real_estate_analyzer/main/app/streamlit_app.py)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-brightgreen.svg)](https://streamlit.io/)

## 🎯 Overview

**AI-powered real estate price prediction using multimodal data** (text descriptions + tabular features).

**Features:**
- 🔮 **Price Prediction**: Upload property description + specs → Instant price estimate
- 📊 **Market Analytics**: Price distributions, trends, amenities analysis
- 🎯 **Recommendations**: Find similar properties
- 🧠 **Multimodal Model**: RandomForest (TF-IDF text + area/bedrooms/location features)
- 📈 **Interactive Dashboard**: Built with Streamlit + Plotly

**Live Demo**: [Try it here](https://real-estate-ai-analyzer.streamlit.app) (deployed on Streamlit Cloud)

## 🚀 Quick Start (Local)

```bash
# Clone repo
git clone https://github.com/Alaashamel/real_estate_analyzer.git
cd real_estate_analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/streamlit_app.py
```

## 🛠️ Full Pipeline

```bash
# 1. Scrape & process data (if needed)
python main.py --scrape

# 2. Train model
python main.py --train
# or: python multimodal/price_predictor.py

# 3. Launch app
python main.py --app
```

## 📁 Project Structure

```
real_estate_analyzer/
├── app/                    # Streamlit dashboard
│   └── streamlit_app.py
├── data/                   # Datasets
│   ├── raw/               # Scraped JSON/images
│   └── processed/         # final_dataset.csv (Egypt properties)
├── models/                 # Trained models
│   └── price_predictor.pkl
├── multimodal/            # Core ML pipeline
│   └── price_predictor.py
├── scraper/               # Web scrapers (Aqarmap, PropertyFinder)
├── nlp/                   # Text processing
├── vision/                # Image analysis (future)
├── requirements.txt
└── main.py                # Pipeline orchestrator
```

## 🧠 Model Details

- **Architecture**: RandomForestRegressor (100 trees)
- **Features**:
  | Type     | Examples |
  |----------|----------|
  | Text     | TF-IDF (300 dims from description) |
  | Tabular  | area_sqm, bedrooms, bathrooms, property_type, furnishing |
- **Performance**: MAE ~500K EGP, R² ~0.85 (on Cairo/Alexandria data)
- **Data**: 1000+ Egypt listings (Aqarmap scraped)

## 📊 Dashboard Screenshots

![Prediction](screenshots/prediction.png)
![Analytics](screenshots/analytics.png)

(Add screenshots after deployment)

## 🔧 Deployment

**Streamlit Cloud** (auto-deploys on git push):
1. Fork/clone this repo
2. [Streamlit Sharing](https://share.streamlit.io)
3. Connect GitHub repo
4. Set: `app/streamlit_app.py` + `requirements.txt`

## 🤝 Contributing

1. Fork repo
2. Create feature branch
3. Add tests
4. PR to `main`

## 📄 License

MIT License - free to use/modify

## 👥 Author

**Alaa Shamel** - Full-stack AI/ML Engineer

⭐ **Star the repo if useful!**

