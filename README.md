 🏠 California Housing Price Prediction

This project is part of the **Regression Analysis - Spring 2025** course at Alexandria National University.  
It presents a complete machine learning pipeline for predicting housing prices in California using regression models and an interactive **Streamlit** web application.

---

 📊 Project Overview

We use the **California Housing dataset** (20,000+ records) to explore, analyze, and model median house values based on geographic and demographic factors.

---

🔧 Features

✅ Exploratory Data Analysis (EDA):  
- Histograms, Boxplots  
- Correlation Heatmap  
- Geographical data visualization (Latitude & Longitude)  

✅ Data Preparation & Cleaning:  
- Outlier capping  
- Feature engineering (e.g., `RoomsPerPopulation`, `IncomePerRoom`)  
- Missing value detection

✅ Modeling:  
- Linear Regression  
- Ridge, Lasso, and ElasticNet  
- Polynomial Regression (degree 2)  
- Optional PCA (dimensionality reduction)  
- Feature selection (`SelectKBest`)  
- Cross-validation (configurable folds)  
- Residual analysis and Q-Q plots

✅ Deployment:  
- Interactive **Streamlit** interface  
- Real-time price prediction based on user inputs  
- Display of feature importance and contribution

---

🧪 Technologies Used

- Python 🐍  
- Streamlit 📱  
- scikit-learn  
- pandas, numpy  
- seaborn, matplotlib  
- statsmodels  
- joblib  
- Pillow (for images)

---

 🚀 How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/yourusername/california-housing-regression.git
cd california-housing-regression

2.Install the requirements:

pip install -r requirements.txt

3.Run the Streamlit app:

streamlit run app.py
