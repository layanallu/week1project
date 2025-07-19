# 🌍 World Government Expenditure Dashboard

## 📌 Project Overview  
An interactive Streamlit dashboard for exploring, analyzing, and visualizing **government expenditure data** from 2000 to 2021.  
This app provides insights into:

- Government spending trends over time  
- Top funded sectors and countries  
- GDP percentage allocation per sector  
- Sector-wise comparisons  
- Health expenditure forecasting using machine learning  

---

## 📁 Dataset Description  

The dataset includes information on:

- **Country**: Name of the country  
- **Year**: From 2000 to 2021  
- **Sector**: Area of expenditure (e.g., Health, Education, Military)  
- **Expenditure (million USD)**: Government spending in million USD  
- **GDP (%)**: Share of expenditure as a percentage of GDP  

---

## 🎯 Features  

### 🧾 1. Dataset Overview & Cleaning  
- View shape, missing values. 
- Data cleaning preview with steps like removing nulls, cleaning columns, etc.

### 📊 2. Statistics per Country & Year  
- Mean, median, mode of spending  
- Visual filtering by country and year  

### 📈 3. Visualizations  
- Log-scaled histogram of expenditures  
- Correlation heatmap  
- Time-series: Health vs Education  
- Sector-level bar charts  
- GDP% visual breakdown per sector and country  

### ❓ 4. Key Questions Answered  
- Top 5 countries by total spending  
- Top 5 sectors globally  
- Countries with highest GDP% spending  
- Sector growth comparison (2000 vs 2021)  
- Most stable sectors (lowest standard deviation)  
- Average GDP% per sector   
- Global expenditure trends over time  

### 🤖 5. Machine Learning Forecast  
- Random Forest Regression 
- R² score and Mean Squared Error shown in dashboard  

---

## 🛠️ How to Run the App  

### 1. Clone the Repository:
```bash
git clone https://github.com/your-username/world-expenditure-dashboard.git
cd world-expenditure-dashboard
```
### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App:
```bash
streamlit run WorldExpenditures.py
```

