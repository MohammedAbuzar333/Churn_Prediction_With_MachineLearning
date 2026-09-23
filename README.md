# ChurnGuard AI — Streamlit Frontend

## Files
- `app.py` — complete Streamlit dashboard
- `requirements.txt` — required Python packages

## Dataset
Place your dataset in the same folder as `app.py` with one of these names:
- `online_retail_customer_churn.csv`
- `customer_churn.csv`
- `churn.csv`

Or upload the CSV from the sidebar.

The dataset used in the project notebook contains:
Customer_ID, Age, Gender, Annual_Income, Total_Spend, Years_as_Customer,
Num_of_Purchases, Average_Transaction_Amount, Num_of_Returns,
Num_of_Support_Contacts, Satisfaction_Score, Last_Purchase_Days_Ago,
Email_Opt_In, Promotion_Response, Target_Churn.

## Run
Open a terminal in this folder and run:

```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn
streamlit run churnguard_streamlit_app.py
```

The app includes:
- Bright dashboard theme
- Startup loading animation
- Dataset explorer
- Churn infographic cards
- Interactive typed numeric ranges
- Customer churn prediction
- Model comparison
- Project workflow
- SMOTE training pipeline
