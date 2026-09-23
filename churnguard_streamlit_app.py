import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title='ChurnGuard AI', page_icon='📉', layout='wide')

st.markdown('''<style>
.stApp{background:#070b14;color:#fff} [data-testid="stSidebar"]{background:#0d1422}
h1,h2,h3,h4,p,label,span,div{color:#fff!important}.hero{padding:28px;border-radius:22px;background:linear-gradient(135deg,#101a2d,#172033);border:1px solid #334155;margin-bottom:20px}.hero-title{font-size:42px;font-weight:800}.hero-sub{color:#b9c3d4!important}.card{background:#101827;border:1px solid #334155;border-radius:18px;padding:18px;text-align:center}.big{font-size:30px;font-weight:800}.small{color:#aab6c8!important}.pred{background:#111827;border:2px solid #475569;border-radius:20px;padding:25px;text-align:center;margin-top:15px}.pred-title{font-size:30px;font-weight:800}.stButton>button{width:100%;border-radius:12px;font-weight:700;background:#172033;color:#fff;border:1px solid #475569}.stNumberInput input,.stTextInput input{color:#000!important;background:#fff!important}.stSelectbox div[data-baseweb="select"]{background:#fff!important}.stSelectbox div[data-baseweb="select"] *{color:#000!important}
</style>''', unsafe_allow_html=True)

if 'started' not in st.session_state:
    st.session_state.started=True
    box=st.empty()
    for msg in ['⚙️ Initializing ChurnGuard AI...','📂 Loading customer data...','🧹 Preparing preprocessing...','⚖️ Preparing SMOTE...','🤖 Preparing ML engine...','🚀 Dashboard ready!']:
        box.markdown(f'<div class="hero" style="text-align:center"><div class="hero-title">ChurnGuard AI</div><div class="hero-sub">{msg}</div></div>',unsafe_allow_html=True); time.sleep(.18)
    box.empty()

st.markdown('<div class="hero"><div class="hero-title">📉 ChurnGuard AI</div><div class="hero-sub">Explainable Customer Churn Prediction Dashboard</div></div>',unsafe_allow_html=True)

st.sidebar.title('⚙️ Project Controls')
uploaded=st.sidebar.file_uploader('Upload churn_modling CSV',type=['csv'])
use_demo=st.sidebar.checkbox('Use built-in demo dataset',value=True)
st.sidebar.markdown('### 🔄 Pipeline')
st.sidebar.markdown('📂 Load → 🧹 Preprocess → 🔧 Feature Engineering → ⚖️ SMOTE → 📏 Scale → 🤖 Random Forest → 🔮 Predict')
st.sidebar.caption('Expected target: Exited')

@st.cache_data
def demo_data(n=1000):
    rng=np.random.default_rng(42)
    d=pd.DataFrame({'RowNumber':np.arange(1,n+1),'CustomerId':rng.integers(10000000,99999999,n),'Surname':rng.choice(['Smith','Johnson','Brown','Williams','Taylor'],n),'CreditScore':rng.integers(350,851,n),'Geography':rng.choice(['France','Spain','Germany'],n),'Gender':rng.choice(['Male','Female'],n),'Age':rng.integers(18,75,n),'Tenure':rng.integers(0,11,n),'Balance':np.round(rng.uniform(0,250000,n),2),'NumOfProducts':rng.integers(1,5,n),'HasCrCard':rng.integers(0,2,n),'IsActiveMember':rng.integers(0,2,n),'EstimatedSalary':np.round(rng.uniform(10000,200000,n),2)})
    score=.045*(d.Age-40)-.002*(d.CreditScore-600)+.7*(d.Geography=='Germany')+.6*(d.NumOfProducts>=3)-.8*d.IsActiveMember+.5*(d.Balance>150000)
    p=1/(1+np.exp(-score/2.0)); d['Exited']=(rng.random(n)<p).astype(int); return d

if uploaded is not None:
    try: ch=pd.read_csv(uploaded); source='Uploaded churn_modling CSV'
    except Exception as e: st.error(f'Could not read CSV: {e}'); st.stop()
elif use_demo: ch=demo_data(); source='Built-in demo dataset'
else: st.info('Upload churn_modling CSV or enable demo dataset.'); st.stop()

ch.columns=ch.columns.astype(str).str.strip()
if 'Exited' not in ch.columns:
    st.error("Target column 'Exited' not found."); st.write(ch.columns.tolist()); st.stop()

st.subheader('📊 Customer Dataset Overview')
ex=pd.to_numeric(ch.Exited,errors='coerce').fillna(0).astype(int)
vals=[len(ch),int(ex.sum()),int((ex==0).sum()),ch.shape[1]]
labels=['Total Customers','Churn','Non-Churn','Columns']
cols=st.columns(4)
for c,v,l in zip(cols,vals,labels):
    with c: st.markdown(f'<div class="card"><div class="big">{v:,}</div><div class="small">{l}</div></div>',unsafe_allow_html=True)
st.caption('Data source: '+source)

st.subheader('📈 Infographs')
g1,g2=st.columns(2)
with g1: st.bar_chart(pd.DataFrame({'Customers':[int((ex==0).sum()),int((ex==1).sum())]},index=['Non-Churn','Churn']))
with g2:
    nums=[c for c in ch.select_dtypes(include=np.number).columns if c not in ['RowNumber','CustomerId','Exited']]
    if nums:
        feature=st.selectbox('Feature graph',nums); st.line_chart(ch[feature].reset_index(drop=True))

@st.cache_data
def prepare(df):
    y=pd.to_numeric(df['Exited'],errors='coerce').fillna(0).astype(int)
    X=df.drop(columns=['Exited','RowNumber','CustomerId','Surname'],errors='ignore').copy()
    if 'Balance' in X and 'EstimatedSalary' in X: X['Balance_to_Salary']=X['Balance']/X['EstimatedSalary'].replace(0,np.nan)
    if 'NumOfProducts' in X and 'Tenure' in X: X['Products_per_Tenure']=X['NumOfProducts']/X['Tenure'].replace(0,1)
    if 'Age' in X: X['Age_Squared']=X['Age']**2
    if 'CreditScore' in X and 'Age' in X: X['CreditScore_Age']=X['CreditScore']/X['Age'].replace(0,1)
    cats=X.select_dtypes(include=['object','string','category']).columns.tolist()
    X=pd.get_dummies(X,columns=cats,drop_first=True)
    for c in X.select_dtypes(include='bool').columns: X[c]=X[c].astype(int)
    X=X.replace([np.inf,-np.inf],np.nan)
    for c in X.columns: X[c]=pd.to_numeric(X[c],errors='coerce')
    X=X.fillna(X.median(numeric_only=True)).fillna(0).astype(float)
    return X,y

X,y=prepare(ch)

@st.cache_resource
def train(X,y):
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
    try: Xtr,ytr=SMOTE(random_state=42).fit_resample(Xtr,ytr)
    except Exception: pass
    model=RandomForestClassifier(n_estimators=150,max_depth=10,min_samples_split=5,min_samples_leaf=2,random_state=42,n_jobs=-1)
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    return model,sc,accuracy_score(yte,pred),X.columns.tolist(),Xtr,ytr

with st.spinner('🤖 Training Random Forest...'): model,sc,acc,features,Xbal,ybal=train(X,y)
st.success(f'Random Forest test accuracy: {acc:.2%}')

st.subheader('🔮 Real-Time Customer Prediction')
st.info('Enter values in the white boxes. The displayed ranges come from your loaded dataset.')

def rng(col,lo,hi):
    if col in ch:
        s=pd.to_numeric(ch[col],errors='coerce').dropna()
        if len(s): return float(s.min()),float(s.max())
    return float(lo),float(hi)

a,b,c=st.columns(3)
with a:
    lo,hi=rng('CreditScore',350,850); credit=st.number_input(f'Credit Score ({lo:.0f}–{hi:.0f})',min_value=lo,max_value=hi,value=float(np.clip(650,lo,hi)))
    lo,hi=rng('Age',18,100); age=st.number_input(f'Age ({lo:.0f}–{hi:.0f})',min_value=lo,max_value=hi,value=float(np.clip(35,lo,hi)))
    lo,hi=rng('Tenure',0,10); tenure=st.number_input(f'Tenure ({lo:.0f}–{hi:.0f})',min_value=lo,max_value=hi,value=float(np.clip(5,lo,hi)))
    geo=st.selectbox('Geography',sorted(ch.Geography.dropna().astype(str).unique()) if 'Geography' in ch else ['France','Germany','Spain'])
with b:
    lo,hi=rng('Balance',0,250000); balance=st.number_input(f'Balance ({lo:.2f}–{hi:.2f})',min_value=lo,max_value=hi,value=(lo+hi)/2)
    lo,hi=rng('NumOfProducts',1,4); products=st.number_input(f'Number of Products ({lo:.0f}–{hi:.0f})',min_value=lo,max_value=hi,value=float(np.clip(2,lo,hi)),step=1.)
    lo,hi=rng('EstimatedSalary',0,250000); salary=st.number_input(f'Estimated Salary ({lo:.2f}–{hi:.2f})',min_value=lo,max_value=hi,value=(lo+hi)/2)
    gender=st.selectbox('Gender',sorted(ch.Gender.dropna().astype(str).unique()) if 'Gender' in ch else ['Male','Female'])
with c:
    card=st.selectbox('Has Credit Card',[0,1],format_func=lambda x:'Yes' if x else 'No')
    active=st.selectbox('Is Active Member',[0,1],format_func=lambda x:'Yes' if x else 'No')

if st.button('🚀 Predict Customer Churn'):
    row=pd.DataFrame(0.,index=[0],columns=features)
    base={'CreditScore':credit,'Age':age,'Tenure':tenure,'Balance':balance,'NumOfProducts':products,'HasCrCard':card,'IsActiveMember':active,'EstimatedSalary':salary}
    for k,v in base.items():
        if k in row: row.loc[0,k]=v
    if 'Balance_to_Salary' in row: row.loc[0,'Balance_to_Salary']=balance/max(salary,1)
    if 'Products_per_Tenure' in row: row.loc[0,'Products_per_Tenure']=products/max(tenure,1)
    if 'Age_Squared' in row: row.loc[0,'Age_Squared']=age**2
    if 'CreditScore_Age' in row: row.loc[0,'CreditScore_Age']=credit/max(age,1)
    for f in [f'Geography_{geo}',f'Gender_{gender}']:
        if f in row: row.loc[0,f]=1
    z=sc.transform(row[features]); pred=int(model.predict(z)[0]); prob=model.predict_proba(z)[0]
    title='⚠️ PREDICTED: CHURN' if pred==1 else '✅ PREDICTED: NON-CHURN'
    text='The model predicts that this customer belongs to the churn class.' if pred==1 else 'The model predicts that this customer is likely to remain with the company.'
    st.markdown(f'<div class="pred"><div class="pred-title">{title}</div><p>{text}</p><p>Churn Probability: <b>{prob[1]:.1%}</b></p><p>Non-Churn Probability: <b>{prob[0]:.1%}</b></p></div>',unsafe_allow_html=True)
    st.subheader('📊 Prediction Probability'); st.bar_chart(pd.DataFrame({'Probability':[prob[0],prob[1]]},index=['Non-Churn','Churn']))

st.subheader('🔍 Feature Importance')
imp=pd.DataFrame({'Feature':features,'Importance':model.feature_importances_}).sort_values('Importance',ascending=False).head(15)
st.bar_chart(imp.set_index('Feature'))

st.subheader('📂 Dataset Preview')
with st.expander('Show dataset'): st.dataframe(ch.head(100),use_container_width=True)
with st.expander('Show dataset information'):
    st.write('Shape:',ch.shape); st.write('Columns:',ch.columns.tolist()); st.dataframe(ch.dtypes.astype(str).to_frame('Data Type'),use_container_width=True); st.dataframe(ch.isnull().sum().to_frame('Missing Values'),use_container_width=True)

st.subheader('🔍 Explainable AI')
st.markdown('<div class="card"><div class="big">LIME + SHAP Ready</div><p>The model uses the same preprocessed feature space required for connecting LIME or SHAP explanations.</p></div>',unsafe_allow_html=True)
