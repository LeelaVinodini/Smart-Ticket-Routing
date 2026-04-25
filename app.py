import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ✅ FIRST COMMAND
st.set_page_config(page_title="AI Ticket Routing", page_icon="🚀", layout="wide")

# =====================================
# 🔥 BIGGER TRAINING DATA (KEY UPGRADE)
# =====================================
def get_data():
    data = {
        "text": [
            # Technical (more variations)
            "app crashes on startup", "screen flickering issue",
            "software not responding", "error while logging in",
            "system not loading", "app freezes frequently",
            "bug in application", "server error occurred",
            "app not opening", "system failure issue",

            # Account
            "cannot login to account", "password reset not working",
            "account locked", "unable to verify email",
            "forgot password issue", "cannot update profile",
            "login failed error", "account suspended",
            "otp not received", "cannot access account",

            # Billing
            "charged twice for subscription", "refund not received",
            "incorrect billing amount", "payment failed but money deducted",
            "overcharged on my card", "unexpected transaction detected",
            "double payment issue", "refund pending",
            "billing error on invoice", "extra charges applied",

            # Logistics
            "order not delivered", "where is my package",
            "delivery delayed", "tracking not working",
            "shipment stuck in transit", "wrong delivery address",
            "package lost", "delivery not attempted",
            "courier delay issue", "order arrived late"
        ],
        "label": [
            "Technical","Technical","Technical","Technical","Technical",
            "Technical","Technical","Technical","Technical","Technical",

            "Account","Account","Account","Account","Account",
            "Account","Account","Account","Account","Account",

            "Billing","Billing","Billing","Billing","Billing",
            "Billing","Billing","Billing","Billing","Billing",

            "Logistics","Logistics","Logistics","Logistics","Logistics",
            "Logistics","Logistics","Logistics","Logistics","Logistics"
        ]
    }
    return pd.DataFrame(data)

# =====================================
# 🔥 IMPROVED MODEL
# =====================================
@st.cache_resource
def train_model():
    df = get_data()

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1,2),
            stop_words='english',
            max_features=1000
        )),
        ('clf', LogisticRegression(max_iter=300))
    ])

    pipeline.fit(df['text'], df['label'])
    return pipeline

model_pipeline = train_model()

# =====================================
# 🔥 SMART URGENCY SYSTEM
# =====================================
def get_urgency(text, category):
    text = text.lower()

    high_keywords = [
        "charged", "overcharged", "double", "refund",
        "payment failed", "money deducted",
        "not received", "lost", "crash", "error", "locked"
    ]

    if any(word in text for word in high_keywords):
        return "High"

    if category in ["Billing", "Account"]:
        return "Medium"

    return "Low"

# =====================================
# UI
# =====================================
st.title("🚀 AI Smart Ticket Routing System")
st.write("High Accuracy NLP-based Classification")

tab1, tab2 = st.tabs(["🧍 Single Ticket", "📂 Bulk Upload"])

# =====================
# SINGLE INPUT
# =====================
with tab1:
    text = st.text_area("Enter Support Ticket")

    if st.button("Classify"):
        if text.strip() == "":
            st.warning("Enter some text")
        else:
            pred = model_pipeline.predict([text])[0]
            probs = model_pipeline.predict_proba([text])[0]
            confidence = max(probs)

            urgency = get_urgency(text, pred)

            col1, col2, col3 = st.columns(3)
            col1.metric("Category", pred)
            col2.metric("Confidence", f"{confidence*100:.2f}%")
            col3.metric("Urgency", urgency)

# =====================
# BULK INPUT
# =====================
with tab2:
    file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            st.dataframe(df.head())

            col = st.selectbox("Select column", df.columns)

            if st.button("Process"):
                texts = df[col].astype(str)

                df["Category"] = model_pipeline.predict(texts)
                probs = model_pipeline.predict_proba(texts)
                df["Confidence"] = probs.max(axis=1)

                df["Urgency"] = df.apply(
                    lambda x: get_urgency(x[col], x["Category"]),
                    axis=1
                )

                st.success("Done ✅")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("Download Results", csv, "results.csv")

        except Exception as e:
            st.error(f"Error: {e}")

# =====================
# INFO
# =====================
with st.expander("Model Info"):
    st.write("""
    ✔ Uses NLP (TF-IDF with bi-grams)  
    ✔ Logistic Regression (optimized)  
    ✔ Improved dataset → better accuracy  
    ✔ Smart urgency detection  
    """)
