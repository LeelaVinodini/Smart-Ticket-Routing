import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ✅ MUST BE FIRST
st.set_page_config(page_title="Smart Ticket Routing", page_icon="🚀", layout="wide")

# ==============================
# STEP 1: IMPROVED TRAINING DATA
# ==============================
def get_data():
    data = {
        "text": [
            # Technical
            "app crashes on startup", "screen flickering issue",
            "software not responding", "error while logging in",
            "system not loading", "app freezes frequently",

            # Account
            "cannot login to account", "password reset not working",
            "account locked", "unable to verify email",
            "forgot password issue", "cannot update profile",

            # Billing
            "charged twice for subscription", "refund not received",
            "incorrect billing amount", "payment failed but money deducted",
            "overcharged on my card", "unexpected transaction detected",

            # Logistics
            "order not delivered", "where is my package",
            "delivery delayed", "tracking not working",
            "shipment stuck in transit", "wrong delivery address"
        ],
        "label": [
            "Technical","Technical","Technical","Technical","Technical","Technical",
            "Account","Account","Account","Account","Account","Account",
            "Billing","Billing","Billing","Billing","Billing","Billing",
            "Logistics","Logistics","Logistics","Logistics","Logistics","Logistics"
        ]
    }
    return pd.DataFrame(data)

# ==============================
# STEP 2: TRAIN MODEL (IMPROVED)
# ==============================
@st.cache_resource
def train_model():
    df = get_data()

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=200))
    ])

    pipeline.fit(df['text'], df['label'])
    return pipeline

model_pipeline = train_model()

# ==============================
# STEP 3: SMART URGENCY LOGIC
# ==============================
def get_urgency(text, category):
    text = text.lower()

    high_keywords = [
        "charged", "refund", "overcharged", "double charged",
        "payment failed", "money deducted",
        "not received", "crash", "error", "not working", "locked"
    ]

    if any(word in text for word in high_keywords):
        return "High"

    if category in ["Billing", "Account"]:
        return "Medium"

    return "Low"

# ==============================
# UI
# ==============================
st.title("🚀 Smart Ticket Routing System")
st.write("Supports **Single + Bulk Classification with Improved Accuracy**")

tab1, tab2 = st.tabs(["🧍 Single Ticket", "📂 Bulk Upload"])

# ==============================
# SINGLE INPUT
# ==============================
with tab1:
    ticket_input = st.text_area("Enter Support Ticket")

    if st.button("Classify Ticket"):
        if ticket_input.strip() == "":
            st.warning("⚠️ Please enter text")
        else:
            prediction = model_pipeline.predict([ticket_input])[0]
            probs = model_pipeline.predict_proba([ticket_input])
            confidence = max(probs[0])

            urgency = get_urgency(ticket_input, prediction)

            col1, col2, col3 = st.columns(3)
            col1.metric("Category", prediction)
            col2.metric("Confidence", f"{round(confidence*100,2)}%")
            col3.metric("Urgency", urgency)

# ==============================
# BULK INPUT
# ==============================
with tab2:
    st.subheader("Upload CSV / Excel File")

    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("✅ File uploaded")
            st.dataframe(df.head())

            text_column = st.selectbox("Select column to classify", df.columns)

            if st.button("Process Bulk Data"):
                texts = df[text_column].astype(str)

                df["Category"] = model_pipeline.predict(texts)
                probs = model_pipeline.predict_proba(texts)
                df["Confidence"] = probs.max(axis=1)

                df["Urgency"] = df.apply(
                    lambda x: get_urgency(x[text_column], x["Category"]),
                    axis=1
                )

                st.success("✅ Processing Complete")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "classified_results.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ==============================
# MODEL INFO
# ==============================
with st.expander("⚙️ Model Details"):
    st.table({
        "Feature": ["Vectorization", "Model", "Accuracy Boost"],
        "Details": [
            "TF-IDF (Bi-grams)",
            "Logistic Regression (max_iter=200)",
            "Improved dataset + smarter urgency logic"
        ]
    })