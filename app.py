import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# =========================
# PAGE CONFIG (FIRST LINE)
# =========================
st.set_page_config(page_title="Smart Ticket Routing", layout="wide")

# =========================
# TRAINING DATA (IMPROVED)
# =========================
def get_data():
    data = {
        "text": [
            # Technical
            "app crashes", "system error", "screen flickering",
            "software not working", "server failure",
            
            # Account
            "cannot login", "account locked", "forgot password",
            "otp not received", "reset password issue",
            
            # Billing
            "charged twice", "refund not received", "payment failed",
            "overcharged", "billing error",
            
            # Logistics
            "order not delivered", "delivery delayed",
            "tracking not working", "package lost", "shipment stuck"
        ],
        "label": [
            "Technical","Technical","Technical","Technical","Technical",
            "Account","Account","Account","Account","Account",
            "Billing","Billing","Billing","Billing","Billing",
            "Logistics","Logistics","Logistics","Logistics","Logistics"
        ]
    }
    return pd.DataFrame(data)

@st.cache_resource
def train_model():
    df = get_data()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipeline.fit(df["text"], df["label"])
    return pipeline

model = train_model()

# =========================
# URGENCY LOGIC
# =========================
def get_urgency(text):
    text = text.lower()

    high = ["charged", "refund", "failed", "error", "crash", "lost"]
    medium = ["login", "account", "password"]

    if any(x in text for x in high):
        return "High"
    elif any(x in text for x in medium):
        return "Medium"
    else:
        return "Low"

# =========================
# ROUTING LOGIC
# =========================
def route(category):
    return {
        "Technical": "Tech Team",
        "Account": "Account Support",
        "Billing": "Billing Team",
        "Logistics": "Delivery Team"
    }.get(category, "General Support")

# =========================
# AI RESPONSE
# =========================
def generate_reply(category):
    replies = {
        "Technical": "We are looking into the technical issue. Please try again shortly.",
        "Account": "Please verify your account or reset your password.",
        "Billing": "Our billing team will review your issue and update you.",
        "Logistics": "Your delivery issue is being checked. We will update you soon."
    }
    return replies.get(category, "Our support team will contact you.")

# =========================
# TITLE
# =========================
st.title("🚀 Smart Ticket Routing System")

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["🧍 Single Ticket", "📂 Bulk Upload"])

# =========================
# SINGLE TICKET
# =========================
with tab1:
    text = st.text_area("Enter Customer Issue")

    if st.button("Analyze Ticket"):
        if text.strip() == "":
            st.warning("Please enter text")
        else:
            pred = model.predict([text])[0]
            probs = model.predict_proba([text])[0]
            conf = max(probs)

            urgency = get_urgency(text)
            routed = route(pred)
            reply = generate_reply(pred)

            if conf < 0.5:
                st.warning(f"⚠️ Low Confidence: {conf*100:.2f}%")

            st.success("Analysis Complete")

            col1, col2, col3 = st.columns(3)
            col1.metric("Category", pred)
            col2.metric("Confidence", f"{conf*100:.2f}%")
            col3.metric("Urgency", urgency)

            st.info(f"📍 Routed To: {routed}")

            st.subheader("🤖 Suggested Response")
            st.text_area("Reply", reply)

            # CHART
            st.subheader("📊 Category Confidence")
            fig, ax = plt.subplots()
            ax.barh(model.classes_, probs)
            ax.set_xlabel("Probability")
            ax.set_ylabel("Category")
            st.pyplot(fig)

# =========================
# BULK UPLOAD (FIXED)
# =========================
with tab2:
    file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.dataframe(df.head())

        column = st.selectbox("Select text column", df.columns)

        if st.button("Process Bulk Data"):
            texts = df[column].astype(str)

            df["Category"] = model.predict(texts)
            probs = model.predict_proba(texts)
            df["Confidence"] = probs.max(axis=1)

            df["Urgency"] = texts.apply(get_urgency)
            df["Routed To"] = df["Category"].apply(route)

            st.success("Bulk Processing Done")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("Download Results", csv, "results.csv")

            # BULK CHART
            st.subheader("📊 Category Distribution")
            fig2, ax2 = plt.subplots()
            df["Category"].value_counts().plot(kind="bar", ax=ax2)
            ax2.set_xlabel("Category")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

# =========================
# MODEL INFO
# =========================
with st.expander("ℹ️ Model Info"):
    st.write("""
    - Uses NLP (TF-IDF)
    - Logistic Regression model
    - Supports Single & Bulk processing
    - Confidence-based prediction
    """)
