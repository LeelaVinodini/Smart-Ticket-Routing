import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Ticket Triage", layout="wide")

# =========================
# CUSTOM DARK STYLE
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.block-container {
    padding: 2rem;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1f2937, #111827);
    color: white;
    margin-bottom: 10px;
}
.big-text {
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TRAINING DATA
# =========================
def get_data():
    data = {
        "text": [
            "app crashes", "screen flickering", "software error",
            "cannot login", "account locked", "password reset issue",
            "charged twice", "refund not received", "payment failed",
            "order not delivered", "tracking issue", "delivery delayed"
        ],
        "label": [
            "Technical","Technical","Technical",
            "Account","Account","Account",
            "Billing","Billing","Billing",
            "Logistics","Logistics","Logistics"
        ]
    }
    return pd.DataFrame(data)

@st.cache_resource
def train():
    df = get_data()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(df["text"], df["label"])
    return pipe

model = train()

# =========================
# URGENCY LOGIC
# =========================
def get_urgency(text):
    text = text.lower()
    if any(x in text for x in ["charged", "refund", "crash", "error", "failed"]):
        return "High"
    if any(x in text for x in ["login", "account"]):
        return "Medium"
    return "Low"

# =========================
# ROUTING LOGIC
# =========================
def route(category):
    mapping = {
        "Technical": "Tech Team",
        "Account": "Account Support",
        "Billing": "Billing Team",
        "Logistics": "Delivery Team"
    }
    return mapping.get(category, "Auto-Reply / Bot")

# =========================
# AI RESPONSE
# =========================
def generate_reply(category):
    responses = {
        "Technical": "We are checking the technical issue. Please try again shortly.",
        "Account": "Please reset your password or verify your account details.",
        "Billing": "Our billing team will review your transaction and update you.",
        "Logistics": "Your delivery is being tracked. We will update you soon."
    }
    return responses.get(category, "Please contact support.")

# =========================
# UI
# =========================
st.title("🧾 New Ticket Triage")

col_main, col_side = st.columns([3,1])

with col_main:
    text = st.text_area("Enter Customer Ticket Text", "i have issue with login")

    if st.button("Analyze Ticket"):
        pred = model.predict([text])[0]
        probs = model.predict_proba([text])[0]
        conf = max(probs)

        urgency = get_urgency(text)
        routed = route(pred)
        reply = generate_reply(pred)

        # Warning
        if conf < 0.5:
            st.warning(f"⚠️ Low Confidence ({conf*100:.2f}%)")

        st.success("Analysis Complete!")

        # CARDS
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f'<div class="card"><div>Category</div><div class="big-text">{pred}</div></div>', unsafe_allow_html=True)

        with c2:
            st.markdown(f'<div class="card"><div>Urgency</div><div class="big-text">{urgency}</div></div>', unsafe_allow_html=True)

        with c3:
            st.markdown(f'<div class="card"><div>Routed To</div><div class="big-text">{routed}</div></div>', unsafe_allow_html=True)

        # AI RESPONSE
        st.subheader("🤖 Suggested AI Response")
        st.text_area("Draft Reply", reply)

        # =========================
        # CHARTS
        # =========================
        st.subheader("📊 Model Confidence Breakdown")

        colA, colB = st.columns(2)

        # Category Chart
        with colA:
            labels = model.classes_
            plt.figure()
            plt.barh(labels, probs)
            plt.xlabel("Probability")
            st.pyplot(plt)

        # Urgency Chart
        with colB:
            urg_levels = ["Low", "Medium", "High"]
            urg_scores = [0.2, 0.6, 0.2] if urgency == "Medium" else [0.1,0.2,0.7]
            plt.figure()
            plt.barh(urg_levels, urg_scores)
            st.pyplot(plt)

# SIDE PANEL
with col_side:
    st.info("""
### ℹ️ How it works
1. Input ticket  
2. NLP analyzes text  
3. Categorizes issue  
4. Routes automatically  
5. Generates reply  
""")
