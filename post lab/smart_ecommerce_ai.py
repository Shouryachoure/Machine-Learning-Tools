# ============================================================
#  Smart E-commerce AI System
#  Built with Python + Streamlit + Scikit-learn
#  Features: Product Recommendation | Sentiment Analysis | Customer Segmentation
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart E-commerce AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — clean dark-card UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

    .stApp { background: #0d0f14; color: #e8eaf0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #13161e !important;
        border-right: 1px solid #1e2230;
    }

    /* Cards */
    .ai-card {
        background: #13161e;
        border: 1px solid #1e2230;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .ai-card h3 { margin: 0 0 0.4rem; font-size: 1.05rem; color: #a78bfa; }
    .ai-card p  { margin: 0; font-size: 0.88rem; color: #8b90a0; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a0533 0%, #0d1a3a 50%, #001a2e 100%);
        border: 1px solid #2a1a4e;
        border-radius: 18px;
        padding: 2rem 2.4rem;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0;
               background: linear-gradient(90deg,#c084fc,#60a5fa,#34d399);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p  { margin: 0.5rem 0 0; color: #8b90a0; font-size: 0.95rem; }

    /* Result badges */
    .badge-pos { background:#052e16; color:#4ade80; border:1px solid #14532d;
                 padding:0.25rem 0.75rem; border-radius:999px; font-size:0.82rem; font-weight:600; }
    .badge-neg { background:#2d0a0a; color:#f87171; border:1px solid #7f1d1d;
                 padding:0.25rem 0.75rem; border-radius:999px; font-size:0.82rem; font-weight:600; }
    .badge-neu { background:#1c1a05; color:#fbbf24; border:1px solid #78350f;
                 padding:0.25rem 0.75rem; border-radius:999px; font-size:0.82rem; font-weight:600; }

    /* Product recommendation cards */
    .rec-card {
        background: #1a1d27;
        border: 1px solid #252a38;
        border-left: 3px solid #a78bfa;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .rec-card .title  { font-weight: 600; font-size: 0.93rem; color: #e2e8f0; }
    .rec-card .meta   { font-size: 0.8rem; color: #6b7280; margin-top: 0.15rem; }
    .rec-card .score  { font-size: 0.78rem; color: #a78bfa; float: right; font-weight: 600; }

    /* Section headers */
    .section-header {
        display: flex; align-items: center; gap: 0.6rem;
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem; font-weight: 700;
        color: #e2e8f0; margin-bottom: 1.2rem;
    }
    .section-header .icon { font-size: 1.4rem; }

    /* Metric tile */
    .metric-tile {
        background: #13161e; border: 1px solid #1e2230;
        border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
    }
    .metric-tile .val { font-size: 1.7rem; font-weight: 800;
                        color: #a78bfa; font-family: 'Syne', sans-serif; }
    .metric-tile .lbl { font-size: 0.78rem; color: #6b7280; margin-top: 0.2rem; }

    /* Override Streamlit defaults */
    .stSelectbox label, .stTextArea label { color: #9ca3af !important; font-size: 0.85rem !important; }
    div[data-baseweb="select"] > div { background:#1a1d27 !important; border-color:#252a38 !important; }
    .stButton > button {
        background: linear-gradient(135deg,#7c3aed,#3b82f6);
        color: #fff; border: none; border-radius: 8px;
        font-family: 'Syne', sans-serif; font-weight: 600;
        padding: 0.55rem 1.4rem; font-size: 0.9rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
    textarea { background: #1a1d27 !important; color: #e2e8f0 !important;
               border-color: #252a38 !important; }
    .stSlider .st-bw { color: #a78bfa; }
    hr { border-color: #1e2230; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  1. SAMPLE DATASET
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def load_products():
    """Create a sample product catalogue."""
    data = {
        "product_id": range(1, 21),
        "product_name": [
            "Wireless Bluetooth Headphones", "Gaming Mechanical Keyboard",
            "4K Ultra HD Monitor", "Noise-Cancelling Earbuds",
            "Ergonomic Office Chair", "Standing Desk",
            "USB-C Hub 7-in-1", "Webcam 1080p HD",
            "LED Ring Light", "Laptop Cooling Pad",
            "Smart Watch Fitness Tracker", "Portable Charger 20000mAh",
            "Phone Gimbal Stabilizer", "Wireless Charging Pad",
            "RGB Desk Lamp", "Mechanical Gaming Mouse",
            "Noise-Cancelling Headset", "Portable SSD 1TB",
            "Smart Home Speaker", "Video Capture Card"
        ],
        "category": [
            "Audio", "Input Devices", "Display", "Audio",
            "Furniture", "Furniture", "Accessories", "Accessories",
            "Lighting", "Accessories", "Wearables", "Power",
            "Photography", "Power", "Lighting", "Input Devices",
            "Audio", "Storage", "Smart Home", "Streaming"
        ],
        "price": [
            79.99, 129.99, 349.99, 59.99, 299.99, 499.99,
            49.99, 89.99, 39.99, 29.99, 149.99, 44.99,
            119.99, 24.99, 34.99, 69.99, 99.99, 109.99,
            79.99, 189.99
        ],
        "rating": [
            4.5, 4.7, 4.6, 4.3, 4.8, 4.4, 4.2, 4.5,
            4.1, 4.0, 4.6, 4.4, 4.3, 4.2, 4.0, 4.7,
            4.5, 4.8, 4.3, 4.5
        ],
        "description": [
            "Over-ear headphones with deep bass wireless 30hr battery Bluetooth 5.0 premium sound quality music",
            "Tactile mechanical switches RGB backlight gaming keyboard programmable keys fast response",
            "4K resolution IPS panel 144Hz gaming monitor wide color gamut HDR display",
            "Active noise cancellation earbuds TWS wireless compact portable music commute",
            "Lumbar support adjustable armrest ergonomic office chair comfortable sitting posture",
            "Electric height adjustable standing desk motorized sit stand workspace productivity",
            "Multi-port USB-C hub HDMI 4K USB 3.0 SD card reader power delivery hub adapter",
            "1080p full HD webcam autofocus streaming video call meeting home office",
            "18-inch LED ring light adjustable brightness color temperature photography selfie studio",
            "Laptop cooling pad dual fans USB powered quiet slim portable gaming",
            "Heart rate monitor step counter GPS fitness tracker sleep smart watch",
            "20000mAh fast charge power bank USB-C portable battery multiple devices travel",
            "3-axis gimbal stabilizer smartphone video recording smooth cinematic shots",
            "10W fast wireless charging pad Qi compatible iPhone Android slim",
            "RGB LED desk lamp touch control USB charging port dimmable office study",
            "Programmable gaming mouse 25600 DPI adjustable RGB lightweight FPS",
            "Surround sound headset microphone noise-cancelling gaming office communication",
            "1TB portable SSD USB 3.2 Gen2 fast transfer compact rugged reliable",
            "360 degree smart speaker voice assistant Bluetooth WiFi home automation",
            "4K HDMI video capture card live streaming recording gaming USB"
        ]
    }
    return pd.DataFrame(data)


@st.cache_data
def load_reviews():
    """Sample product reviews for sentiment analysis."""
    return [
        ("Absolutely love these headphones! Crystal clear sound and super comfortable.", "positive"),
        ("Terrible product, broke after two weeks. Waste of money.", "negative"),
        ("Decent quality for the price. Shipping was a bit slow.", "neutral"),
        ("Best keyboard I have ever used! The tactile feedback is amazing.", "positive"),
        ("Very disappointed. Product description was misleading.", "negative"),
        ("Works as expected. Nothing spectacular but gets the job done.", "neutral"),
        ("Outstanding build quality and the battery life is incredible!", "positive"),
        ("Stopped working after first use. Customer service unhelpful.", "negative"),
        ("Good value for money. Would recommend to a friend.", "positive"),
        ("Screen has dead pixels and colours look washed out. Not happy.", "negative"),
    ]


@st.cache_data
def load_customers():
    """Sample customer behaviour data for clustering."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id":      range(1, n + 1),
        "total_spent":      np.round(np.random.exponential(scale=250, size=n), 2),
        "num_orders":       np.random.randint(1, 50, size=n),
        "avg_order_value":  np.round(np.random.uniform(20, 500, size=n), 2),
        "days_since_last":  np.random.randint(1, 365, size=n),
        "num_reviews":      np.random.randint(0, 20, size=n),
    })


# ═══════════════════════════════════════════════════════════════
#  2. DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def build_recommendation_model(products_df):
    """
    Content-Based Filtering using TF-IDF + Cosine Similarity.
    Each product is represented by its description as a TF-IDF vector.
    Similarity is the cosine angle between two such vectors.
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(products_df["description"])
    sim_matrix   = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return sim_matrix


def simple_sentiment(text: str) -> tuple[str, float]:
    """
    Rule-based sentiment classifier (no external model needed).
    Returns (label, confidence_score).
    For production use, replace with a trained ML model or
    a transformer like DistilBERT / VADER.
    """
    positive_words = {
        "love", "amazing", "excellent", "great", "awesome", "fantastic",
        "wonderful", "best", "outstanding", "perfect", "good", "nice",
        "happy", "incredible", "superb", "brilliant", "recommend",
        "satisfied", "quality", "comfortable", "fast", "clear", "reliable"
    }
    negative_words = {
        "terrible", "awful", "horrible", "bad", "worst", "poor",
        "disappointed", "broken", "defective", "useless", "waste",
        "unhappy", "return", "refund", "misleading", "dead", "stopped",
        "broke", "cheap", "slow", "delayed", "damage", "fail"
    }

    tokens     = text.lower().split()
    pos_count  = sum(1 for t in tokens if t.strip(".,!?") in positive_words)
    neg_count  = sum(1 for t in tokens if t.strip(".,!?") in negative_words)
    total      = pos_count + neg_count

    if total == 0:
        return "neutral", 0.55

    pos_ratio = pos_count / total
    if pos_ratio >= 0.6:
        return "positive", round(0.5 + pos_ratio * 0.45, 2)
    elif pos_ratio <= 0.4:
        return "negative", round(0.5 + (1 - pos_ratio) * 0.45, 2)
    else:
        return "neutral", 0.55


@st.cache_resource
def build_segmentation_model(customers_df):
    """
    K-Means clustering on customer behaviour features.
    StandardScaler normalises features before clustering.
    """
    features     = ["total_spent", "num_orders", "avg_order_value", "days_since_last"]
    X            = customers_df[features]
    scaler       = StandardScaler()
    X_scaled     = scaler.fit_transform(X)
    kmeans       = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels       = kmeans.fit_predict(X_scaled)
    return kmeans, scaler, labels, features


# ═══════════════════════════════════════════════════════════════
#  3. SIDEBAR — NAVIGATION
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.15rem;font-weight:800;
                    background:linear-gradient(90deg,#c084fc,#60a5fa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            🛍️ E-commerce AI
        </div>
        <div style='color:#4b5563;font-size:0.78rem;margin-top:0.2rem;'>
            Smart Retail Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate to",
        ["🏠 Dashboard", "🎯 Product Recommender", "💬 Sentiment Analyser", "👥 Customer Segmentation"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem;color:#4b5563;line-height:1.7;'>
        <b style='color:#6b7280;'>Tech Stack</b><br>
        Python · Streamlit<br>
        Scikit-learn · Pandas<br>
        TF-IDF · K-Means<br>
        Cosine Similarity
    </div>
    """, unsafe_allow_html=True)


# ─── Load data once ───
products_df  = load_products()
reviews_data = load_reviews()
customers_df = load_customers()
sim_matrix   = build_recommendation_model(products_df)
kmeans, scaler, cluster_labels, cluster_features = build_segmentation_model(customers_df)
customers_df["segment"] = cluster_labels


# ═══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown("""
    <div class='hero'>
        <h1>Smart E-commerce AI System</h1>
        <p>Three ML-powered modules — recommendations, sentiment analysis, and customer segmentation — in one interface.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-tile'><div class='val'>{len(products_df)}</div>
        <div class='lbl'>Products</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-tile'><div class='val'>{len(reviews_data)}</div>
        <div class='lbl'>Sample Reviews</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-tile'><div class='val'>{len(customers_df)}</div>
        <div class='lbl'>Customers</div></div>""", unsafe_allow_html=True)
    with c4:
        avg_r = products_df["rating"].mean()
        st.markdown(f"""<div class='metric-tile'><div class='val'>⭐ {avg_r:.1f}</div>
        <div class='lbl'>Avg Rating</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""<div class='ai-card'>
        <h3>🎯 Product Recommender</h3>
        <p>Content-based filtering using TF-IDF vectorisation and cosine similarity to find products similar to what a customer is viewing.</p>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class='ai-card'>
        <h3>💬 Sentiment Analyser</h3>
        <p>Classifies product reviews as Positive, Negative, or Neutral using rule-based NLP — easily swappable with a trained ML model.</p>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""<div class='ai-card'>
        <h3>👥 Customer Segmentation</h3>
        <p>Groups customers into behavioural clusters with K-Means, enabling targeted marketing and personalised promotions.</p>
        </div>""", unsafe_allow_html=True)

    # Category distribution chart
    st.markdown("---")
    st.markdown("<div class='section-header'><span class='icon'>📊</span> Product Category Distribution</div>",
                unsafe_allow_html=True)

    cat_counts = products_df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("#13161e")
    ax.set_facecolor("#13161e")
    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(cat_counts)))
    bars = ax.barh(cat_counts.index, cat_counts.values, color=colors, height=0.6)
    for bar, val in zip(bars, cat_counts.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color="#9ca3af", fontsize=9)
    ax.spines[:].set_visible(False)
    ax.tick_params(colors="#9ca3af", labelsize=9)
    ax.xaxis.set_visible(False)
    ax.set_title("Number of products per category", color="#6b7280", fontsize=9, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  PAGE: PRODUCT RECOMMENDER
# ═══════════════════════════════════════════════════════════════

elif page == "🎯 Product Recommender":
    st.markdown("<div class='section-header'><span class='icon'>🎯</span> Content-Based Product Recommender</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='ai-card'>
    <h3>How it works</h3>
    <p>Each product description is converted to a TF-IDF vector. 
    Cosine similarity between vectors finds the most related products — 
    similar to how Netflix or Amazon recommend "customers also viewed".</p>
    </div>""", unsafe_allow_html=True)

    product_names = products_df["product_name"].tolist()
    selected = st.selectbox("Select a product to find similar items", product_names)
    top_n = st.slider("Number of recommendations", min_value=2, max_value=8, value=4)

    if st.button("🔍 Get Recommendations"):
        # ── Model: find top-N most similar products ──
        idx      = products_df[products_df["product_name"] == selected].index[0]
        scores   = list(enumerate(sim_matrix[idx]))
        scores   = sorted(scores, key=lambda x: x[1], reverse=True)
        scores   = [(i, s) for i, s in scores if i != idx][:top_n]

        st.markdown(f"<br>**Showing {top_n} recommendations for:** _{selected}_<br>", unsafe_allow_html=True)

        for rank, (i, score) in enumerate(scores, 1):
            row = products_df.iloc[i]
            stars = "⭐" * int(round(row["rating"]))
            pct   = int(score * 100)
            st.markdown(f"""
            <div class='rec-card'>
                <span class='score'>{pct}% match</span>
                <div class='title'>{rank}. {row['product_name']}</div>
                <div class='meta'>
                    {row['category']} &nbsp;·&nbsp; ₹{row['price']:.2f} 
                    &nbsp;·&nbsp; {stars} {row['rating']}
                </div>
            </div>""", unsafe_allow_html=True)

        # Mini bar chart of similarity scores
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 2.8))
        fig.patch.set_facecolor("#13161e")
        ax.set_facecolor("#13161e")
        names  = [products_df.iloc[i]["product_name"][:28] + "…"
                  if len(products_df.iloc[i]["product_name"]) > 28
                  else products_df.iloc[i]["product_name"]
                  for i, _ in scores]
        vals   = [s for _, s in scores]
        gradient = plt.cm.plasma(np.linspace(0.3, 0.85, len(vals)))
        ax.barh(names[::-1], vals[::-1], color=gradient[::-1], height=0.5)
        ax.set_xlim(0, 1)
        ax.spines[:].set_visible(False)
        ax.tick_params(colors="#9ca3af", labelsize=8)
        ax.set_xlabel("Cosine Similarity Score", color="#6b7280", fontsize=8)
        ax.set_title("Similarity Scores", color="#9ca3af", fontsize=9, pad=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════
#  PAGE: SENTIMENT ANALYSER
# ═══════════════════════════════════════════════════════════════

elif page == "💬 Sentiment Analyser":
    st.markdown("<div class='section-header'><span class='icon'>💬</span> Product Review Sentiment Analyser</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='ai-card'>
    <h3>How it works</h3>
    <p>Review text is tokenised and scored against curated positive/negative lexicons. 
    The ratio of positive-to-negative tokens determines the sentiment label and confidence. 
    In production, replace this with a trained Logistic Regression or DistilBERT model.</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["✏️ Analyse Custom Review", "📋 Sample Reviews"])

    # ── Tab 1: custom input ──
    with tab1:
        review_text = st.text_area(
            "Enter a product review",
            placeholder="e.g. This product is absolutely amazing! Battery life is incredible and sound quality is perfect.",
            height=120
        )
        if st.button("🔍 Analyse Sentiment"):
            if review_text.strip():
                label, confidence = simple_sentiment(review_text)

                badge_map = {
                    "positive": "<span class='badge-pos'>✅ POSITIVE</span>",
                    "negative": "<span class='badge-neg'>❌ NEGATIVE</span>",
                    "neutral":  "<span class='badge-neu'>⚖️ NEUTRAL</span>",
                }
                st.markdown(f"""
                <div class='ai-card' style='margin-top:1rem;'>
                    <div style='display:flex;align-items:center;gap:1rem;'>
                        <div>
                            <div style='font-size:0.8rem;color:#6b7280;margin-bottom:0.4rem;'>Predicted Sentiment</div>
                            {badge_map[label]}
                        </div>
                        <div style='margin-left:2rem;'>
                            <div style='font-size:0.8rem;color:#6b7280;margin-bottom:0.2rem;'>Confidence</div>
                            <div style='font-size:1.4rem;font-weight:700;color:#a78bfa;font-family:Syne,sans-serif;'>{int(confidence*100)}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                fig, ax = plt.subplots(figsize=(5, 0.7))
                fig.patch.set_facecolor("#13161e")
                ax.set_facecolor("#13161e")
                color = "#4ade80" if label == "positive" else ("#f87171" if label == "negative" else "#fbbf24")
                ax.barh(["confidence"], [confidence], color=color, height=0.4)
                ax.barh(["confidence"], [1], color="#1e2230", height=0.4, zorder=0)
                ax.set_xlim(0, 1); ax.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Please enter a review first.")

    # ── Tab 2: sample reviews ──
    with tab2:
        results = [(text, actual, *simple_sentiment(text)) for text, actual in reviews_data]
        badge_map = {
            "positive": "<span class='badge-pos'>✅ POSITIVE</span>",
            "negative": "<span class='badge-neg'>❌ NEGATIVE</span>",
            "neutral":  "<span class='badge-neu'>⚖️ NEUTRAL</span>",
        }
        for text, actual, pred, conf in results:
            st.markdown(f"""
            <div class='ai-card' style='padding:0.85rem 1.1rem;'>
                <div style='font-size:0.88rem;color:#d1d5db;margin-bottom:0.5rem;'>"{text}"</div>
                <div style='display:flex;gap:0.8rem;align-items:center;'>
                    <div style='font-size:0.75rem;color:#6b7280;'>Predicted:</div>
                    {badge_map.get(pred, pred)}
                    <div style='font-size:0.75rem;color:#6b7280;margin-left:0.5rem;'>Confidence: <b style='color:#a78bfa;'>{int(conf*100)}%</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Sentiment distribution pie
        st.markdown("---")
        labels_list = [pred for _, _, pred, _ in results]
        counts = {k: labels_list.count(k) for k in ["positive", "negative", "neutral"]}
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#13161e")
        ax.set_facecolor("#13161e")
        wedge_colors = ["#4ade80", "#f87171", "#fbbf24"]
        non_zero = {k: v for k, v in counts.items() if v > 0}
        ax.pie(non_zero.values(), labels=non_zero.keys(), colors=wedge_colors[:len(non_zero)],
               autopct="%1.0f%%", textprops={"color": "#9ca3af", "fontsize": 9},
               wedgeprops={"linewidth": 2, "edgecolor": "#13161e"})
        ax.set_title("Sentiment Distribution (Sample)", color="#9ca3af", fontsize=9, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════
#  PAGE: CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════

elif page == "👥 Customer Segmentation":
    st.markdown("<div class='section-header'><span class='icon'>👥</span> K-Means Customer Segmentation</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='ai-card'>
    <h3>How it works</h3>
    <p>Four customer behaviour features (total spend, order count, average order value, recency) 
    are normalised with StandardScaler, then grouped into 4 clusters by K-Means. 
    Each cluster represents a distinct shopper persona.</p>
    </div>""", unsafe_allow_html=True)

    # Cluster naming heuristic
    cluster_stats = customers_df.groupby("segment")[cluster_features].mean()
    segment_names = {}
    for seg in range(4):
        row   = cluster_stats.loc[seg]
        spent = row["total_spent"]
        orders = row["num_orders"]
        days   = row["days_since_last"]
        if spent > 300 and orders > 25:
            segment_names[seg] = ("👑 VIP Champions", "#a78bfa")
        elif spent > 200 and days < 100:
            segment_names[seg] = ("🔥 Loyal Regulars", "#34d399")
        elif days > 250:
            segment_names[seg] = ("😴 At-Risk Churners", "#f87171")
        else:
            segment_names[seg] = ("🌱 New / Occasional", "#60a5fa")

    # Summary cards
    cols = st.columns(4)
    for idx, seg in enumerate(sorted(customers_df["segment"].unique())):
        cnt  = (customers_df["segment"] == seg).sum()
        name, colour = segment_names.get(seg, (f"Segment {seg}", "#9ca3af"))
        avg_spent = customers_df[customers_df["segment"] == seg]["total_spent"].mean()
        with cols[idx]:
            st.markdown(f"""
            <div class='metric-tile' style='border-top:3px solid {colour};'>
                <div class='val' style='color:{colour};font-size:1.1rem;'>{name}</div>
                <div class='lbl' style='margin-top:0.5rem;'>👤 {cnt} customers</div>
                <div class='lbl'>💰 Avg spend: ₹{avg_spent:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Scatter plot
    tab_scatter, tab_table = st.tabs(["📈 Cluster Visualisation", "📋 Customer Table"])

    with tab_scatter:
        x_axis = st.selectbox("X-axis feature", cluster_features, index=0)
        y_axis = st.selectbox("Y-axis feature", cluster_features, index=1)

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#13161e")
        ax.set_facecolor("#1a1d27")
        palette = ["#a78bfa", "#34d399", "#f87171", "#60a5fa"]

        for seg in sorted(customers_df["segment"].unique()):
            mask  = customers_df["segment"] == seg
            name, _ = segment_names.get(seg, (f"Seg {seg}", "#fff"))
            ax.scatter(
                customers_df.loc[mask, x_axis],
                customers_df.loc[mask, y_axis],
                c=palette[seg % 4], s=40, alpha=0.7, label=name
            )

        # Cluster centres (inverse transform)
        centres = scaler.inverse_transform(kmeans.cluster_centers_)
        feat_idx = {f: i for i, f in enumerate(cluster_features)}
        for seg, c in enumerate(centres):
            ax.scatter(c[feat_idx[x_axis]], c[feat_idx[y_axis]],
                       marker="X", s=200, c=palette[seg % 4], edgecolors="#fff", zorder=5)

        ax.set_xlabel(x_axis.replace("_", " ").title(), color="#9ca3af", fontsize=10)
        ax.set_ylabel(y_axis.replace("_", " ").title(), color="#9ca3af", fontsize=10)
        ax.tick_params(colors="#6b7280")
        ax.spines[:].set_color("#252a38")
        ax.set_title("Customer Segments (✕ = cluster centroid)", color="#9ca3af", fontsize=10, pad=10)
        legend = ax.legend(fontsize=8, facecolor="#13161e", edgecolor="#252a38",
                           labelcolor="#9ca3af", loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab_table:
        display_df = customers_df.copy()
        display_df["Segment Name"] = display_df["segment"].map(
            lambda s: segment_names.get(s, (f"Seg {s}",))[0]
        )
        st.dataframe(
            display_df[["customer_id", "total_spent", "num_orders",
                         "avg_order_value", "days_since_last", "Segment Name"]].head(50),
            use_container_width=True,
            hide_index=True
        )

    # ── Predict segment for a new customer ──
    st.markdown("---")
    st.markdown("#### 🔮 Predict Segment for a New Customer")
    col1, col2, col3, col4 = st.columns(4)
    with col1:  new_spent  = st.number_input("Total Spent (₹)", 0.0, 5000.0, 150.0, step=10.0)
    with col2:  new_orders = st.number_input("Number of Orders",  1, 100, 5)
    with col3:  new_aov    = st.number_input("Avg Order Value (₹)", 0.0, 1000.0, 100.0, step=10.0)
    with col4:  new_days   = st.number_input("Days Since Last Order", 1, 365, 60)

    if st.button("🎯 Predict Customer Segment"):
        # ── Prediction step ──
        new_data    = np.array([[new_spent, new_orders, new_aov, new_days]])
        new_scaled  = scaler.transform(new_data)
        pred_seg    = kmeans.predict(new_scaled)[0]
        seg_name, seg_colour = segment_names.get(pred_seg, (f"Segment {pred_seg}", "#9ca3af"))

        st.markdown(f"""
        <div class='ai-card' style='margin-top:1rem;border-left:4px solid {seg_colour};'>
            <h3 style='color:{seg_colour};font-size:1.1rem;'>{seg_name}</h3>
            <p>This customer belongs to <b>Cluster {pred_seg}</b>. 
            Tailor your marketing strategy accordingly — 
            {'offer loyalty rewards and early access' if 'VIP' in seg_name or 'Loyal' in seg_name
             else 'send a win-back campaign with a discount code' if 'Risk' in seg_name
             else 'nurture with onboarding offers and first-purchase incentives'}.</p>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem;padding-top:1.5rem;border-top:1px solid #1e2230;
            text-align:center;font-size:0.78rem;color:#374151;'>
    Smart E-commerce AI System · Built with Python, Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)