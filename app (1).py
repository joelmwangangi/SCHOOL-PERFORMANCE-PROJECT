"""
╔══════════════════════════════════════════════════════════════╗
║      STUDENT PERFORMANCE PREDICTION SYSTEM – Streamlit       ║
║      ANN-powered early warning system for at-risk students   ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, time, os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Header ── */
  .main-header {
    background: linear-gradient(135deg, #4361ee, #7209b7);
    padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 1.5rem;
  }
  .main-header h1 { color: white; font-size: 2.2rem; margin: 0; }
  .main-header p  { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 0.5rem 0 0 0; }

  /* ── KPI cards ── */
  .metric-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1.2rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .metric-card .value { font-size: 2rem; font-weight: 700; color: #4361ee; }
  .metric-card .label { font-size: 0.85rem; color: #64748b; margin-top: 4px; }

  /* ── Result boxes ── */
  .pass-box { background: #d1fae5; border-left: 5px solid #10b981;
              padding: 1.2rem; border-radius: 8px; }
  .fail-box { background: #fee2e2; border-left: 5px solid #ef4444;
              padding: 1.2rem; border-radius: 8px; }

  /* ── Section headers ── */
  .section-title {
    font-size: 1rem; font-weight: 700; color: #4361ee;
    margin: 1.2rem 0 0.6rem 0; display: flex; align-items: center; gap: 6px;
  }

  /* ── Stepper widget card ── */
  .field-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 0.55rem 0.75rem 0.45rem;
    margin-bottom: 0.55rem;
  }
  .field-label {
    font-size: 0.72rem; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.04em;
    margin-bottom: 2px;
  }
  .stepper-row {
    display: flex; align-items: center; justify-content: space-between;
  }
  .stepper-val {
    font-size: 1.05rem; font-weight: 600; color: #1e293b;
    min-width: 40px; text-align: left;
  }
  .stepper-btns { display: flex; gap: 4px; }
  .step-btn {
    width: 28px; height: 28px; border-radius: 6px;
    border: 1.5px solid #cbd5e1; background: #f8fafc;
    font-size: 1.1rem; font-weight: 700; color: #4361ee;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: all 0.15s;
  }
  .step-btn:hover { background: #4361ee; color: white; border-color: #4361ee; }

  /* ── Dropdowns ── */
  .field-select select {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    font-size: 0.95rem !important;
    padding: 0.4rem 0.6rem !important;
  }

  /* ── Override Streamlit number_input to look like stepper ── */
  div[data-testid="stNumberInput"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.3rem 0.5rem;
  }
  div[data-testid="stNumberInput"] label {
    font-size: 0.72rem !important; font-weight: 600 !important;
    color: #64748b !important; text-transform: uppercase;
    letter-spacing: 0.04em !important;
  }
  div[data-testid="stNumberInput"] input {
    font-size: 1.05rem !important; font-weight: 600 !important;
    color: #1e293b !important; border: none !important;
    background: transparent !important; padding: 0 !important;
  }
  /* +/- stepper buttons inside number_input */
  div[data-testid="stNumberInput"] button {
    background: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 6px !important;
    color: #4361ee !important;
    font-weight: 700 !important;
    width: 28px !important; height: 28px !important;
    padding: 0 !important;
    transition: all 0.15s !important;
  }
  div[data-testid="stNumberInput"] button:hover {
    background: #4361ee !important;
    color: white !important;
    border-color: #4361ee !important;
  }

  /* ── Selectbox ── */
  div[data-testid="stSelectbox"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.3rem 0.5rem;
    margin-bottom: 0.3rem;
  }
  div[data-testid="stSelectbox"] label {
    font-size: 0.72rem !important; font-weight: 600 !important;
    color: #64748b !important; text-transform: uppercase;
    letter-spacing: 0.04em !important;
  }
  div[data-testid="stSelectbox"] > div > div {
    border: none !important; background: transparent !important;
    font-size: 0.98rem !important; font-weight: 600 !important; color: #1e293b !important;
  }

  /* ── Primary action button ── */
  .stButton > button[kind="primary"],
  .stButton > button {
    background: linear-gradient(135deg, #4361ee, #7209b7) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    padding: 0.6rem 2rem !important; font-size: 1rem !important;
    font-weight: 600 !important; transition: opacity 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] { background: #f8fafc; }

  /* ── Form section divider ── */
  .form-section {
    background: #f8fafc; border-radius: 12px;
    padding: 1rem 1.2rem; margin-bottom: 1rem;
    border: 1px solid #e2e8f0;
  }
  .form-section-title {
    font-size: 0.9rem; font-weight: 700; color: #4361ee;
    margin-bottom: 0.8rem; display: flex; align-items: center; gap: 6px;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'history' not in st.session_state:
    st.session_state.history = []
if 'registered_users' not in st.session_state:
    st.session_state.registered_users = {
        'admin':   {'password': 'admin123',   'role': 'admin',   'name': 'System Admin'},
        'teacher': {'password': 'teacher123', 'role': 'teacher', 'name': 'Mr. Kamau'},
        'student': {'password': 'student123', 'role': 'student', 'name': 'Alice Wanjiku'},
    }

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    ann_model = joblib.load(os.path.join(base, 'models', 'ann_model.pkl'))
    scaler    = joblib.load(os.path.join(base, 'models', 'scaler.pkl'))
    le_map    = joblib.load(os.path.join(base, 'models', 'label_encoders.pkl'))
    with open(os.path.join(base, 'models', 'model_meta.json')) as f:
        meta = json.load(f)
    return ann_model, scaler, le_map, meta

# Always define globals so nothing is ever undefined
MODEL_LOADED = False
MODEL_ERROR  = ""
ann          = None
scaler       = None
le_map       = {}
meta         = {
    'features': [], 'test_accuracy': 0, 'auc_roc': 0,
    'n_epochs_run': 0, 'hidden_layers': [], 'activation': 'relu'
}

try:
    ann, scaler, le_map, meta = load_model()
    MODEL_LOADED = True
except Exception as _load_exc:
    MODEL_ERROR = str(_load_exc)


# ── Shared error banner ───────────────────────────────────────────────────────
def _model_error_banner():
    st.error("⚠️ Model files not found. The `models/` folder must be committed to your GitHub repo.")
    st.code(f"Error: {MODEL_ERROR}", language="bash")
    st.info("""
**Required repo structure for Streamlit Cloud:**
```
your-repo/
├── app.py
├── requirements.txt
├── models/
│   ├── ann_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── model_meta.json
└── assets/
    └── *.png
```
**Fix steps:**
1. Run the Jupyter notebook locally → generates `models/` and `assets/`
2. `git add models/ assets/` → commit → push to GitHub
3. Redeploy on Streamlit Cloud
    """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def show_login():
    st.markdown("""
    <div class="main-header">
      <h1>🎓 Student Performance AI</h1>
      <p>ANN-powered early-warning system for academic success</p>
    </div>""", unsafe_allow_html=True)

    tab_login, tab_reg = st.tabs(["🔐 Login", "📝 Register"])

    with tab_login:
        col_mid = st.columns([1, 2, 1])[1]
        with col_mid:
            st.subheader("Sign In")
            username = st.text_input("Username", key="li_user")
            password = st.text_input("Password", type="password", key="li_pw")
            role     = st.selectbox("Login as", ["student", "teacher", "admin"])
            if st.button("Login", use_container_width=True):
                users = st.session_state.registered_users
                if (username in users
                        and users[username]['password'] == password
                        and users[username]['role'] == role):
                    st.session_state.logged_in = True
                    st.session_state.user_role = role
                    st.session_state.username  = username
                    st.success(f"Welcome, {users[username]['name']}!")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials or role mismatch.")
            st.caption("Demo: admin/admin123 · teacher/teacher123 · student/student123")

    with tab_reg:
        col_mid2 = st.columns([1, 2, 1])[1]
        with col_mid2:
            st.subheader("Create Account")
            new_name = st.text_input("Full Name",  key="reg_name")
            new_user = st.text_input("Username",   key="reg_user")
            new_pw   = st.text_input("Password",   type="password", key="reg_pw")
            new_pw2  = st.text_input("Confirm",    type="password", key="reg_pw2")
            new_role = st.selectbox("Role", ["student", "teacher"], key="reg_role")
            if st.button("Register", use_container_width=True):
                if not all([new_name, new_user, new_pw, new_pw2]):
                    st.warning("Please fill all fields.")
                elif new_pw != new_pw2:
                    st.error("Passwords don't match.")
                elif new_user in st.session_state.registered_users:
                    st.error("Username already taken.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    st.session_state.registered_users[new_user] = {
                        'password': new_pw, 'role': new_role, 'name': new_name
                    }
                    st.success(f"✅ Account created for {new_name}! Please login.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICTION LOGIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def predict_student(data_dict):
    row = {}
    cat_defaults = {
        'school': 'GP', 'sex': 'F', 'address': 'U', 'famsize': 'GT3',
        'Pstatus': 'T', 'Mjob': 'other', 'Fjob': 'other', 'reason': 'course',
        'guardian': 'mother', 'schoolsup': 'no', 'famsup': 'no', 'paid': 'no',
        'activities': 'no', 'nursery': 'yes', 'higher': 'yes',
        'internet': 'yes', 'romantic': 'no'
    }
    for col, default in cat_defaults.items():
        val = data_dict.get(col, default)
        if col in le_map:
            try:
                val = le_map[col].transform([val])[0]
            except Exception:
                val = 0
        row[col] = val

    for c in ['age','Medu','Fedu','traveltime','studytime','failures',
              'famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2']:
        row[c] = data_dict.get(c, 0)

    g1, g2 = row['G1'], row['G2']
    row['G1G2_avg']   = (g1 + g2) / 2
    row['G1G2_prod']  = g1 * g2
    row['G1G2_diff']  = g2 - g1
    row['study_fail'] = row['studytime'] * (row['failures'] + 1)
    row['abs_study']  = row['absences'] / (row['studytime'] + 1)

    X    = np.array([[row.get(f, 0) for f in meta['features']]])
    X_sc = scaler.transform(X)
    pred  = ann.predict(X_sc)[0]
    proba = ann.predict_proba(X_sc)[0]
    return int(pred), float(proba[1])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def show_sidebar():
    with st.sidebar:
        st.markdown("### 🎓 Student Performance AI")
        uname = st.session_state.username
        st.markdown(f"**User:** {st.session_state.registered_users[uname]['name']}")
        st.markdown(f"**Role:** `{st.session_state.user_role.capitalize()}`")
        st.divider()

        role  = st.session_state.user_role
        pages = {
            'student': ["🏠 Dashboard", "📊 My Prediction", "📜 My History"],
            'teacher': ["🏠 Dashboard", "🔮 Predict Student", "📁 Batch Upload",
                        "📜 Prediction History", "📈 Analytics"],
            'admin':   ["🏠 Dashboard", "🔮 Predict Student", "📁 Batch Upload",
                        "📜 Prediction History", "📈 Analytics",
                        "🤖 Model Info", "👥 Manage Users"],
        }
        page = st.radio("Navigation", pages[role], label_visibility="collapsed")
        st.divider()
        if MODEL_LOADED:
            st.success(f"✅ Model ready\n\nAccuracy: **{meta['test_accuracy']:.1%}**")
        else:
            st.error("⚠️ Model not loaded\nCheck Dashboard for fix.")
        if st.button("🔓 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_role = ''
            st.session_state.username  = ''
            st.rerun()
        return page


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEPPER INPUT FORM  (− value + style)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def student_input_form(key_prefix=""):
    """Renders the student data entry form using number_input (stepper) widgets
    styled to match the − value + card layout in the reference image."""

    # ── Section 1: Personal Information ──────────────────────────────────────
    st.markdown('<div class="form-section-title">👤 Personal Information</div>',
                unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", min_value=15, max_value=22, value=17, step=1,
                                  key=f"{key_prefix}_age")
        with c2:
            sex = st.selectbox("Sex", ["Female (F)", "Male (M)"],
                               key=f"{key_prefix}_sex")
            sex = "F" if "F" in sex else "M"
        with c3:
            school = st.selectbox("School", ["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"],
                                  key=f"{key_prefix}_school")
            school = "GP" if "GP" in school else "MS"
        with c4:
            address = st.selectbox("Address", ["Urban (U)", "Rural (R)"],
                                   key=f"{key_prefix}_address")
            address = "U" if "U" in address else "R"

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            famsize = st.selectbox("Family Size", ["Greater than 3 (GT3)", "Less/Equal 3 (LE3)"],
                                   key=f"{key_prefix}_famsize")
            famsize = "GT3" if "GT3" in famsize else "LE3"
        with c6:
            Pstatus = st.selectbox("Parent Status", ["Together (T)", "Apart (A)"],
                                   key=f"{key_prefix}_Pstatus")
            Pstatus = "T" if "T" in Pstatus else "A"
        with c7:
            guardian = st.selectbox("Guardian", ["Mother", "Father", "Other"],
                                    key=f"{key_prefix}_guardian")
            guardian = guardian.lower()
        with c8:
            romantic = st.selectbox("In a Relationship", ["No", "Yes"],
                                    key=f"{key_prefix}_romantic")
            romantic = romantic.lower()

    st.divider()

    # ── Section 2: Family Background ─────────────────────────────────────────
    st.markdown('<div class="form-section-title">👨‍👩‍👧 Family Background</div>',
                unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            Medu = st.number_input("Mother's Education (0–4)", min_value=0, max_value=4,
                                   value=2, step=1, key=f"{key_prefix}_Medu",
                                   help="0=none, 1=primary, 2=5th-9th, 3=secondary, 4=higher")
        with c2:
            Fedu = st.number_input("Father's Education (0–4)", min_value=0, max_value=4,
                                   value=2, step=1, key=f"{key_prefix}_Fedu",
                                   help="0=none, 1=primary, 2=5th-9th, 3=secondary, 4=higher")
        with c3:
            Mjob = st.selectbox("Mother's Job",
                                ["Teacher", "Health", "Services", "At Home", "Other"],
                                key=f"{key_prefix}_Mjob")
            Mjob = Mjob.lower().replace(" ", "_") if Mjob != "At Home" else "at_home"
        with c4:
            Fjob = st.selectbox("Father's Job",
                                ["Teacher", "Health", "Services", "At Home", "Other"],
                                key=f"{key_prefix}_Fjob")
            Fjob = Fjob.lower().replace(" ", "_") if Fjob != "At Home" else "at_home"

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            famrel = st.number_input("Family Relations (1–5)", min_value=1, max_value=5,
                                     value=3, step=1, key=f"{key_prefix}_famrel",
                                     help="1=very bad → 5=excellent")
        with c6:
            famsup = st.selectbox("Family Support", ["Yes", "No"], key=f"{key_prefix}_famsup")
            famsup = famsup.lower()
        with c7:
            reason = st.selectbox("Reason for School",
                                  ["Course", "Home", "Reputation", "Other"],
                                  key=f"{key_prefix}_reason")
            reason = reason.lower()
        with c8:
            nursery = st.selectbox("Attended Nursery", ["Yes", "No"],
                                   key=f"{key_prefix}_nursery")
            nursery = nursery.lower()

    st.divider()

    # ── Section 3: Academic Performance ──────────────────────────────────────
    st.markdown('<div class="form-section-title">📚 Academic Performance</div>',
                unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            G1 = st.number_input("Grade Period 1 — G1", min_value=0, max_value=20,
                                 value=10, step=1, key=f"{key_prefix}_g1")
        with c2:
            G2 = st.number_input("Grade Period 2 — G2", min_value=0, max_value=20,
                                 value=10, step=1, key=f"{key_prefix}_g2")
        with c3:
            failures = st.number_input("Past Failures", min_value=0, max_value=3,
                                       value=0, step=1, key=f"{key_prefix}_fail")
        with c4:
            absences = st.number_input("Number of Absences", min_value=0, max_value=75,
                                       value=5, step=1, key=f"{key_prefix}_abs")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            studytime = st.number_input("Study Time (1–4)", min_value=1, max_value=4,
                                        value=2, step=1, key=f"{key_prefix}_study",
                                        help="1=<2h, 2=2–5h, 3=5–10h, 4=>10h")
        with c6:
            traveltime = st.number_input("Travel Time (1–4)", min_value=1, max_value=4,
                                         value=1, step=1, key=f"{key_prefix}_travel",
                                         help="1=<15min, 2=15–30min, 3=30–60min, 4=>1hr")
        with c7:
            schoolsup = st.selectbox("School Extra Support", ["No", "Yes"],
                                     key=f"{key_prefix}_ssup")
            schoolsup = schoolsup.lower()
        with c8:
            paid = st.selectbox("Extra Paid Classes", ["No", "Yes"],
                                key=f"{key_prefix}_paid")
            paid = paid.lower()

        c9, c10 = st.columns([1, 3])
        with c9:
            higher = st.selectbox("Wants Higher Education", ["Yes", "No"],
                                  key=f"{key_prefix}_higher")
            higher = higher.lower()

    st.divider()

    # ── Section 4: Lifestyle ──────────────────────────────────────────────────
    st.markdown('<div class="form-section-title">🌐 Lifestyle & Social</div>',
                unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            freetime = st.number_input("Free Time After School (1–5)", min_value=1,
                                       max_value=5, value=3, step=1,
                                       key=f"{key_prefix}_free",
                                       help="1=very low → 5=very high")
        with c2:
            goout = st.number_input("Going Out with Friends (1–5)", min_value=1,
                                    max_value=5, value=3, step=1,
                                    key=f"{key_prefix}_goout",
                                    help="1=very low → 5=very high")
        with c3:
            Dalc = st.number_input("Weekday Alcohol Use (1–5)", min_value=1,
                                   max_value=5, value=1, step=1,
                                   key=f"{key_prefix}_dalc",
                                   help="1=very low → 5=very high")
        with c4:
            Walc = st.number_input("Weekend Alcohol Use (1–5)", min_value=1,
                                   max_value=5, value=1, step=1,
                                   key=f"{key_prefix}_walc",
                                   help="1=very low → 5=very high")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            health = st.number_input("Health Status (1–5)", min_value=1, max_value=5,
                                     value=3, step=1, key=f"{key_prefix}_health",
                                     help="1=very bad → 5=very good")
        with c6:
            internet = st.selectbox("Internet at Home", ["Yes", "No"],
                                    key=f"{key_prefix}_internet")
            internet = internet.lower()
        with c7:
            activities = st.selectbox("Extra-Curricular Activities", ["No", "Yes"],
                                      key=f"{key_prefix}_act")
            activities = activities.lower()

    return dict(
        school=school, sex=sex, age=int(age), address=address, famsize=famsize,
        Pstatus=Pstatus, Medu=int(Medu), Fedu=int(Fedu), Mjob=Mjob, Fjob=Fjob,
        traveltime=int(traveltime), studytime=int(studytime), failures=int(failures),
        absences=int(absences), G1=int(G1), G2=int(G2), health=int(health),
        internet=internet, schoolsup=schoolsup, famsup=famsup, paid=paid,
        activities=activities, higher=higher, romantic=romantic,
        Dalc=int(Dalc), Walc=int(Walc), famrel=int(famrel),
        freetime=int(freetime), goout=int(goout),
        reason=reason, guardian=guardian, nursery=nursery
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def page_dashboard():
    st.markdown("""
    <div class="main-header">
      <h1>🎓 Student Performance AI</h1>
      <p>Artificial Neural Network · Early Warning System · Academic Success Prediction</p>
    </div>""", unsafe_allow_html=True)

    if not MODEL_LOADED:
        _model_error_banner(); return

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (f"{meta.get('test_accuracy', 0):.1%}", "ANN Test Accuracy"),
        (f"{meta.get('auc_roc', 0):.3f}",       "AUC-ROC Score"),
        (str(meta.get('n_epochs_run', '–')),     "Training Epochs"),
        (str(len(st.session_state.history)),     "Predictions Made"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="value">{val}</div>
              <div class="label">{lbl}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📋 About This System")
        st.markdown("""
        This system uses an **Artificial Neural Network (ANN)** with 4 hidden layers
        to predict whether a student will **pass or fail** their final exam.

        **Features used (37 total):**
        - 📚 Academic: G1, G2 grades, study time, failures
        - 🏫 School: school type, extra support, reason for choice
        - 👨‍👩‍👧 Family: parental education, job, family size
        - 🌐 Lifestyle: absences, alcohol consumption, internet access

        **ANN Architecture:**
        `Input(37) → 256 → 128 → 64 → 32 → Output(2)`
        """)
    with col_b:
        st.subheader("📉 Training Loss Curve")
        base     = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base, 'assets', 'loss_curve.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Run the Jupyter notebook to generate plots.")

    st.subheader("📖 Quick Start Guide")
    role = st.session_state.user_role
    if role == 'student':
        st.info("👉 Go to **My Prediction** to enter your details and get a forecast.")
    elif role == 'teacher':
        st.info("👉 Use **Predict Student** for individuals or **Batch Upload** for a full class CSV.")
    else:
        st.info("👉 Full access: predict students, upload batches, view analytics, and manage users.")


def show_prediction_result(pred, proba, student_name="Student"):
    st.markdown("---")
    st.subheader("🤖 ANN Prediction Result")
    col_r, col_g = st.columns(2)
    with col_r:
        if pred == 1:
            st.markdown(f"""
            <div class="pass-box">
              <h2 style="color:#065f46;margin:0">✅ PASS</h2>
              <p style="margin:8px 0 0 0;color:#065f46">
                <strong>{student_name}</strong> is predicted to <strong>pass</strong>
                with <strong>{proba:.1%}</strong> confidence.
              </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="fail-box">
              <h2 style="color:#7f1d1d;margin:0">⚠️ AT-RISK (Fail)</h2>
              <p style="margin:8px 0 0 0;color:#7f1d1d">
                <strong>{student_name}</strong> is at risk of failing.
                Pass probability: <strong>{proba:.1%}</strong>.
              </p>
            </div>""", unsafe_allow_html=True)
    with col_g:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        for start, end, color in [(0, 0.4, '#ef4444'),(0.4, 0.7, '#f59e0b'),(0.7, 1.0, '#10b981')]:
            ax.barh(0, end - start, left=start, height=0.4, color=color, alpha=0.7)
        ax.axvline(proba, color='#1e293b', lw=3, label=f'Score: {proba:.1%}')
        ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.8)
        ax.set_xticks([0, 0.4, 0.7, 1.0])
        ax.set_xticklabels(['0%', '40%', '70%', '100%'])
        ax.set_yticks([]); ax.set_xlabel('Pass Probability')
        ax.set_title('Prediction Confidence'); ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    risk_level = "Low Risk 🟢" if proba > 0.7 else ("Medium Risk 🟡" if proba > 0.4 else "High Risk 🔴")
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| Pass Probability | {proba:.1%} |
| Fail Probability | {1-proba:.1%} |
| Risk Level | {risk_level} |
| Prediction | {'Pass ✅' if pred==1 else 'Fail ⚠️'} |
    """)
    if pred == 0 or proba < 0.6:
        st.markdown("#### 💡 Recommended Interventions")
        st.markdown("""
- 📚 **Academic Support** — Enrol in extra tutoring or paid classes
- 🏫 **School Counselling** — Schedule sessions with the academic advisor
- 📋 **Attendance Review** — Monitor and reduce absences
- 📖 **Study Plan** — Increase weekly study time to 5+ hours
- 👨‍👩‍👧 **Parent Meeting** — Engage family in academic support
        """)


def page_predict(title="🔮 Predict Student Performance"):
    st.header(title)
    if not MODEL_LOADED:
        _model_error_banner(); return
    student_name = st.text_input("Student Name (optional)", "Student")
    data = student_input_form("pred")
    if st.button("🔮 Run ANN Prediction", use_container_width=True):
        with st.spinner("Running ANN inference..."):
            time.sleep(0.4)
            pred, proba = predict_student(data)
        show_prediction_result(pred, proba, student_name)
        st.session_state.history.append({
            'timestamp':  datetime.now().strftime("%Y-%m-%d %H:%M"),
            'student':    student_name,
            'prediction': 'Pass' if pred == 1 else 'Fail',
            'pass_prob':  f"{proba:.1%}",
            'by':         st.session_state.username,
            'G1': data['G1'], 'G2': data['G2'],
        })
        st.success(f"Saved to history ({len(st.session_state.history)} total).")


def page_batch():
    st.header("📁 Batch Prediction – Upload CSV")
    if not MODEL_LOADED:
        _model_error_banner(); return
    st.markdown("""
Upload a CSV with student data. Required columns:
`school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian,
traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher,
internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2`
    """)
    uploaded = st.file_uploader("Upload student CSV", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Run Batch Prediction"):
            results = []
            bar = st.progress(0)
            for i, row in df.iterrows():
                pred, proba = predict_student(row.to_dict())
                results.append({
                    **row.to_dict(),
                    'Predicted': 'Pass' if pred == 1 else 'Fail',
                    'Pass_Prob': round(proba, 3),
                    'Risk': 'Low' if proba > 0.7 else ('Medium' if proba > 0.4 else 'High')
                })
                bar.progress((i + 1) / len(df))
            res_df = pd.DataFrame(results)
            st.success(f"✅ Processed {len(res_df)} students!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Pass",      (res_df['Predicted'] == 'Pass').sum())
            c2.metric("Fail",      (res_df['Predicted'] == 'Fail').sum())
            c3.metric("High Risk", (res_df['Risk'] == 'High').sum())
            st.dataframe(res_df[['Predicted','Pass_Prob','Risk'] +
                                 [c for c in res_df.columns
                                  if c not in ['Predicted','Pass_Prob','Risk']]])
            st.download_button("⬇️ Download Results CSV",
                               res_df.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")


def page_history():
    st.header("📜 Prediction History")
    hist = st.session_state.history
    if not hist:
        st.info("No predictions yet."); return
    df = pd.DataFrame(hist)
    st.dataframe(df, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Total Predictions", len(df))
    if 'prediction' in df.columns:
        c2.metric("Pass Rate", f"{(df['prediction']=='Pass').mean():.1%}")
    st.download_button("⬇️ Download History",
                       df.to_csv(index=False).encode(), "history.csv", "text/csv")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []; st.rerun()


def page_analytics():
    st.header("📈 Analytics Dashboard")
    base      = os.path.dirname(os.path.abspath(__file__))
    asset_dir = os.path.join(base, 'assets')
    plots = [
        ('grade_distribution.png', '📊 Grade Distributions'),
        ('correlation.png',        '🔥 Feature Correlation with G3'),
        ('loss_curve.png',         '📉 ANN Training Loss Curve'),
        ('confusion_matrix.png',   '🎯 Confusion Matrix'),
        ('roc_curve.png',          '📈 ROC Curve'),
        ('feature_importance.png', '🏆 Feature Importance'),
        ('eda_plots.png',          '🔍 EDA: Pass/Fail & Study Time'),
    ]
    found = False
    for fname, title in plots:
        path = os.path.join(asset_dir, fname)
        if os.path.exists(path):
            found = True
            st.subheader(title)
            st.image(path, use_container_width=True)
            st.divider()
    if not found:
        st.warning("No plots found in `assets/`. Run the Jupyter notebook then commit `assets/` to your repo.")


def page_model_info():
    st.header("🤖 Model Information")
    if not MODEL_LOADED:
        _model_error_banner(); return
    st.json(meta)
    st.markdown(f"""
### ANN Architecture
| Layer | Type   | Units | Activation |
|-------|--------|-------|------------|
| 0 | Input  | {len(meta['features'])} | — |
| 1 | Dense  | 256 | ReLU |
| 2 | Dense  | 128 | ReLU |
| 3 | Dense  |  64 | ReLU |
| 4 | Dense  |  32 | ReLU |
| 5 | Output |   2 | Softmax |

### Training Config
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| L2 Regularisation | 0.0001 |
| Early Stopping | Yes (patience = 40) |
| Epochs Completed | {meta.get('n_epochs_run','–')} |
| Validation Split | 12% |
    """)


def page_manage_users():
    st.header("👥 User Management")
    users = st.session_state.registered_users
    st.dataframe(
        pd.DataFrame([{'Username': u, 'Name': v['name'], 'Role': v['role']}
                      for u, v in users.items()]),
        use_container_width=True
    )
    st.subheader("Add New User")
    col1, col2 = st.columns(2)
    with col1:
        new_u = st.text_input("Username",  key="adm_u")
        new_n = st.text_input("Full Name", key="adm_n")
    with col2:
        new_p = st.text_input("Password", type="password", key="adm_p")
        new_r = st.selectbox("Role", ["student","teacher","admin"], key="adm_r")
    if st.button("➕ Add User"):
        if new_u and new_p and new_n:
            st.session_state.registered_users[new_u] = {
                'password': new_p, 'role': new_r, 'name': new_n
            }
            st.success(f"User '{new_u}' added!"); st.rerun()
        else:
            st.warning("Fill all fields.")


def page_student_prediction():
    st.header("📊 My Performance Prediction")
    if not MODEL_LOADED:
        _model_error_banner(); return
    uname     = st.session_state.username
    user_name = st.session_state.registered_users[uname]['name']
    st.markdown(f"Hello, **{user_name}**! Enter your details to get a prediction.")
    data = student_input_form("stu")
    if st.button("🔮 Predict My Performance", use_container_width=True):
        with st.spinner("Analysing your data..."):
            time.sleep(0.4)
            pred, proba = predict_student(data)
        show_prediction_result(pred, proba, user_name)
        st.session_state.history.append({
            'timestamp':  datetime.now().strftime("%Y-%m-%d %H:%M"),
            'student':    user_name,
            'prediction': 'Pass' if pred == 1 else 'Fail',
            'pass_prob':  f"{proba:.1%}",
            'by':         uname,
            'G1': data['G1'], 'G2': data['G2'],
        })


def page_student_history():
    st.header("📜 My Prediction History")
    uname = st.session_state.username
    hist  = [h for h in st.session_state.history if h.get('by') == uname]
    if not hist:
        st.info("You haven't made any predictions yet."); return
    st.dataframe(pd.DataFrame(hist), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if not st.session_state.logged_in:
    show_login()
else:
    page = show_sidebar()
    if page == "🏠 Dashboard":            page_dashboard()
    elif page == "📊 My Prediction":      page_student_prediction()
    elif page == "📜 My History":         page_student_history()
    elif page == "🔮 Predict Student":    page_predict()
    elif page == "📁 Batch Upload":       page_batch()
    elif page == "📜 Prediction History": page_history()
    elif page == "📈 Analytics":          page_analytics()
    elif page == "🤖 Model Info":         page_model_info()
    elif page == "👥 Manage Users":       page_manage_users()
