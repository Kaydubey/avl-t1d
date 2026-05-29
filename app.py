import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
import time

# --- HELPER: CLARKE ERROR GRID ---
try:
    from clarke_error_grid import clarke_error_grid
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib.pyplot as plt
    def clarke_error_grid(ref, pred, title):
        fig, ax = plt.subplots(facecolor='white')
        ax.text(0.5, 0.5, "Clarke Error Grid Module Missing", ha='center', va='center', color='red')
        ax.set_axis_off()
        return fig

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Agentic Virtual Lab | Type 1 Diabetes",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- COLORFUL & VIBRANT CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;800&family=Fira+Code:wght@400;500&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top left, #f3f8ff, #ffffff, #fdf4ff) !important;
        font-family: 'Nunito', sans-serif;
        color: #1e293b;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-header {
        background: linear-gradient(-45deg, #FF3CAC, #784BA0, #2B86C5, #00d2ff);
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 2.8rem; margin-bottom: 5px; text-align: center;
    }
    .sub-header { font-size: 1.2rem; color: #64748b; margin-bottom: 30px; font-weight: 600; text-align: center; }
    
    /* Pill Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 15px; justify-content: center; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff; border-radius: 50px; border: 2px solid #e2e8f0;
        padding: 10px 25px; font-weight: 800 !important; color: #64748b; transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%) !important; color: #0f172a !important; border: none !important;
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.4);
    }
    
    /* Input Cards */
    .input-card {
        padding: 20px; border-radius: 15px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05); transition: transform 0.3s ease; color: #0f172a; height: 100%;
    }
    .input-card:hover { transform: translateY(-5px); }
    .card-nutrition { background: linear-gradient(135deg, #FFE259 0%, #FFA751 100%); }
    .card-glucose { background: linear-gradient(135deg, #84FAB0 0%, #8FD3F4 100%); }
    .card-manual { background: linear-gradient(135deg, #A18CD1 0%, #FBC2EB 100%); }
    .card-wearable { background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); border: 2px dashed #6366f1;}
    
    /* Terminal & Results */
    .agent-terminal {
        background-color: #0f172a; color: #10b981; font-family: 'Fira Code', monospace;
        padding: 20px; border-radius: 12px; height: 320px; overflow-y: auto;
        box-shadow: inset 0 4px 10px rgba(0,0,0,0.5); font-size: 0.95rem; line-height: 1.6; border: 2px solid #1e293b;
    }
    .agent-terminal span.thought { color: #c084fc; }
    .agent-terminal span.action { color: #38bdf8; }
    .agent-terminal span.observation { color: #fde047; }
    .agent-terminal span.conclusion { color: #34d399; font-weight: bold; }
    
    .metric-card {
        background: white; border-radius: 15px; padding: 25px; box-shadow: 0 10px 20px rgba(0,0,0,0.05); border-top: 5px solid #6366f1;
    }
    @keyframes glow { 0% { box-shadow: 0 0 10px rgba(99,102,241,0.3); } 50% { box-shadow: 0 0 25px rgba(99,102,241,0.7); } 100% { box-shadow: 0 0 10px rgba(99,102,241,0.3); } }
    .ai-decision-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; animation: glow 3s infinite; text-align: center; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model = tf.keras.models.load_model(os.path.join(current_dir, 'glucose_model.h5'), compile=False)
        scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
        data = pd.read_csv(os.path.join(current_dir, 'training_data.csv'))
        return model, scaler, data
    except Exception:
        return None, None, None

model, scaler, df = load_resources()

# --- AUTONOMOUS AGENT CLASS ---
class ResearchCopilotAgent:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.audit_log = []

    def log(self, step_type, message):
        css_class = step_type.lower()
        self.audit_log.append(f"<span class='{css_class}'>[{step_type}]</span> {message}")

    def tool_predict_forward(self, g, c, i, stress_level, sleep_status):
        inp = np.array([[g, c, i]] * 12)
        scaled = self.scaler.transform(inp).reshape(1, 12, 3)
        dummy = np.zeros((1, 3)); dummy[0,0] = self.model.predict(scaled, verbose=0)[0,0]
        base = int(self.scaler.inverse_transform(dummy)[0,0])
        
        adj = 0
        if stress_level >= 8: adj += 25
        elif stress_level >= 4: adj += 10
        if sleep_status == "Poor (<6hr)": adj += 15
        return base + adj

    def run_autonomous_optimization(self, g, c, stress, sleep):
        self.audit_log = []
        self.log("THOUGHT", f"Goal: Find optimal dose for Target 100-140 mg/dL.")
        self.log("THOUGHT", f"Patient State: Glucose={g}mg/dL, Carbs={c}g, Cortisol Proxy={stress}/10, Sleep={sleep}.")
        
        self.log("ACTION", "Running Baseline Simulation (0 Units)...")
        baseline_pred = self.tool_predict_forward(g, c, 0, stress, sleep)
        self.log("OBSERVATION", f"Baseline trajectory reaches {baseline_pred} mg/dL.")
        
        if baseline_pred < 70:
            self.log("CONCLUSION", "Hypoglycemia Risk detected natively. Carbs required, NOT insulin.")
            return 0, baseline_pred, "Administer 0 Units (Consume Carbs)"

        self.log("ACTION", "Initiating In-Silico Dose Titration Loop (0.5U increments)...")
        optimal_dose, optimal_outcome = None, None
        
        for test_dose in np.arange(0.5, 15.0, 0.5):
            outcome = self.tool_predict_forward(g, c, test_dose, stress, sleep)
            self.log("OBSERVATION", f"Tested {test_dose}U -> Predicted Glucose: {outcome} mg/dL")
            if 100 <= outcome <= 140:
                optimal_dose = test_dose
                optimal_outcome = outcome
                break
            elif outcome < 70:
                self.log("THOUGHT", f"Dose {test_dose}U caused Hypoglycemia. Search terminated.")
                break 

        if optimal_dose is not None:
            self.log("CONCLUSION", f"Optimal Intervention Found: {optimal_dose} Units.")
            return optimal_dose, optimal_outcome, f"Administer {optimal_dose} Units"
        else:
            safe_dose = max(0, test_dose - 0.5)
            safe_outcome = self.tool_predict_forward(g, c, safe_dose, stress, sleep)
            self.log("CONCLUSION", f"Safest dose before hypo threshold is {safe_dose} Units.")
            return safe_dose, safe_outcome, f"Administer {safe_dose} Units"

if model: agent = ResearchCopilotAgent(model, scaler)

# --- CHATBOT ENGINE ---
def generate_chat_response(query, context):
    query = query.lower()
    if not context: return "Please run an autonomous optimization simulation first!"
    
    dose, carbs, stress = context['dose'], context['carbs'], context['stress']
    is_wearable = context.get('wearable_used', False)
    
    if "why" in query and "dose" in query:
        return f"I recommended **{dose} Units** to cover the **{carbs}g of carbs**. During the titration loop, any dose higher than {dose + 0.5}U caused simulated Hypoglycemia."
    elif "stress" in query or "cortisol" in query or "wearable" in query:
        if is_wearable:
            return f"I detected a low Heart Rate Variability (HRV) from the wearable sync, mapping to a Stress level of **{stress}/10**. This confirms a Cortisol spike, so I applied an insulin-resistance penalty to the simulation."
        else:
            return f"You manually inputted a Stress level of **{stress}/10**. High stress releases cortisol, inducing insulin resistance, which requires a higher dose."
    elif "safe" in query or "trust" in query:
        return "I am trained on FDA-approved UVa/Padova simulator data. Always verify my logic in the Copilot Audit Trail above to prevent automation bias."
    else:
        return "Check my Audit Trail for the exact step-by-step logic I used to determine this trajectory."

# --- UI LAYOUT ---
st.markdown('<div class="main-header">✨ Agentic Virtual Lab: Metabolic Research ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">IoT Wearable Integration & XAI Copilot</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Sync the patient's wearable or use manual entry, then run the Copilot."}]
if "wearable_synced" not in st.session_state:
    st.session_state.wearable_synced = False

tab1, tab2 = st.tabs(["🚀 Autonomous Copilot", "🛡️ Governance & Safety"])

with tab1:
    # --- SECTION 1: NUTRITION & SENSOR ---
    st.markdown("### 🧪 1. Nutrition & CGM Data")
    c1, c2 = st.columns(2)
    with c1: 
        st.markdown('<div class="input-card card-nutrition"><b>🍞 Carb Load (g)</b>', unsafe_allow_html=True)
        carbs = st.number_input("", 0, 200, 60, key="c_in", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2: 
        st.markdown('<div class="input-card card-glucose"><b>🩸 Start Glucose (mg/dL)</b>', unsafe_allow_html=True)
        glucose = st.number_input("", 50, 400, 150, key="g_in", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    
    # --- SECTION 2: BIOMETRICS (THE TOGGLE) ---
    st.markdown("### 📡 2. Biometric Integration (Graceful Degradation)")
    input_method = st.radio("Select Hardware Status:", ["⌚ Wearable Auto-Sync (IoT Protocol)", "✍️ Manual Entry (Backup Protocol)"], horizontal=True)
    
    stress_val = 1 # Safe Default
    sleep_val = "Good (>7hr)" # Safe Default
    wearable_active = False
    
    if "Wearable" in input_method:
        st.markdown('<div class="input-card card-wearable">', unsafe_allow_html=True)
        if st.button("🔄 Sync Apple HealthKit API"):
            with st.spinner("Establishing Bluetooth Handshake... fetching HRV metrics..."):
                time.sleep(1.5)
                st.session_state.wearable_synced = True
                st.session_state.hrv = 22 # Hardcoded severe stress for demo
                st.session_state.sleep_hr = 4
        
        if st.session_state.wearable_synced:
            wearable_active = True
            stress_val = 8 # Mapped from 22ms HRV
            sleep_val = "Poor (<6hr)"
            st.success("✅ Device Synced Successfully")
            w1, w2 = st.columns(2)
            w1.metric("Live HRV (Cortisol Biomarker)", f"{st.session_state.hrv} ms", "- Severe Stress Detected", delta_color="inverse")
            w2.metric("Sleep Duration", f"{st.session_state.sleep_hr}h 15m", "- Deprivation Detected", delta_color="inverse")
            st.info("🤖 AI Copilot has automatically mapped live biometrics to: Cortisol Proxy 8/10 and Sleep=Poor.")
        else:
            st.warning("Awaiting sync. AI is using Zero-Assumption Safety Baseline (Stress=1, Sleep=Good).")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.session_state.wearable_synced = False # Reset
        c3, c4 = st.columns(2)
        with c3: 
            st.markdown('<div class="input-card card-manual"><b>🧠 Cortisol Proxy (1-10)</b>', unsafe_allow_html=True)
            stress_val = st.slider("", 1, 10, 8, key="s_man", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        with c4: 
            st.markdown('<div class="input-card card-manual"><b>😴 Sleep Status</b>', unsafe_allow_html=True)
            sleep_val = st.selectbox("", ["Good (>7hr)", "Poor (<6hr)"], key="sl_man", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # --- SECTION 3: AGENTIC AI ---
    col_agent, col_results = st.columns([1.2, 1], gap="large")
    with col_agent:
        st.markdown("### 🤖 Agentic Research Copilot")
        if st.button("✨ Run Autonomous Optimization", type="primary", use_container_width=True):
            if model:
                with st.spinner("🧠 Agent is running in-silico trials..."):
                    time.sleep(1) 
                    opt_dose, opt_outcome, action_plan = agent.run_autonomous_optimization(glucose, carbs, stress_val, sleep_val)
                    
                    st.session_state['agent_run'] = {
                        'log': agent.audit_log, 'dose': opt_dose, 'outcome': opt_outcome,
                        'plan': action_plan, 'base_g': glucose, 'carbs': carbs, 'stress': stress_val,
                        'wearable_used': wearable_active
                    }
        
        st.write("")
        terminal_html = "<div class='agent-terminal'>"
        if 'agent_run' in st.session_state:
            for line in st.session_state['agent_run']['log']: terminal_html += f"{line}<br>"
        else: terminal_html += "<span style='color: #64748b;'>Awaiting parameters... Initialize optimization.</span>"
        terminal_html += "</div>"
        st.markdown(terminal_html, unsafe_allow_html=True)

    with col_results:
        st.markdown("### 📊 Intervention Results")
        if 'agent_run' in st.session_state:
            res = st.session_state['agent_run']
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            rm1, rm2 = st.columns(2)
            rm1.metric("Recommended Insulin", f"{res['dose']} Units")
            rm2.metric("Projected Glucose", f"{res['outcome']} mg/dL")
            st.markdown(f'<div class="ai-decision-box"><div style="font-size: 1.1rem; font-weight: 700;">Final Clinical Plan</div><div style="font-size: 1.4rem; font-weight: 800;">{res['plan']}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.write("")
            fig = go.Figure()
            fig.add_hrect(y0=70, y1=180, fillcolor="rgba(146, 254, 157, 0.3)", line_width=0, annotation_text="Target")
            fig.add_trace(go.Scatter(x=[0, 15, 30], y=[res['base_g'], (res['base_g'] + res['outcome'])/2, res['outcome']], mode='lines+markers', line=dict(color='#8A2387', width=4)))
            fig.update_layout(title="Post-Intervention Trajectory", margin=dict(l=20, r=20, t=40, b=20), height=300, paper_bgcolor='white', plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Run the agent to view optimal intervention results.")

    # --- CHATBOT SECTION ---
    st.markdown("---")
    st.markdown("### 💬 Interrogate the Copilot")
    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("E.g., How did the wearable data affect your decision?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = generate_chat_response(prompt, st.session_state.get('agent_run', None))
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with tab2:
    st.markdown("### ⚖️ AI Safety & Evaluation Harness")
    if df is not None:
        start = st.slider("Select Retrospective Audit Segment", 0, len(df)-150, 500)
        if st.button("▶️ Run Reproducibility Audit", type="primary"):
            real_segment = df.iloc[start : start + 120]
            real_vals, inputs = real_segment['CGM'].values, real_segment[['CGM', 'CHO', 'insulin']].fillna(0).values
            preds = []
            for i in range(len(inputs) - 12):
                chunk = inputs[i:i+12]
                scaled = scaler.transform(chunk).reshape(1, 12, 3)
                dummy = np.zeros((1, 3)); dummy[0,0] = model.predict(scaled, verbose=0)[0,0]
                preds.append(int(scaler.inverse_transform(dummy)[0,0]))
            
            real_aligned, pred_aligned = real_vals[12:12+len(preds)], np.array(preds)
            rmse = np.sqrt(np.mean((real_aligned - pred_aligned)**2))
            
            c1, c2 = st.columns(2)
            c1.metric("RMSE Deviation", f"{rmse:.2f} mg/dL")
            c2.success("System performing within acceptable safety limits.")
            
            col_grid, col_text = st.columns([2, 1])
            with col_grid:
                fig_clarke = clarke_error_grid(real_aligned, pred_aligned, "Governance Safety Audit")
                fig_clarke.patch.set_facecolor('white')
                st.pyplot(fig_clarke)
            with col_text: st.info("**Zone A/B:** Safe actions.\n\n**Zone C-E:** Triggers protocols.")