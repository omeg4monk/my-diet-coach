import sqlite3
import json
from datetime import date
from typing import List, Optional
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# 1. PAGE CONFIG & MODERN HIGH-CONTRAST CSS
# ==========================================

st.set_page_config(
    page_title="NutriSync AI — Adaptive Diet Coach",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Deep Navy, Crisp White, and Cool Mint / Sage accents
st.markdown("""
<style>
    /* Base background and text */
    .stApp {
        background-color: #0b1329;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Top Brand Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f1f3d 0%, #172a4d 50%, #1e3a5f 100%);
        border: 1px solid #2e4c7e;
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.45);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .hero-tagline {
        font-size: 1.05rem;
        color: #6ee7b7; /* Cool mint accent */
        font-weight: 600;
        margin-top: 4px;
        margin-bottom: 12px;
    }

    .hero-desc {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.55;
        margin: 0;
        max-width: 900px;
    }

    /* High-contrast Metric Cards */
    .metric-card {
        background-color: #111e38;
        border: 1.5px solid #263e69;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        margin-bottom: 15px;
    }

    .metric-label {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #cbd5e1 !important;
        text-transform: uppercase;
    }

    .metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 4px 0;
    }

    .metric-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
    }

    /* Content Cards */
    .content-box {
        background: #111e38;
        border: 1px solid #263e69;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Custom Streamlit Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
    }

    /* High Contrast Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d172e;
        border-radius: 12px;
        padding: 6px;
        border: 1px solid #1e3357;
        gap: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 10px 18px !important;
    }

    .stTabs [aria-selected="true"] {
        background: #1e3a63 !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
    }

    /* Streamlit labels readability */
    label, p, span {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PYDANTIC SCHEMAS FOR STRUCTURED GEMINI
# ==========================================

class FoodItem(BaseModel):
    name: str = Field(description="Dish or item name")
    estimated_grams: int = Field(description="Estimated portion weight in grams")
    calories: int = Field(description="Calories in kcal")
    protein_g: float = Field(description="Protein in grams")
    carbs_g: float = Field(description="Carbohydrates in grams")
    fat_g: float = Field(description="Fat in grams")
    fiber_g: float = Field(description="Fiber in grams")

class MealAnalysis(BaseModel):
    meal_summary: str = Field(description="Title of the meal")
    items: List[FoodItem] = Field(description="Itemized food components")
    total_calories: int
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    total_fiber_g: float
    nutritional_feedback: str = Field(description="Coaching notes on balance and micronutrients")

class DailyVoiceLog(BaseModel):
    meals: List[MealAnalysis] = Field(description="All distinct meals extracted from the voice note")
    day_summary: str = Field(description="Holistic daily summary of dietary intake")

class PlannedMeal(BaseModel):
    meal_slot: str
    dish_suggestion: str
    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float
    why_this_works: str

class AdaptiveDietPlan(BaseModel):
    strategy_note: str
    recommended_meals: List[PlannedMeal]

# ==========================================
# 3. SQLITE DATABASE ENGINE
# ==========================================

DB_FILE = "diet_tracker.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_date TEXT,
                meal_type TEXT,
                meal_summary TEXT,
                calories INTEGER,
                protein_g REAL,
                carbs_g REAL,
                fat_g REAL,
                fiber_g REAL,
                raw_json TEXT
            )
        """)

def save_meal(log_date: str, meal_type: str, analysis: MealAnalysis):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            INSERT INTO meals (
                log_date, meal_type, meal_summary, calories,
                protein_g, carbs_g, fat_g, fiber_g, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_date, meal_type, analysis.meal_summary, analysis.total_calories,
            analysis.total_protein_g, analysis.total_carbs_g, analysis.total_fat_g,
            analysis.total_fiber_g, analysis.model_dump_json()
        ))

def get_meals_for_date(log_date: str):
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM meals WHERE log_date = ? ORDER BY id ASC", (log_date,))
        return [dict(row) for row in cur.fetchall()]

def delete_meal(meal_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM meals WHERE id = ?", (meal_id,))

init_db()

# ==========================================
# 4. BACKEND API CLIENT INITIALIZATION
# ==========================================

api_key = st.secrets.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash"

if not api_key:
    st.error("⚠️ GEMINI_API_KEY is not set in Streamlit Cloud Secrets. Please add it to your secrets.toml.")
    st.stop()

client = genai.Client(api_key=api_key)

# ==========================================
# 5. BRAND BANNER & INTRO
# ==========================================

st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">NutriSync AI</h1>
    <div class="hero-tagline">Intelligent Multimodal Nutrition & Adaptive Fueling Engine</div>
    <p class="hero-desc">
        Welcome to your adaptive nutrition companion. Log your meals effortlessly by snapping a photo or recording a natural voice recap of your day. 
        NutriSync analyzes macros with Gemini, logs your ledger in real time, and dynamically recalculates your remaining meals to ensure you hit your goals without guesswork.
    </p>
</div>
""", unsafe_allow_html=True)

# Profile & Target Accordion
with st.expander("⚙️ Daily Targets & Lifestyle Preferences", expanded=False):
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        target_cal = st.number_input("Target Calories (kcal)", 1200, 5000, 2200, 50)
    with col_t2:
        target_pro = st.number_input("Protein (g)", 40, 300, 140, 5)
    with col_t3:
        target_carb = st.number_input("Carbs (g)", 50, 600, 220, 10)
    with col_t4:
        target_fat = st.number_input("Fat (g)", 20, 200, 65, 5)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        diet_type = st.selectbox("Dietary Style", ["High-Protein Vegetarian", "Eggetarian", "Omnivore", "Vegan"])
        cuisine_pref = st.text_input("Preferred Cuisine & Staples", "Indian home-style (Dal, Paneer, Dosa, Curd, Whey, Oats)")
    with col_p2:
        tracking_date = st.date_input("Tracking Date", value=date.today())
        schedule_pref = st.text_input("Routine & Activity Notes", "Early morning workout / badminton, active daily routine")

selected_date_str = tracking_date.isoformat()
logged_meals = get_meals_for_date(selected_date_str)

consumed = {
    "calories": sum(m["calories"] for m in logged_meals),
    "protein_g": sum(m["protein_g"] for m in logged_meals),
    "carbs_g": sum(m["carbs_g"] for m in logged_meals),
    "fat_g": sum(m["fat_g"] for m in logged_meals),
}

# ==========================================
# 6. HIGH-CONTRAST METRIC DASHBOARD
# ==========================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    cal_rem = target_cal - consumed["calories"]
    cal_status = f"{abs(cal_rem)} left" if cal_rem >= 0 else f"{abs(cal_rem)} over"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Calories</div>
        <div class="metric-val" style="color: #38bdf8 !important;">{consumed['calories']} <span style="font-size:0.9rem; font-weight:400; color:#94a3b8;">/ {target_cal} kcal</span></div>
        <div class="metric-sub">{cal_status}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    pro_rem = target_pro - consumed["protein_g"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Protein</div>
        <div class="metric-val" style="color: #34d399 !important;">{consumed['protein_g']:.1f}g <span style="font-size:0.9rem; font-weight:400; color:#94a3b8;">/ {target_pro}g</span></div>
        <div class="metric-sub">{max(0.0, pro_rem):.1f}g remaining</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    carb_rem = target_carb - consumed["carbs_g"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Carbohydrates</div>
        <div class="metric-val" style="color: #fbbf24 !important;">{consumed['carbs_g']:.1f}g <span style="font-size:0.9rem; font-weight:400; color:#94a3b8;">/ {target_carb}g</span></div>
        <div class="metric-sub">{max(0.0, carb_rem):.1f}g remaining</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    fat_rem = target_fat - consumed["fat_g"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Healthy Fats</div>
        <div class="metric-val" style="color: #f472b6 !important;">{consumed['fat_g']:.1f}g <span style="font-size:0.9rem; font-weight:400; color:#94a3b8;">/ {target_fat}g</span></div>
        <div class="metric-sub">{max(0.0, fat_rem):.1f}g remaining</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 7. MAIN FUNCTIONAL TABS
# ==========================================

tab_snap, tab_voice, tab_plan, tab_log = st.tabs([
    "📸 Snap Meal",
    "🎙️ Voice Journal",
    "🧠 Adaptive Plan",
    "📋 Today's Ledger"
])

# --- TAB 1: SNAP MEAL ---
with tab_snap:
    st.markdown("#### Upload Plate Photo")
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        slot = st.selectbox("Meal Time Slot", ["Breakfast", "Lunch", "Pre-Workout Snack", "Post-Workout", "Dinner", "Snack"])
        snap_file = st.file_uploader("Take or upload photo", type=["jpg", "jpeg", "png", "webp"], key="photo_picker")
        snap_notes = st.text_input("Additional notes (oils, condiments, serving counts)", placeholder="e.g. 5 pieces appe cooked in 1 tsp ghee with coconut chutney")
        snap_btn = st.button("🔍 Analyze Plate with Gemini Vision", use_container_width=True)

    with c_right:
        if snap_file:
            img = Image.open(snap_file)
            st.image(img, caption="Plate Preview", use_container_width=True)
