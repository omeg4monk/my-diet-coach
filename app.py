import sqlite3
import json
from datetime import date
from typing import List
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# 1. PAGE CONFIG & MODERN CSS INJECTION
# ==========================================

st.set_page_config(
    page_title="NutriSync AI — Adaptive Diet Coach",
    page_icon="🥗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a modern, sleek mobile-first design
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Default Header & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom Card Containers */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 12px;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Typography & Titles */
    h1, h2, h3 {
        color: #f1f5f9;
        font-weight: 700;
    }
    
    .app-title {
        font-size: 2.2rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 6px;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PYDANTIC SCHEMAS FOR GEMINI OUTPUTS
# ==========================================

class FoodItem(BaseModel):
    name: str = Field(description="Name of the food item or dish identified")
    estimated_grams: int = Field(description="Estimated portion weight in grams")
    calories: int = Field(description="Estimated calories (kcal)")
    protein_g: float = Field(description="Estimated protein in grams")
    carbs_g: float = Field(description="Estimated carbohydrates in grams")
    fat_g: float = Field(description="Estimated fat in grams")
    fiber_g: float = Field(description="Estimated dietary fiber in grams")

class MealAnalysis(BaseModel):
    meal_summary: str = Field(description="Overall title of the meal (e.g., 'Paniyaram with Chutney')")
    items: List[FoodItem] = Field(description="Breakdown of individual items on the plate")
    total_calories: int = Field(description="Sum of calories for the entire meal")
    total_protein_g: float = Field(description="Sum of protein in grams")
    total_carbs_g: float = Field(description="Sum of carbs in grams")
    total_fat_g: float = Field(description="Sum of fat in grams")
    total_fiber_g: float = Field(description="Sum of fiber in grams")
    nutritional_feedback: str = Field(
        description="1-2 sentences of actionable feedback on macro balance and how to adjust the next meal"
    )

class PlannedMeal(BaseModel):
    meal_slot: str = Field(description="e.g., Breakfast, Lunch, Dinner")
    dish_suggestion: str = Field(description="Specific dish with exact portion sizes")
    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float
    why_this_works: str = Field(description="How this fits the remaining macro budget")

class AdaptiveDietPlan(BaseModel):
    strategy_note: str = Field(description="Brief explanation of how the plan adapts to what was eaten today")
    recommended_meals: List[PlannedMeal]

# ==========================================
# 3. LOCAL SQLITE DATABASE LAYER
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
# 4. API KEY CONFIG (HIDDEN FOR END USER)
# ==========================================

# Automatically loads your backend key from Streamlit Cloud Secrets.
# No API key box or sidebar clutter is shown to your end-users.
api_key = st.secrets.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-3.8-flash"

if not api_key:
    st.error("⚠️ System Configuration Error: GEMINI_API_KEY is not set in Streamlit Cloud Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# ==========================================
# 5. HEADER & PROFILE SETTINGS EXPANDER
# ==========================================

st.markdown('<p class="app-title">NutriSync AI</p>', unsafe_allow_html=True)
st.caption("✨ Real-time adaptive nutrition powered by Gemini Vision")

with st.expander("⚙️ Customize Your Daily Goals & Preferences"):
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        target_cal = st.number_input("Target Calories", 1200, 4000, 2200, 50)
        target_pro = st.number_input("Target Protein (g)", 40, 300, 140, 5)
    with col_g2:
        target_carb = st.number_input("Target Carbs (g)", 50, 500, 220, 10)
        target_fat = st.number_input("Target Fat (g)", 30, 200, 65, 5)
    
    diet_type = st.selectbox("Diet Style", ["High-Protein Vegetarian", "Omnivore", "Eggetarian", "Vegan"])
    cuisine_pref = st.text_input("Favorite Staples", value="Home-style Indian, South Indian, Dals, Paneer, Oats")
    selected_date = st.date_input("Tracking Date", value=date.today()).isoformat()

# Fetch current logs & calculate progress
logged_meals = get_meals_for_date(selected_date if 'selected_date' in locals() else date.today().isoformat())
consumed = {
    "calories": sum(m["calories"] for m in logged_meals),
    "protein_g": sum(m["protein_g"] for m in logged_meals),
    "carbs_g": sum(m["carbs_g"] for m in logged_meals),
    "fat_g": sum(m["fat_g"] for m in logged_meals),
}
target_cal = target_cal if 'target_cal' in locals() else 2200
target_pro = target_pro if 'target_pro' in locals() else 140
target_carb = target_carb if 'target_carb' in locals() else 220
target_fat = target_fat if 'target_fat' in locals() else 65

# ==========================================
# 6. DASHBOARD METRIC CARDS
# ==========================================

st.write("")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)

with mcol1:
    cal_left = target_cal - consumed['calories']
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8rem; color: #94a3b8;">CALORIES</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #38bdf8;">{consumed['calories']}</div>
        <div style="font-size: 0.75rem; color: #64748b;">Target: {target_cal}</div>
    </div>
    """, unsafe_allow_html=True)

with mcol2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8rem; color: #94a3b8;">PROTEIN</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #818cf8;">{consumed['protein_g']:.1f}g</div>
        <div style="font-size: 0.75rem; color: #64748b;">Target: {target_pro}g</div>
    </div>
    """, unsafe_allow_html=True)

with mcol3:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8rem; color: #94a3b8;">CARBS</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #34d399;">{consumed['carbs_g']:.1f}g</div>
        <div style="font-size: 0.75rem; color: #64748b;">Target: {target_carb}g</div>
    </div>
    """, unsafe_allow_html=True)

with mcol4:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8rem; color: #94a3b8;">FATS</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #f472b6;">{consumed['fat_g']:.1f}g</div>
        <div style="font-size: 0.75rem; color: #64748b;">Target: {target_fat}g</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 7. MAIN INTERACTIVE TABS
# ==========================================

tab_log, tab_plan, tab_history = st.tabs(["📸 Snap & Log", "🧠 Adaptive Plan", "📋 Meal History"])

# --- TAB 1: SNAP & LOG ---
with tab_log:
    st.markdown("### Upload Meal Photo")
    meal_type = st.selectbox("Meal Time", ["Breakfast", "Lunch", "Pre-Workout Snack", "Post-Workout", "Dinner"])
    
    uploaded_file = st.file_uploader("Take or upload a photo of your plate", type=["jpg", "jpeg", "png", "webp"])
    user_notes = st.text_input("Quick note (optional)", placeholder="e.g., Cooked in 1 tsp oil, included coconut chutney")
    
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

    if st.button("✨ Analyze Plate with Gemini Vision"):
        if not uploaded_file:
            st.warning("Please upload a photo first.")
        else:
            with st.spinner("Gemini is breaking down your ingredients and macros..."):
                img = Image.open(uploaded_file)
                prompt = f"""
                You are an expert clinical nutritionist. Analyze this photo of a {meal_type} meal.
                User notes: "{user_notes if user_notes else 'None'}"
                1. Identify every visible item and portion size in grams.
                2. Calculate calories, protein, carbs, fat, and fiber.
                3. Provide 1-2 sentences of actionable coaching feedback.
                """
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[img, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=MealAnalysis,
                        temperature=0.2,
                    ),
                )
                analysis = MealAnalysis.model_validate_json(response.text)
                save_meal(selected_date.isoformat(), meal_type, analysis)
                st.session_state["last_analysis"] = analysis
                st.success("Meal logged successfully!")
                st.rerun()

    if "last_analysis" in st.session_state:
        res: MealAnalysis = st.session_state["last_analysis"]
        st.markdown(f"""
        <div class="glass-card">
            <h4>🎯 {res.meal_summary} ({res.total_calories} kcal)</h4>
            <p style="color: #38bdf8; font-size: 0.9rem;">💡 {res.nutritional_feedback}</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: ADAPTIVE DIET PLAN ---
with tab_plan:
    st.markdown("### AI Adaptive Meal Rebalancer")
    st.markdown("Generates your remaining meals for today based on what you've already eaten.")
    
    if st.button("⚡ Re-Optimize Remaining Day"):
        with st.spinner("Re-calculating macros and crafting your menu..."):
            remaining = {
                "calories": max(0, target_cal - consumed["calories"]),
                "protein": max(0.0, target_pro - consumed["protein_g"]),
                "carbs": max(0.0, target_carb - consumed["carbs_g"]),
                "fat": max(0.0, target_fat - consumed["fat_g"]),
            }
            eaten_summary = [f"{m['meal_type']}: {m['meal_summary']} ({m['calories']} kcal)" for m in logged_meals]
            
            prompt = f"""
            Build a realistic meal plan for the REMAINING meals today.
            Diet Style: {diet_type} | Staples: {cuisine_pref}
            Remaining Macro Budget: {remaining['calories']} kcal, {remaining['protein']}g protein, {remaining['carbs']}g carbs, {remaining['fat']}g fat.
            Already eaten: {json.dumps(eaten_summary)}
            """
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AdaptiveDietPlan,
                    temperature=0.4,
                ),
            )
            plan = AdaptiveDietPlan.model_validate_json(response.text)
            st.session_state["current_plan"] = plan

    if "current_plan" in st.session_state:
        plan: AdaptiveDietPlan = st.session_state["current_plan"]
        st.info(f"🧭 {plan.strategy_note}")
        for m in plan.recommended_meals:
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b style="color: #38bdf8; font-size: 1.1rem;">{m.meal_slot}: {m.dish_suggestion}</b>
                    <span style="background: #1e293b; padding: 4px 10px; border-radius: 8px; font-size: 0.85rem;">{m.calories} kcal</span>
                </div>
                <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px; margin-bottom: 4px;">P: {m.protein_g}g | C: {m.carbs_g}g | F: {m.fat_g}g</p>
                <p style="font-size: 0.9rem; font-style: italic; color: #cbd5e1;">{m.why_this_works}</p>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 3: MEAL HISTORY ---
with tab_history:
    st.markdown("### Today's Logged Meals")
    if not logged_meals:
        st.info("No meals logged yet today. Snap your first plate in the first tab!")
    else:
        for m in logged_meals:
            st.markdown(f"""
            <div class="glass-card" style="padding: 16px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between;">
                    <b>{m['meal_type']}: {m['meal_summary']}</b>
                    <span style="color: #38bdf8;">{m['calories']} kcal</span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                    Protein: {m['protein_g']}g | Carbs: {m['carbs_g']}g | Fat: {m['fat_g']}g
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🗑️ Delete entry", key=f"del_{m['id']}"):
                delete_meal(m["id"])
                st.rerun()
