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
# 1. PYDANTIC SCHEMAS FOR GEMINI OUTPUTS
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
    meal_summary: str = Field(description="Overall title of the meal (e.g., 'Paniyaram with Coconut Chutney')")
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
    meal_slot: str = Field(description="e.g., Pre-Workout, Breakfast, Lunch, Evening Fuel, Dinner")
    dish_suggestion: str = Field(description="Specific dish with exact portion sizes")
    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float
    why_this_works: str = Field(description="How this fits the remaining macro budget and activity schedule")

class AdaptiveDietPlan(BaseModel):
    strategy_note: str = Field(description="Brief explanation of how the plan adapts to what was already eaten today")
    recommended_meals: List[PlannedMeal]

# ==========================================
# 2. LOCAL SQLITE DATABASE LAYER
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
            log_date,
            meal_type,
            analysis.meal_summary,
            analysis.total_calories,
            analysis.total_protein_g,
            analysis.total_carbs_g,
            analysis.total_fat_g,
            analysis.total_fiber_g,
            analysis.model_dump_json()
        ))

def get_meals_for_date(log_date: str):
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM meals WHERE log_date = ? ORDER BY id ASC", (log_date,))
        return [dict(row) for row in cur.fetchall()]

def delete_meal(meal_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM meals WHERE id = ?", (meal_id,))

# ==========================================
# 3. GEMINI API INTEGRATION
# ==========================================

def analyze_meal_photo(
    client: genai.Client,
    model_name: str,
    image: Image.Image,
    user_notes: str,
    meal_type: str
) -> MealAnalysis:
    prompt = f"""
    You are an expert sports and clinical nutritionist.
    Analyze this photo of a {meal_type} meal.
    User context / notes on portions or ingredients: "{user_notes if user_notes else 'None provided'}"
    
    1. Identify every distinct food item, condiment, and beverage visible.
    2. Estimate realistic portion weights (in grams) using plate geometry and standard serving sizes.
    3. Calculate calories, protein, carbs, fat, and fiber accurately.
    4. Provide concise, constructive feedback on the nutritional quality of the meal.
    """
    response = client.models.generate_content(
        model=model_name,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MealAnalysis,
            temperature=0.2,
        ),
    )
    return MealAnalysis.model_validate_json(response.text)

def generate_adaptive_plan(
    client: genai.Client,
    model_name: str,
    targets: dict,
    consumed: dict,
    logged_meals: list,
    preferences: dict
) -> AdaptiveDietPlan:
    remaining = {
        "calories": max(0, targets["calories"] - consumed["calories"]),
        "protein_g": max(0.0, targets["protein_g"] - consumed["protein_g"]),
        "carbs_g": max(0.0, targets["carbs_g"] - consumed["carbs_g"]),
        "fat_g": max(0.0, targets["fat_g"] - consumed["fat_g"]),
    }
    
    eaten_summary = [
        f"{m['meal_type']}: {m['meal_summary']} ({m['calories']} kcal, {m['protein_g']}g P, {m['carbs_g']}g C, {m['fat_g']}g F)"
        for m in logged_meals
    ]

    prompt = f"""
    You are an adaptive AI nutrition coach. Build a realistic meal plan for the REMAINING meals today
    (or a full-day plan if nothing has been eaten yet).

    USER PROFILE & PREFERENCES:
    - Dietary Preference: {preferences['diet_type']}
    - Preferred Staples / Cuisine: {preferences['cuisine']}
    - Daily Schedule & Training: {preferences['schedule']}

    DAILY TARGETS:
    - Calories: {targets['calories']} kcal | Protein: {targets['protein_g']}g | Carbs: {targets['carbs_g']}g | Fat: {targets['fat_g']}g

    ALREADY EATEN TODAY:
    {json.dumps(eaten_summary, indent=2) if eaten_summary else "Nothing logged yet today."}

    REMAINING MACRO BUDGET FOR TODAY:
    - Calories left: {remaining['calories']} kcal
    - Protein left: {remaining['protein_g']:.1f}g
    - Carbs left: {remaining['carbs_g']:.1f}g
    - Fat left: {remaining['fat_g']:.1f}g

    INSTRUCTIONS:
    1. Only plan the meals that have NOT been eaten yet today (or a full day if nothing is logged).
    2. Re-balance macros intelligently: for example, if earlier meals were heavy on carbs or fats, steer the remaining meals toward lean protein and fiber so the user lands right on their daily target.
    3. Give exact household measurements (e.g., "6 pieces appe/paniyaram", "150g grilled paneer", "2 bowls moong dal").
    """
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AdaptiveDietPlan,
            temperature=0.4,
        ),
    )
    return AdaptiveDietPlan.model_validate_json(response.text)

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Gemini Adaptive Diet Coach", page_icon="🥗", layout="wide")
init_db()

st.title("🥗 Gemini Vision & Adaptive Diet Coach")
st.caption("Upload meal photos to log macros automatically and dynamically re-balance your daily diet plan.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔑 Gemini API Setup")
    # Automatically loads your saved key from Streamlit Cloud so you don't type it on mobile
    default_key = st.secrets.get("GEMINI_API_KEY", "")
    api_key = default_key or st.text_input("Gemini API Key", type="password")
    if default_key:
        st.success("API Key connected automatically ✅")
        
    model_name = st.selectbox(
        "Model",
        options=["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        index=0
    )
    
    st.divider()
    st.header("🎯 Daily Macro Targets")
    target_cal = st.number_input("Daily Calories (kcal)", min_value=1000, max_value=5000, value=2200, step=50)
    target_pro = st.number_input("Protein (g)", min_value=30, max_value=300, value=140, step=5)
    target_carb = st.number_input("Carbs (g)", min_value=50, max_value=600, value=230, step=10)
    target_fat = st.number_input("Fat (g)", min_value=20, max_value=200, value=65, step=5)

    st.divider()
    st.header("🏃‍♂️ Profile & Routine")
    diet_type = st.selectbox("Dietary Style", ["Vegetarian (High Protein)", "Eggetarian", "Omnivore", "Vegan"])
    cuisine_pref = st.text_area(
        "Staples & Cuisine Preferences",
        value="Indian home-style (Appe/Paniyaram, Dosa, Dal, Paneer, Curd, Sprouts, Whey, Oats)"
    )
    schedule_pref = st.text_area(
        "Daily Routine & Workouts",
        value="Early 5:00 AM wake-up, morning pre-workout fuel, active swimming or badminton session, light dinner."
    )

    selected_date = st.date_input("Tracking Date", value=date.today()).isoformat()

# Fetch current day's logs
logged_meals = get_meals_for_date(selected_date)
consumed = {
    "calories": sum(m["calories"] for m in logged_meals),
    "protein_g": sum(m["protein_g"] for m in logged_meals),
    "carbs_g": sum(m["carbs_g"] for m in logged_meals),
    "fat_g": sum(m["fat_g"] for m in logged_meals),
    "fiber_g": sum(m["fiber_g"] for m in logged_meals),
}

# --- Top Macro Dashboard ---
st.subheader(f"📊 Daily Progress ({selected_date})")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Calories", f"{consumed['calories']} / {target_cal} kcal", f"{target_cal - consumed['calories']} left")
    st.progress(min(1.0, consumed["calories"] / target_cal))
with col2:
    st.metric("Protein", f"{consumed['protein_g']:.1f} / {target_pro} g", f"{target_pro - consumed['protein_g']:.1f}g left")
    st.progress(min(1.0, consumed["protein_g"] / target_pro))
with col3:
    st.metric("Carbs", f"{consumed['carbs_g']:.1f} / {target_carb} g", f"{target_carb - consumed['carbs_g']:.1f}g left")
    st.progress(min(1.0, consumed["carbs_g"] / target_carb))
with col4:
    st.metric("Fats", f"{consumed['fat_g']:.1f} / {target_fat} g", f"{target_fat - consumed['fat_g']:.1f}g left")
    st.progress(min(1.0, consumed["fat_g"] / target_fat))

st.divider()

# --- Main Tabs ---
tab_log, tab_plan, tab_history = st.tabs([
    "📸 1. Log Meal via Photo",
    "🧠 2. Dynamic Diet Planner",
    "📋 3. Today's Meal Ledger"
])

# TAB 1: PHOTO UPLOAD & VISION ANALYSIS
with tab_log:
    left_col, right_col = st.columns([1, 1])
    with left_col:
        meal_type = st.selectbox(
            "Meal Slot",
            ["Pre-Workout", "Breakfast", "Lunch", "Evening Snack / Post-Sport", "Dinner"]
        )
        uploaded_file = st.file_uploader("Upload a photo of your meal", type=["jpg", "jpeg", "png", "webp"])
        user_notes = st.text_input(
            "Optional Context (improves accuracy for hidden ingredients)",
            placeholder="e.g., Cooked in 1 tsp ghee, 6 pieces of rava appe with peanut chutney"
        )
        analyze_btn = st.button("🔍 Analyze & Log Meal", type="primary", use_container_width=True)

    with right_col:
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Meal", use_container_width=True)

    if analyze_btn:
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar first.")
        elif not uploaded_file:
            st.warning("Please upload a meal photo first.")
        else:
            with st.spinner("Gemini Vision is estimating portion weights and macros..."):
                client = genai.Client(api_key=api_key)
                analysis = analyze_meal_photo(client, model_name, img, user_notes, meal_type)
                save_meal(selected_date, meal_type, analysis)
                st.session_state["last_analysis"] = analysis
                st.rerun()

    if "last_analysis" in st.session_state:
        res: MealAnalysis = st.session_state["last_analysis"]
        st.success(f"Logged: **{res.meal_summary}** ({res.total_calories} kcal)")
        st.info(f"💡 **Nutritionist Note:** {res.nutritional_feedback}")
        st.dataframe(
            [item.model_dump() for item in res.items],
            use_container_width=True,
            hide_index=True
        )

# TAB 2: DYNAMIC DIET PLANNER
with tab_plan:
    st.markdown(
        "Generate your full day's diet or **re-calculate your remaining meals** based on the photos you've already logged today."
    )
    if st.button("⚡ Generate / Re-Balance Diet Plan for Today", type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        else:
            with st.spinner("Re-optimizing your remaining meals to hit your macro targets..."):
                client = genai.Client(api_key=api_key)
                targets_dict = {
                    "calories": target_cal,
                    "protein_g": target_pro,
                    "carbs_g": target_carb,
                    "fat_g": target_fat,
                }
                prefs_dict = {
                    "diet_type": diet_type,
                    "cuisine": cuisine_pref,
                    "schedule": schedule_pref,
                }
                plan = generate_adaptive_plan(
                    client, model_name, targets_dict, consumed, logged_meals, prefs_dict
                )
                st.session_state["current_plan"] = plan

    if "current_plan" in st.session_state:
        plan: AdaptiveDietPlan = st.session_state["current_plan"]
        st.info(f"🧭 **Adjustment Strategy:** {plan.strategy_note}")
        for m in plan.recommended_meals:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"#### {m.meal_slot}: {m.dish_suggestion}")
                c1.caption(f"*{m.why_this_works}*")
                c2.markdown(
                    f"**{m.calories} kcal**  \n"
                    f"P: `{m.protein_g}g` | C: `{m.carbs_g}g` | F: `{m.fat_g}g`"
                )

# TAB 3: LOGGED MEALS LEDGER
with tab_history:
    if not logged_meals:
        st.write("No meals logged for this date yet.")
    else:
        for m in logged_meals:
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 2, 1])
                c1.markdown(f"**{m['meal_type']}** — {m['meal_summary']}")
                c2.markdown(
                    f"`{m['calories']} kcal` | P: `{m['protein_g']}g` | "
                    f"C: `{m['carbs_g']}g` | F: `{m['fat_g']}g` | Fiber: `{m['fiber_g']}g`"
                )
                if c3.button("🗑️ Delete", key=f"del_{m['id']}"):
                    delete_meal(m["id"])
                    st.rerun()
