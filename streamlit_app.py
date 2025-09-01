
import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.joblib")
    le = joblib.load("models/label_encoder.joblib")
    return model, le

model, le = load_artifacts()

st.set_page_config(page_title="Obesity Level Predictor", layout="centered")
st.title("Определение уровня веса/ожирения")
st.caption("Введите данные — модель предскажет категорию (NObeyesdad)")

st.subheader("Антропометрия")
col1, col2, col3 = st.columns(3)
age = col1.number_input("Age (лет)", min_value=10, max_value=90, value=25, step=1)
height_cm = col2.number_input("Height (см)", min_value=140, max_value=210, value=175, step=1)
weight = col3.number_input("Weight (кг)", min_value=35, max_value=200, value=75, step=1)
height = float(height_cm) / 100.0

st.subheader("Привычки и образ жизни")
col4, col5, col6 = st.columns(3)
gender = col4.selectbox("Gender", ["Male","Female"])
family_hist = col5.radio("Family history with overweight", ["yes","no"], horizontal=True)
favc = col6.radio("High-calorie food (FAVC)", ["yes","no"], horizontal=True)

col7, col8, col9 = st.columns(3)
caec = col7.selectbox("Snacks between meals (CAEC)", ["No","Sometimes","Frequently","Always"])
smoke = col8.radio("SMOKE", ["yes","no"], horizontal=True)
scc = col9.radio("Calories monitoring (SCC)", ["yes","no"], horizontal=True)

col10, col11, col12 = st.columns(3)
calc = col10.selectbox("Alcohol (CALC)", ["No","Sometimes","Frequently","Always"])
mtrans = col11.selectbox("Transportation (MTRANS)", ["Walking","Bike","Public_Transportation","Automobile","Motorbike"])

st.subheader("Количественные шкалы")
col13, col14, col15, col16 = st.columns(4)
fcvc = col13.slider("Vegetables frequency (FCVC)", min_value=1, max_value=3, value=3, step=1)
ncp = col14.slider("Main meals per day (NCP)", min_value=1, max_value=4, value=3, step=1)
ch2o = col15.slider("Water per day (CH2O)", min_value=1, max_value=3, value=2, step=1)
faf = col16.slider("Physical activity (FAF)", min_value=0, max_value=3, value=1, step=1)

tue = st.slider("Time using technology (TUE)", min_value=0, max_value=2, value=1, step=1)

if st.button("Предсказать уровень"):
    record = {
        "Age": int(age),
        "Height": float(height),
        "Weight": float(weight),
        "FCVC": int(fcvc),
        "NCP": int(ncp),
        "CH2O": int(ch2o),
        "FAF": int(faf),
        "TUE": int(tue),
        "Gender": gender,
        "family_history_with_overweight": family_hist,
        "FAVC": favc,
        "CAEC": caec,
        "SMOKE": smoke,
        "SCC": scc,
        "CALC": calc,
        "MTRANS": mtrans
    }
    df_one = pd.DataFrame([record])
    pred_num = model.predict(df_one)[0]
    proba = model.predict_proba(df_one)[0]
    pred_label = le.inverse_transform([pred_num])[0]

    st.subheader("Результат")
    left, right = st.columns([1,1])

    with left:
        st.success(f"Предсказанная категория: **{pred_label}**")
        st.caption("Вероятности по классам:")
        prob_df = pd.DataFrame({"class": le.classes_, "probability": proba})
        prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True, height=260)

    with right:
        order = [
            "Insufficient_Weight",
            "Normal_Weight",
            "Overweight_Level_I",
            "Overweight_Level_II",
            "Obesity_Type_I",
            "Obesity_Type_II",
            "Obesity_Type_III"
        ]
        desc = {
            "Insufficient_Weight": "Недостаточный вес",
            "Normal_Weight": "Нормальный вес",
            "Overweight_Level_I": "Избыточный вес (уровень I)",
            "Overweight_Level_II": "Избыточный вес (уровень II)",
            "Obesity_Type_I": "Ожирение I степени",
            "Obesity_Type_II": "Ожирение II степени",
            "Obesity_Type_III": "Ожирение III степени"
        }
        ref_df = pd.DataFrame({
            "Класс": order,
            "Описание": [desc[c] for c in order]
        })
        st.markdown("**Справочная шкала уровней**")
        st.dataframe(ref_df, use_container_width=True, height=260)
