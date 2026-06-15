import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras


st.set_page_config(
    page_title="Forest Cover Type Classification",
    page_icon="🌲",
    layout="wide"
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "forest_cover_ann.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()
scaler = load_scaler()


class_names = {
    0: "Spruce/Fir",
    1: "Lodgepole Pine",
    2: "Ponderosa Pine",
    3: "Cottonwood/Willow",
    4: "Aspen",
    5: "Douglas-fir",
    6: "Krummholz"
}


st.title("🌲 Forest Cover Type Classification using ANN")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    [
        "📖 About Project",
        "🌳 Forest Types",
        "🔮 Prediction"
    ]
)


with tab1:

    st.header("Project Overview")

    st.write("""
    This project uses an Artificial Neural Network (ANN)
    to classify forest cover types based on cartographic variables.

    The dataset originates from the Roosevelt National Forest
    in Colorado and contains environmental and geographical features
    such as elevation, slope, hillshade values, soil types,
    and wilderness area indicators.
    """)

    st.subheader("Model Architecture")

    st.code("""
Input Layer (54 Features)

↓ Dense(128, ReLU)
↓ Dropout(0.3)

↓ Dense(64, ReLU)
↓ Dropout(0.3)

↓ Dense(32, ReLU)

↓ Dense(7, Softmax)

Output: Forest Cover Type
""")

    st.subheader("Features Used")

    st.write("""
    • Elevation

    • Aspect

    • Slope

    • Distance to Hydrology

    • Distance to Roadways

    • Hillshade Measurements

    • Distance to Fire Points

    • Wilderness Area Indicators

    • Soil Type Indicators
    """)


with tab2:

    st.header("Forest Cover Classes")

    classes_df = pd.DataFrame(
        {
            "Class": [1, 2, 3, 4, 5, 6, 7],
            "Forest Type": [
                "Spruce/Fir",
                "Lodgepole Pine",
                "Ponderosa Pine",
                "Cottonwood/Willow",
                "Aspen",
                "Douglas-fir",
                "Krummholz"
            ]
        }
    )

    st.dataframe(
        classes_df,
        use_container_width=True
    )

    st.info(
        "The model predicts one of these seven forest cover categories."
    )
    st.subheader("Model Performance - Confusion Matrix")

with tab3:

    st.header("🔮 Forest Cover Prediction")

    st.write(
        """
        Generate a realistic forest location sample and adjust
        environmental factors to predict the most likely forest cover type.
        """
    )

    DATASET_PATH = os.path.join(
        BASE_DIR,
        "data",
        "covtype.csv"
    )

    df = pd.read_csv(DATASET_PATH)

    if "sample" not in st.session_state:

        st.session_state.sample = (
            df.drop("Cover_Type", axis=1)
            .sample(1)
            .iloc[0]
            .copy()
        )

    col1, col2 = st.columns([1, 1])

    with col1:

        if st.button(
            "🎲 Generate Random Forest Sample",
            use_container_width=True
        ):

            st.session_state.sample = (
                df.drop("Cover_Type", axis=1)
                .sample(1)
                .iloc[0]
                .copy()
            )

    sample = st.session_state.sample.copy()

    st.subheader("Environmental Characteristics")

    c1, c2, c3 = st.columns(3)

    with c1:

        sample["Elevation"] = st.number_input(
            "Elevation",
            value=float(sample["Elevation"])
        )

        sample["Aspect"] = st.number_input(
            "Aspect",
            value=float(sample["Aspect"])
        )

        sample["Slope"] = st.number_input(
            "Slope",
            value=float(sample["Slope"])
        )

        sample["Horizontal_Distance_To_Hydrology"] = st.number_input(
            "Distance To Hydrology",
            value=float(sample["Horizontal_Distance_To_Hydrology"])
        )

    with c2:

        sample["Vertical_Distance_To_Hydrology"] = st.number_input(
            "Vertical Distance To Hydrology",
            value=float(sample["Vertical_Distance_To_Hydrology"])
        )

        sample["Horizontal_Distance_To_Roadways"] = st.number_input(
            "Distance To Roadways",
            value=float(sample["Horizontal_Distance_To_Roadways"])
        )

        sample["Horizontal_Distance_To_Fire_Points"] = st.number_input(
            "Distance To Fire Points",
            value=float(sample["Horizontal_Distance_To_Fire_Points"])
        )

    with c3:

        sample["Hillshade_9am"] = st.number_input(
            "Hillshade 9 AM",
            value=float(sample["Hillshade_9am"])
        )

        sample["Hillshade_Noon"] = st.number_input(
            "Hillshade Noon",
            value=float(sample["Hillshade_Noon"])
        )

        sample["Hillshade_3pm"] = st.number_input(
            "Hillshade 3 PM",
            value=float(sample["Hillshade_3pm"])
        )

    st.markdown("---")

    if st.button(
        "🌲 Predict Forest Cover Type",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [sample]
        )

        scaled_input = scaler.transform(
            input_df
        )

        prediction = model.predict(
            scaled_input,
            verbose=0
        )

        pred_class = np.argmax(
            prediction,
            axis=1
        )[0]

        confidence = float(
            prediction[0][pred_class]
        ) * 100

        st.success(
            f"Predicted Forest Type: "
            f"{class_names[pred_class]}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader(
            "Top 3 Most Likely Forest Types"
        )

        probs = prediction[0]

        top3 = np.argsort(
            probs
        )[::-1][:3]

        result_df = pd.DataFrame(
            {
                "Forest Type": [
                    class_names[i]
                    for i in top3
                ],
                "Probability (%)": [
                    round(
                        probs[i] * 100,
                        2
                    )
                    for i in top3
                ]
            }
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.subheader(
            "Prediction Confidence Distribution"
        )

        chart_df = pd.DataFrame(
            {
                "Forest Type": list(
                    class_names.values()
                ),
                "Probability": probs
            }
        )

        st.bar_chart(
            chart_df.set_index(
                "Forest Type"
            )
        )