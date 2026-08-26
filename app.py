import hashlib
import pickle
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import folium
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from streamlit_folium import st_folium

DATA_PATH = Path("housing_data_v2.csv")
MODEL_PATH = Path("best_model_v2.pkl")
DB_PATH = Path("users.db")
PHOTO_ROOT = Path("property_photos")
LOCATION_OPTIONS = ["Virar", "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune"]

SECURITY_QUESTIONS = [
    "What is the name of your first school?",
    "What was the name of your first pet?",
    "Which city were you born in?",
    "What is your favorite teacher's name?",
]


def hash_value(value: str) -> str:
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            security_question TEXT NOT NULL,
            security_answer_hash TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS property_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            building_name TEXT NOT NULL,
            available_flats INTEGER NOT NULL,
            caption TEXT,
            file_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def register_user(username, password, question, answer):
    username = username.strip()
    password = password.strip()
    answer = answer.strip()

    if not username or not password or not question or not answer:
        return False, "Please fill in all fields."
    if len(password) < 4:
        return False, "Password must be at least 4 characters long."

    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if existing:
            return False, "Username already exists."

        conn.execute(
            "INSERT INTO users (username, password_hash, security_question, security_answer_hash) VALUES (?, ?, ?, ?)",
            (username, hash_value(password), question, hash_value(answer)),
        )
        conn.commit()
        return True, "Registration successful. Please log in."
    except sqlite3.Error:
        return False, "Registration failed. Try again."
    finally:
        conn.close()


def login_user(username, password):
    username = username.strip()
    conn = get_connection()
    row = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()

    if row and row["password_hash"] == hash_value(password):
        return True, "Login successful."
    return False, "Invalid username or password."


def get_user_security_question(username):
    username = username.strip()
    conn = get_connection()
    row = conn.execute(
        "SELECT security_question FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()
    if row:
        return row["security_question"]
    return None


def reset_password(username, answer, new_password):
    username = username.strip()
    answer = answer.strip()
    new_password = new_password.strip()

    if not username or not answer or not new_password:
        return False, "Please fill in all fields."
    if len(new_password) < 4:
        return False, "New password must be at least 4 characters long."

    conn = get_connection()
    row = conn.execute(
        "SELECT security_answer_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if not row:
        conn.close()
        return False, "User not found."

    if row["security_answer_hash"] != hash_value(answer):
        conn.close()
        return False, "Security answer is incorrect."

    conn.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (hash_value(new_password), username),
    )
    conn.commit()
    conn.close()
    return True, "Password reset successful. Please log in again."


def generate_dataset(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    locations = LOCATION_OPTIONS
    location_factor = {
        "Virar": 0.88,
        "Delhi": 1.12,
        "Mumbai": 1.18,
        "Bengaluru": 1.05,
        "Hyderabad": 0.98,
        "Chennai": 1.00,
        "Pune": 1.08,
    }

    n_rows = 2400
    df = pd.DataFrame(
        {
            "location": rng.choice(locations, size=n_rows),
            "area": rng.normal(1700, 420, n_rows).clip(600, 5000),
            "bedrooms": rng.integers(1, 6, n_rows),
            "bathrooms": rng.integers(1, 5, n_rows),
            "age": rng.integers(0, 35, n_rows),
            "parking": rng.integers(0, 3, n_rows),
            "floors": rng.integers(1, 5, n_rows),
            "amenities": rng.uniform(1, 10, n_rows).round(2),
            "furnished": rng.choice(["Unfurnished", "Semi-furnished", "Furnished"], size=n_rows),
            "property_type": rng.choice(["Apartment", "Villa", "Independent House"], size=n_rows, p=[0.65, 0.15, 0.20]),
            "condition": rng.choice(["New", "Good", "Needs Repair"], size=n_rows, p=[0.25, 0.60, 0.15]),
            "distance_station": rng.uniform(0.2, 12, n_rows).round(2),
            "balcony": rng.integers(0, 4, n_rows),
            "security": rng.choice(["Yes", "No"], size=n_rows, p=[0.75, 0.25]),
        }
    )

    df["price_in_lakhs"] = (
        df["area"] * 0.058
        + df["bedrooms"] * 5.2
        + df["bathrooms"] * 6.1
        + df["parking"] * 4.0
        + df["floors"] * 5.5
        + df["amenities"] * 1.8
        - df["age"] * 0.45
        - df["distance_station"] * 0.8
        + df["balcony"] * 2.2
        + df["furnished"].map({"Unfurnished": 0, "Semi-furnished": 8, "Furnished": 15})
        + df["property_type"].map({"Apartment": 0, "Villa": 35, "Independent House": 22})
        + df["condition"].map({"New": 18, "Good": 7, "Needs Repair": -12})
        + df["security"].map({"Yes": 6, "No": 0})
    )
    df["price_in_lakhs"] *= df["location"].map(location_factor)
    df["price_in_lakhs"] += rng.normal(0, 6.5, n_rows)
    df["price_in_lakhs"] = df["price_in_lakhs"].clip(lower=12)
    df.to_csv(path, index=False)
    return df


def build_preprocessor():
    categorical_features = ["location", "furnished", "property_type", "condition", "security"]
    numerical_features = [
        "area",
        "bedrooms",
        "bathrooms",
        "age",
        "parking",
        "floors",
        "amenities",
        "distance_station",
        "balcony",
    ]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )


def make_models():
    preprocessor = build_preprocessor()
    return {
        "Linear Regression": Pipeline(
            steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
        ),
        "Decision Tree Regressor": Pipeline(
            steps=[("preprocessor", preprocessor), ("model", DecisionTreeRegressor(random_state=42))]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(n_estimators=300, random_state=42))]
        ),
    }


def evaluate_models(df: pd.DataFrame):
    X = df.drop(columns=["price_in_lakhs"])
    y = df["price_in_lakhs"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    best_model_name = ""
    best_pipeline = None
    best_score = float("-inf")

    for name, model in make_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        results.append({
            "Model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "Pipeline": model,
        })

        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_pipeline = model

    results.sort(key=lambda item: item["R2"], reverse=True)
    return results, best_model_name, best_pipeline


def train_and_save_model():
    if not DATA_PATH.exists():
        generate_dataset(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    _, _, best_pipeline = evaluate_models(df)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(best_pipeline, file)
    return best_pipeline


def load_model():
    if not MODEL_PATH.exists():
        return train_and_save_model()

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def predict_house_price(model, payload):
    sample = pd.DataFrame([payload])
    prediction = model.predict(sample)[0]
    return float(prediction)

def get_feature_importance(model):
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance_values = np.abs(estimator.coef_)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance_values}
    )
    importance["Feature"] = importance["Feature"].str.replace(
        r"^(cat|num)__", "", regex=True
    )
    return importance.sort_values("Importance", ascending=False).head(10)


def safe_file_stem(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return cleaned or "building"


def save_property_photos(location, building_name, available_flats, caption, uploaded_files):
    PHOTO_ROOT.mkdir(exist_ok=True)
    location_folder = PHOTO_ROOT / safe_file_stem(location.lower())
    location_folder.mkdir(exist_ok=True)
    conn = get_connection()

    try:
        for photo_index, uploaded_file in enumerate(uploaded_files):
            extension = Path(uploaded_file.name).suffix.lower()
            file_name = (
                f"{safe_file_stem(building_name)}_"
                f"{pd.Timestamp.now().value}_{photo_index}{extension}"
            )
            photo_path = location_folder / file_name
            photo_path.write_bytes(uploaded_file.getbuffer())
            conn.execute(
                "INSERT INTO property_photos (location, building_name, available_flats, caption, file_path) VALUES (?, ?, ?, ?, ?)",
                (location, building_name.strip(), int(available_flats), caption.strip(), str(photo_path)),
            )
        conn.commit()
        return True, f"Added {len(uploaded_files)} photo(s) for {building_name}."
    except (OSError, sqlite3.Error):
        conn.rollback()
        return False, "The photos could not be saved. Please try again."
    finally:
        conn.close()


def get_property_photos(location):
    conn = get_connection()
    rows = conn.execute(
        "SELECT building_name, available_flats, caption, file_path FROM property_photos WHERE location = ? ORDER BY created_at DESC",
        (location,),
    ).fetchall()
    conn.close()
    return rows


def show_creator_photo_manager():
    with st.expander("Creator tools: add available properties", expanded=False):
        st.caption("Upload building photos so visitors can inspect homes available in each locality.")
        with st.form("property_photo_form"):
            photo_location = st.selectbox("Property locality", LOCATION_OPTIONS)
            building_name = st.text_input("Building name")
            available_flats = st.number_input("Available flats", min_value=1, max_value=999, value=1, step=1)
            caption = st.text_input("Photo caption", placeholder="2 BHK flats near the main gate")
            uploaded_files = st.file_uploader(
                "Building photos",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
            )
            upload_submitted = st.form_submit_button("Publish photos", use_container_width=True)

        if upload_submitted:
            if not building_name.strip() or not uploaded_files:
                st.error("Add a building name and at least one photo.")
            else:
                ok, message = save_property_photos(
                    photo_location,
                    building_name,
                    available_flats,
                    caption,
                    uploaded_files,
                )
                (st.success if ok else st.error)(message)


def show_property_gallery(location):
    photos = get_property_photos(location)
    st.subheader(f"Available buildings in {location}")
    if not photos:
        st.info("No building photos have been published for this locality yet.")
        return

    for start in range(0, len(photos), 3):
        photo_columns = st.columns(3)
        for column, photo in zip(photo_columns, photos[start:start + 3]):
            with column:
                photo_path = Path(photo["file_path"])
                if photo_path.exists():
                    st.image(str(photo_path), use_container_width=True)
                st.markdown(f"**{photo['building_name']}**")
                st.caption(f"{photo['available_flats']} flat(s) available")
                if photo["caption"]:
                    st.caption(photo["caption"])


def show_location_picker():
    default_location = {"lat": 19.4559, "lon": 72.8114}
    if "selected_location" not in st.session_state:
        st.session_state.selected_location = default_location

    selected = st.session_state.selected_location
    location_map = folium.Map(
        location=[selected["lat"], selected["lon"]],
        zoom_start=13,
        tiles="OpenStreetMap",
    )
    folium.Marker(
        [selected["lat"], selected["lon"]],
        tooltip="Selected property location",
        icon=folium.Icon(color="blue", icon="home"),
    ).add_to(location_map)
    map_state = st_folium(
        location_map,
        width=None,
        height=390,
        key="property_location_map",
    )

    if map_state.get("last_clicked"):
        clicked = map_state["last_clicked"]
        st.session_state.selected_location = {
            "lat": clicked["lat"],
            "lon": clicked["lng"],
        }
        selected = st.session_state.selected_location

    st.caption(
        f"Selected coordinates: {selected['lat']:.5f}, {selected['lon']:.5f}"
    )
    google_maps_url = (
        f"https://www.google.com/maps/search/?api=1&query="
        f"{selected['lat']},{selected['lon']}"
    )
    st.link_button("Open selected point in Google Maps", google_maps_url)
    return selected


st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

with open("style.css", "r", encoding="utf-8") as css_file:
    css = css_file.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "page" not in st.session_state:
    st.session_state.page = "login"
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


def show_login_page():
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("<div class='brand-mark'><span class='brand-icon'>⌂</span> EstateIQ</div>", unsafe_allow_html=True)
        st.markdown("### Welcome back")
        st.markdown("<p class='auth-lead'>Make your next property decision with confidence.</p>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='hero-box'>
                <strong>Smart property valuation</strong> for modern buyers and sellers.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='building-scene'>
                <div class='city-header'>
                    <span class='badge'>Premium homes</span>
                    <span class='mini-stat'>+18.4% this year</span>
                </div>
                <div class='skyline'>
                    <div class='building b1'></div>
                    <div class='building b2'></div>
                    <div class='building b3'></div>
                    <div class='building b4'></div>
                    <div class='building b5'></div>
                    <div class='building b6'></div>
                </div>
                <div class='feature-row'>
                    <div class='feature-card'>
                        <span class='feature-icon'>🏙️</span>
                        <div>
                            <strong>1250+</strong>
                            <small>Properties</small>
                        </div>
                    </div>
                    <div class='feature-card'>
                        <span class='feature-icon'>📈</span>
                        <div>
                            <strong>Smart</strong>
                            <small>Insights</small>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("- Compare multiple models")
        st.markdown("- Fast predictions")
        st.markdown("- Real estate decision support")
    with right:
        st.markdown("<div class='auth-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='form-eyebrow'>MEMBER ACCESS</div>", unsafe_allow_html=True)
        st.markdown("## Sign in to your account")
        st.markdown("<p class='form-note'>Welcome back. Enter your details to continue.</p>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            ok, message = login_user(username, password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.session_state.page = "predict"
                st.success(message)
                st.rerun()
            else:
                st.error(message)

        st.markdown("<div class='secondary-link'>New here?</div>", unsafe_allow_html=True)
        if st.button("Create account", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()

        if st.button("Forgot password?", use_container_width=True):
            st.session_state.page = "reset"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def show_register_page():
    left, right = st.columns([1, 1.1])
    with left:
        st.markdown("<div class='brand-mark'><span class='brand-icon'>⌂</span> EstateIQ</div>", unsafe_allow_html=True)
        st.markdown("## Build your property edge")
        st.markdown(
            "<p class='auth-lead'>Create a private workspace for smarter valuations and better real-estate decisions.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='register-perks'>
                <div><span>01</span><strong>Explore confidently</strong><small>Use data-backed estimates before you make a move.</small></div>
                <div><span>02</span><strong>Compare intelligently</strong><small>See how different models evaluate your property.</small></div>
                <div><span>03</span><strong>Keep it personal</strong><small>Your account and valuation workspace stay private.</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div class='auth-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='form-eyebrow'>GET STARTED</div>", unsafe_allow_html=True)
        st.markdown("## Create your account")
        st.markdown("<p class='form-note'>It only takes a moment to get started.</p>", unsafe_allow_html=True)
        with st.form("register_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            question = st.selectbox("Security question", SECURITY_QUESTIONS)
            answer = st.text_input("Security answer")
            submitted = st.form_submit_button("Create account", use_container_width=True)

        if submitted:
            ok, message = register_user(username, password, question, answer)
            if ok:
                st.success(message)
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error(message)

        st.markdown("<div class='form-footer'>Already have an account?</div>", unsafe_allow_html=True)
        if st.button("Back to login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def show_reset_page():
    st.title("🔁 Reset password")
    with st.form("reset_form"):
        username = st.text_input("Username")
        answer = st.text_input("Security answer")
        new_password = st.text_input("New password", type="password")
        submitted = st.form_submit_button("Reset password", use_container_width=True)

    if submitted:
        ok, message = reset_password(username, answer, new_password)
        if ok:
            st.success(message)
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error(message)

    if st.button("Back to login", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()


def show_prediction_page():
    if st.session_state.current_user:
        st.sidebar.write(f"Signed in as: {st.session_state.current_user}")
        if st.session_state.current_user.strip().lower() == "admin":
            show_creator_photo_manager()
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.page = "login"
            st.rerun()

    st.title("🏠 House Price Prediction")
    st.markdown("Estimate the market value of a house from its key features.")

    if not DATA_PATH.exists():
        generate_dataset(DATA_PATH)

    if not MODEL_PATH.exists():
        train_and_save_model()

    model = load_model()
    df = pd.read_csv(DATA_PATH)
    results, best_model_name, best_pipeline = evaluate_models(df)

    st.subheader("Choose the property location")
    st.markdown("Click the building or point on the map where the flat is located.")
    selected_coordinates = show_location_picker()

    if st.button("Reset property form", use_container_width=True):
        for input_key in [
            "prediction_location",
            "prediction_area",
            "prediction_bedrooms",
            "prediction_bathrooms",
            "prediction_property_type",
            "prediction_furnished",
            "prediction_age",
            "prediction_parking",
            "prediction_floors",
            "prediction_amenities",
            "prediction_condition",
            "prediction_distance_station",
            "prediction_balcony",
            "prediction_security",
        ]:
            st.session_state.pop(input_key, None)
        st.session_state.last_prediction = None
        st.rerun()

    with st.form("house_prediction_form"):
        st.subheader("Enter property details")
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox(
                "Locality",
                LOCATION_OPTIONS,
                key="prediction_location",
            )
            area = st.number_input("Area (sq.ft)", min_value=300, max_value=5000, value=1450, step=10, key="prediction_area")
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=8, value=3, step=1, key="prediction_bedrooms")
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=2, step=1, key="prediction_bathrooms")
            property_type = st.selectbox(
                "Property type",
                ["Apartment", "Villa", "Independent House"],
                key="prediction_property_type",
            )
            furnished = st.selectbox(
                "Furnished status",
                ["Unfurnished", "Semi-furnished", "Furnished"],
                key="prediction_furnished",
            )
        with col2:
            age = st.number_input("Age of property (years)", min_value=0, max_value=50, value=8, step=1, key="prediction_age")
            parking = st.number_input("Parking slots", min_value=0, max_value=3, value=1, step=1, key="prediction_parking")
            floors = st.number_input("Floors", min_value=1, max_value=10, value=2, step=1, key="prediction_floors")
            amenities = st.number_input("Amenities score", min_value=1.0, max_value=10.0, value=7.5, step=0.5, key="prediction_amenities")
            condition = st.selectbox("Property condition", ["New", "Good", "Needs Repair"], key="prediction_condition")
            distance_station = st.number_input(
                "Distance from station (km)",
                min_value=0.1,
                max_value=30.0,
                value=2.0,
                step=0.1,
                key="prediction_distance_station",
            )
            balcony = st.number_input("Balconies", min_value=0, max_value=6, value=1, step=1, key="prediction_balcony")
            security = st.selectbox("Security / CCTV", ["Yes", "No"], key="prediction_security")
        submitted = st.form_submit_button("Predict Price", use_container_width=True)

    if submitted:
        payload = {
            "location": location,
            "area": float(area),
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "age": int(age),
            "parking": int(parking),
            "floors": int(floors),
            "amenities": float(amenities),
            "furnished": furnished,
            "property_type": property_type,
            "condition": condition,
            "distance_station": float(distance_station),
            "balcony": int(balcony),
            "security": security,
        }
        price = predict_house_price(model, payload)
        st.session_state.last_prediction = {"price": price, "payload": payload}
        st.session_state.prediction_history.insert(
            0,
            {
                "Time": pd.Timestamp.now().strftime("%d %b %Y, %I:%M %p"),
                "Locality": location,
                "Property": property_type,
                "Area (sq.ft)": area,
                "Estimated Price (Lakhs)": round(price, 2),
            },
        )
        st.session_state.prediction_history = st.session_state.prediction_history[:10]
        st.metric("Predicted House Price", f"₹ {price:.2f} Lakhs")
        st.info(
            f"Estimate for {location} at "
            f"{selected_coordinates['lat']:.5f}, {selected_coordinates['lon']:.5f}."
        )

    show_property_gallery(location)

    if st.session_state.last_prediction:
        last_payload = st.session_state.last_prediction["payload"]
        graph_rows = []
        for graph_area in range(max(300, int(area) - 500), min(5000, int(area) + 501), 100):
            graph_payload = {**last_payload, "area": graph_area}
            graph_rows.append({"Area (sq.ft)": graph_area, "Estimated Price (Lakhs)": predict_house_price(model, graph_payload)})
        st.subheader("Price prediction graph")
        st.line_chart(pd.DataFrame(graph_rows).set_index("Area (sq.ft)"))

    with st.expander("Feature importance", expanded=False):
        importance = get_feature_importance(best_pipeline)
        if importance.empty:
            st.info("Feature importance is not available for the selected model.")
        else:
            st.bar_chart(importance.set_index("Feature"), y="Importance")

    with st.expander("Prediction history", expanded=False):
        if st.session_state.prediction_history:
            st.dataframe(pd.DataFrame(st.session_state.prediction_history), width="stretch", hide_index=True)
        else:
            st.info("Your prediction history will appear here after your first estimate.")

    st.subheader("Model Evaluation")
    metrics = pd.DataFrame(results)[["Model", "R2", "MAE", "RMSE"]].rename(columns={"R2": "R² Score"}).round(4)
    st.dataframe(metrics, width="stretch")
    st.success(f"Best model selected: {best_model_name}")


if st.session_state.page == "login":
    show_login_page()
elif st.session_state.page == "register":
    show_register_page()
elif st.session_state.page == "reset":
    show_reset_page()
else:
    if st.session_state.logged_in:
        show_prediction_page()
    else:
        st.session_state.page = "login"
        st.rerun()