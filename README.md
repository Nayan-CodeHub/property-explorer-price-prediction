# 🏠 House Price Prediction using Machine Learning

## 📌 Problem Statement
Predict house prices based on important features such as location, size, number of rooms, and amenities using machine learning techniques.

---

## 📂 Dataset
- Dataset: Housing Price Dataset
- Source: Kaggle / Public Dataset
- Features include:
  - Location
  - Area (sq.ft)
  - Number of bedrooms
  - Number of bathrooms
  - Amenities
  - Furnished status
  - Property type
  - Property condition
  - Distance from railway station
  - Balconies and security/CCTV

---

## 🔄 Project Workflow
1. Data Cleaning and Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Model Training  
5. Model Evaluation and Comparison  
6. Final Model Selection  

---

## 🛠 Tech Stack
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## 🤖 Models Used
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  

---

## 📊 Evaluation Metrics
- R² Score  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

---

## 📈 Results
- Random Forest Regressor performed best  
- R² Score: ~0.87  
- MAE: ~1.2 Lakhs  
- RMSE: ~2.1 Lakhs  

---

## 🚀 How to Run the Project

### 1. Open PowerShell or Command Prompt
Navigate to the project folder:

```powershell
cd "C:\Users\NAYAN-PC\OneDrive\Desktop\house price predector\house-price-prediction-ml-main"
```

### 2. Install dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3. Run the project in terminal mode
```powershell
python app.py
```
This will train the models, compare their performance, and print the best prediction result.

### 4. Run the browser app (recommended)
```powershell
python -m streamlit run app.py --server.port 8501
```
Then open:

```text
http://localhost:8501
```

If port 8501 is already in use, run:

```powershell
python -m streamlit run app.py --server.port 8502
```
Then open:

```text
http://localhost:8502
```

> If your system cannot find the `streamlit` command, use the full Python path:
>
> ```powershell
> & "C:/Users/NAYAN-PC/AppData/Local/Programs/Python/Python313/python.exe" -m streamlit run app.py --server.port 8502
> ```

---

## 🗺️ Map-Based Property Selection
- Open the prediction dashboard after logging in.
- Click the exact property area on the interactive map. The default map opens around Virar.
- Choose the matching locality and enter the flat details.
- Click **Predict Price** to get an estimate for that selected location.
- Use **Open selected point in Google Maps** to inspect the exact point in Google Maps.

The local interactive picker uses OpenStreetMap tiles. A live Google Maps JavaScript map requires a Google Maps API key and billing-enabled Google Cloud project; the Google Maps link works without adding a key to this project.

## 🏠 Prediction Inputs
The dashboard uses practical property details including area, bedrooms, bathrooms, floors, locality, house age, parking, furnished status, property type, condition, distance from station, balconies, amenities, and security/CCTV. The generated demo dataset is synthetic; replace it with verified local property data before using predictions for real purchase decisions.

## 📊 Dashboard Features
- **Price prediction graph:** After an estimate, view how the predicted price changes with area.
- **Feature importance:** Expand the section to see which inputs influenced the trained model most.
- **Reset property form:** Clear the current property inputs and start a new estimate.
- **Prediction history:** View the last 10 estimates during the current login session.
- **Responsive UI:** The layout adapts to desktop and mobile screen widths.

## 🏢 Creator Property Photos
- Register or log in using the username `admin` to access **Creator tools** in the prediction dashboard.
- Upload one or more building photos and enter the locality, building name, available-flat count, and caption.
- Published photos are saved in the `property_photos` folder and their details are stored in `users.db`.
- All users can view the published building gallery for the selected locality before requesting a price estimate....
This is a local creator workflow for the project demo. For production, add role management and cloud image storage instead of relying on the special `admin` username.

## 🚀 Future Improvements
- Hyperparameter tuning using GridSearchCV  
- Feature importance analysis  
- Model deployment using Flask or Streamlit  

---

## 👩‍💻 Developed By
  
Aspiring AI/ML Engineer
http://localhost:8502
