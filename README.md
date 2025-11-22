[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/A2-80Pii)


CardioNova – Heart Disease Risk & Doctor Dashboard
==================================================

CardioNova is a full‑stack heart health decision‑support app. It lets clinicians run heart‑disease risk assessments, track a patient’s progress over time, and (for doctors) manage patients through a dedicated dashboard.

The project is split into:

- **backend/** – Flask API, ML models, MongoDB persistence
- **frontend/** – React single‑page app (dashboard UI)


Features
--------

- **Heart risk prediction** using pre‑trained ML models (Logistic Regression, Random Forest, XGBoost ensemble).
- **Rich input form** with clinical features and lifestyle/demographic context.
- **Prediction history & Progress Tracker** with trend charts for:
  - Overall risk score
  - Blood pressure (trestbps)
  - Cholesterol (chol)
  - Other key vitals
- **Doctor–patient profiles**:
  - Per‑doctor patient list
  - Patient profile page with timelines, stats, and past assessments
- **PDF export** of assessments including lifestyle summary and doctor name.
- **Authentication & roles** with JWT (e.g. Doctor), stored in MongoDB.


Tech Stack
---------

- **Backend**
  - Python, Flask
  - MongoDB (pymongo)
  - Joblib‑serialized ML models (`backend/models/*.joblib`)
  - JWT auth
- **Frontend**
  - React + React Router
  - Axios for API calls
  - Recharts for charts
  - jsPDF for PDF generation


Backend Setup
-------------

1. **Create / activate environment** (conda or venv), then install deps:

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Environment variables**

   The backend reads settings from `config.py`, which in turn can use a local `.env` file.

   Create `backend/.env` (optional but recommended):

   ```env
   MONGO_URI=mongodb+srv://<user>:<password>@<cluster>/<db>?retryWrites=true&w=majority
   JWT_SECRET=your_super_secret_key
   GEMINI_API_KEY=your_optional_gemini_api_key
   ```

   - `MONGO_URI` – MongoDB connection string (Atlas or local).
   - `JWT_SECRET` – secret key for signing JWT tokens.
   - `GEMINI_API_KEY` – optional; used by `utils/gemini_client.py` for richer text outputs.

3. **Run the backend**

   From `backend/`:

   ```bash
   python app.py
   ```

   The API will listen on:

   ```text
   http://localhost:5000
   ```


Frontend Setup
--------------

1. **Install dependencies**

   ```bash
   cd frontend
   npm install
   ```

2. **Configure API base URL**

   The frontend expects the backend at `http://localhost:5000`. This is already hard‑coded in the small API helpers (e.g. `src/api/historyAPI.js`, `src/api/predictAPI.js`). If you change the backend URL/port, update those files.

3. **Run the frontend**

   ```bash
   npm start
   ```

   The app will run on:

   ```text
   http://localhost:3000
   ```


Core Flows
----------

### Authentication

- **Signup** – `/signup` (backend) via the React signup form.
  - Stores `name`, `email`, `password (hashed)`, `role` in MongoDB.
- **Login** – `/login` returns a JWT plus user name and role.
- The frontend stores the token, name, email, and role in `localStorage`.

### Prediction

- Frontend sends a payload to `POST /predict` including:
  - `features` – ML input features
  - `lifestyle` – smoking / diabetes / family history / pregnancy, etc.
  - `patientName` – free‑text patient full name
- Backend:
  - Runs the ensemble ML models.
  - Generates explanations and suggestions.
  - Saves a history record in Mongo (`predictions` collection), tagged with:
    - `userId` (doctor email)
    - `doctorId` (same as user for now)
    - `patientId` (doctor‑scoped ID built from doctor + patientName)
    - `patientName` and all metrics.

### History & Progress Tracker

- `GET /history/<user_email>` returns prediction history for a user (doctor).
- React **History / Progress Tracker** page:
  - Shows a list of assessments and a chart‑based timeline.
  - Allows exporting each record as PDF.
  - Allows deleting specific history items (via `DELETE /history/item/<id>`), which updates both table and charts.

### Doctor–Patient Views

- **Doctor Patients List** – `GET /doctor/patients`
  - Authenticated & role‑guarded (Doctor only).
  - Returns each unique patient for the doctor, with last visit and assessment count.
- **Patient Profile** – `GET /doctor/patient/<patient_id>`
  - Full history for a single patient, including time series for risk, BP, chol, etc.
  - React page renders summary cards, charts, and history table.


Running the Full Stack
----------------------

1. Start MongoDB (Atlas or local) and confirm your `MONGO_URI` is valid.
2. In one terminal:

   ```bash
   cd backend
   python app.py
   ```

3. In another terminal:

   ```bash
   cd frontend
   npm start
   ```

4. Open the app in a browser:

   ```text
   http://localhost:3000
   ```


Project Structure (High‑Level)
------------------------------

```text
backend/
  app.py                 # Flask app entrypoint
  config.py              # MONGO_URI, JWT_SECRET, env loading
  database/
    mongo.py             # MongoDB client + collections
  models/
    preprocessor.joblib
    logistic_model.joblib
    rf_model.joblib
    xgb_model.joblib
  routes/
    auth_routes.py       # signup/login
    predict_routes.py    # /predict, /history, doctor endpoints
  utils/
    hashing.py           # password hashing helpers
    token.py             # JWT encode/decode
    shap_handler.py      # SHAP top‑feature extraction
    gemini_client.py     # optional Gemini API client

frontend/
  src/
    api/                 # axios wrappers
    components/          # shared UI components (forms, auth, etc.)
    pages/               # main pages (Predict, History, Doctor views)
    App.js               # routing
    DashboardLayout.jsx  # main layout + navigation
```


License
-------

This project is for educational and prototype‑level clinical decision support. It is **not** a medical device and must not be used as a substitute for professional medical judgment.

- **Name**: Heart Disease Risk Predictor
- **Description**: Full-stack app (Flask backend + React frontend) that predicts cardiovascular risk using an ensemble of ML models and returns SHAP explanations.

**Repository Layout**
- **`backend/`**: Flask API, models, DB code, utilities
  - `app.py` - Flask application entry
  - `routes/` - `auth_routes.py`, `predict_routes.py`
  - `database/mongo.py` - MongoDB connection
  - `models/` - `preprocessor.joblib`, `logistic_model.joblib`, `rf_model.joblib`, `xgb_model.joblib`
  - `requirements.txt` - Python deps
- **`frontend/`**: React SPA
  - `src/api/` - `authAPI.js`, `predictAPI.js`
  - `src/components/` - UI components (Login, Signup, PredictForm, ResultsCard)

**Requirements**
- Python 3.9+ (3.10/3.11 recommended)
- Node.js 16+ (Node 18+ recommended)
- MongoDB Atlas or local MongoDB for development

**Setup — Backend (Windows `cmd`)**
1. Create and activate a virtual environment (recommended):
```
cd "c:\hackathon\Heart Disease\backend"
python -m venv .venv
.venv\Scripts\activate
```
2. Install Python dependencies:
```
pip install -r requirements.txt
```
3. Configure environment variables: copy the example and edit
```
copy .env.example .env
notepad .env
```
Set `MONGO_URI` (Atlas SRV string) and `JWT_SECRET` in `backend/.env` or set them in your shell. Do NOT commit real secrets.

4. Quick connectivity test:
```
python test_mongo.py
```
If you see `MongoDB connection OK`, proceed. If you get an auth error, update `MONGO_URI` (see Troubleshooting).

5. Run backend:
```
python app.py
```

**Setup — Frontend**
1. Install frontend deps and start dev server:
```
cd "c:\hackathon\Heart Disease\frontend"
npm install
npm start
```
The frontend runs on port 3000 and calls the backend at `http://localhost:5000` by default.

**Environment variables**
- `backend/.env` (copied from `.env.example`):
  - `MONGO_URI` - MongoDB connection string (SRV form recommended)
  - `JWT_SECRET` - secret for JWT generation

**API Endpoints (backend)**
- `GET /` — health/landing message
- `POST /signup` — register user
  - Body JSON: `{ "name": "...", "email": "...", "password": "..." }`
- `POST /login` — login
  - Body JSON: `{ "email": "...", "password": "..." }`
  - Returns: `{ "token": "..." }`
- `POST /predict` — predict risk
  - Body JSON: features from the form (see `frontend/src/components/PredictForm.jsx`). Example:
    ```json
    {
      "age": 63,
      "sex": 1,
      "cp": 3,
      "trestbps": 145,
      "chol": 233,
      "fbs": 1,
      "restecg": 0,
      "thalach": 150,
      "exang": 0,
      "oldpeak": 2.3,
      "slope": 0,
      "ca": 0,
      "thal": 1
    }
    ```
  - Returns: `{ "risk_score": <float>, "risk_level": "Low|Moderate|High", "top_features": [...], "shap_error"?: "..." }`

**Models**
- Stored in `backend/models/` as joblib files. These must exist to serve `/predict`. If missing, retrain or obtain the model artifacts.

**Troubleshooting**
- MongoDB auth failure (common): error like `pymongo.errors.OperationFailure: bad auth`:
  - Verify user & password in Atlas (Database Access) and copy the SRV string from Atlas Connect.
  - Ensure your IP is allowed in Atlas Network Access (or add 0.0.0.0/0 for dev).
  - If password contains special characters, URL-encode them.
- JWT errors (`module 'jwt' has no attribute 'encode'`):
  - Make sure `PyJWT` is installed (`pip install pyjwt`) and there is no conflicting package named `jwt`.
  - If you previously installed a conflicting `jwt` package, uninstall with `pip uninstall jwt` then reinstall `pyjwt`.
- Preprocessor missing columns on `/predict`: the backend attempts to derive `age_group`, `bp_cat`, `chol_cat`, and `thalch` from inputs. If you still see `columns are missing` errors, inspect `backend/models/preprocessor.joblib` to match the exact feature names used during training.
- SHAP errors/serialization: SHAP is heavy; if shap-related errors appear, the server now returns `shap_error` in the response — check logs for details.

**Development notes**
- Frontend calls are in `frontend/src/api/*.js` and point to `http://localhost:5000`; change if your backend runs elsewhere.
- To build frontend for production:
```
cd frontend
npm run build
```

**Contributing**
- Fork, branch, and open a PR. Keep secrets out of commits; use `.env` for local development.

**License**
- Add a license file if you intend to open-source this project.

---
If you want, I can also:
- Inspect `backend/models/preprocessor.joblib` to list expected input feature names and encoder categories (helps make the `/predict` input mapping exact).
- Add a minimal `Makefile` or npm scripts to streamline dev commands.
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/A2-80Pii)
