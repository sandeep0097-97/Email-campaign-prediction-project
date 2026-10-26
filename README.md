# InspireDirect: Email Campaign Effectiveness Prediction

Predict whether a recipient will **ignore (0)**, **read (1)** or **acknowledge (2)** a marketing email, then serve the model as a live web app on Vercel.

**Final model:** tuned Random Forest · accuracy 68.8% · macro F1 0.45 · catches 40% of reads and 38% of acknowledgements (random guessing ≈ 0.33 macro F1).

---

## 1. What is in this project

```
email-campaign-predictor/
├── api/                      ← everything Vercel runs
│   ├── index.py              Flask app: web page + /api/predict
│   ├── features.py           turns inputs into model features (shared with training)
│   ├── model.joblib          the trained model (18 MB)
│   └── templates/index.html  the web page
├── src/train_model.py        full pipeline: EDA → cleaning → models → tuning → save
├── notebooks/                same pipeline as a Jupyter notebook, already run
├── data/                     the dataset (CSV)
├── reports/                  charts (figures/) and metrics.json
├── presentation/             capstone slide deck (.pptx, with speaker notes)
├── requirements.txt          packages for the web app (Vercel installs these)
├── requirements-train.txt    packages for training (pandas, seaborn, etc.)
├── vercel.json               tells Vercel to send every request to api/index.py
├── .vercelignore             keeps data, notebooks and slides out of the deployment
└── .python-version           Python 3.12
```

**Key idea to teach:** `api/features.py` is used by both training and the web app. That guarantees the model sees live inputs in exactly the same format it was trained on.

---

## 2. Teaching plan (5 sessions, about 1.5 hours each)

| Session | Topic | Files | Student should be able to explain |
|---|---|---|---|
| 1 | Business problem and data | notebook sections 1–2 | The 3 classes; why 80% "Ignored" makes accuracy misleading |
| 2 | EDA and cleaning | notebook section 2–3, `features.py` | Campaign type is the strongest signal; how missing values and location are handled |
| 3 | Modelling | notebook sections 4–5 | Why `class_weight="balanced"`; what macro F1 is; train/test split; cross-validation |
| 4 | Evaluation and insights | notebook sections 6–8 | Reading a confusion matrix; permutation importance; business recommendations |
| 5 | Deployment and presentation | `api/`, `vercel.json`, slides | How a request flows: browser → Vercel → Flask → model → JSON |

**Practice questions for the viva**
1. Why not use accuracy as the main metric here?
2. What does `class_weight="balanced"` change during training?
3. Why is the imputer inside the Pipeline instead of filling the whole dataset first? *(avoids leaking test-set information)*
4. Why did we stratify the train/test split?
5. Random Forest scored 0.45 macro F1. Is that good? *(compare with ≈0.33 for random guessing; explain the data's limits)*
6. Why must `scikit-learn` in `requirements.txt` match the version used for training?

---

## 3. Run it on your computer

You need **Python 3.12** and **Git**. Open a terminal inside the `email-campaign-predictor` folder.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate

# 2. Install packages
pip install -r requirements-train.txt

# 3. (Optional) Retrain the model, takes about 4-5 minutes
python src/train_model.py

# 4. Start the web app
python api/index.py
```

Open **http://127.0.0.1:5000**, click "Sample: read", and you should see a prediction.

> The model is already trained and saved in `api/model.joblib`, so step 3 is only needed if you change the code.

**Google Colab instead?** Upload the `notebooks/` notebook and the CSV, put the CSV in a `data/` folder one level up, or change `DATA_PATH` in the first code cell to point at the uploaded file.

---

## 4. Deploy on Vercel (free)

### Step 1: Put the project on GitHub
1. Create a free account at **github.com**.
2. Click **New repository**, name it `email-campaign-predictor`, leave it empty (no README), and create it.
3. In the terminal, inside the project folder:

```bash
git init
git add .
git commit -m "Email campaign effectiveness predictor"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/email-campaign-predictor.git
git push -u origin main
```

(The model file is 18 MB, well under GitHub's 100 MB limit.)

### Step 2: Import it into Vercel
1. Go to **vercel.com** and sign up with your GitHub account.
2. Click **Add New → Project**, then **Import** next to `email-campaign-predictor`.
3. Leave the settings as they are (Vercel detects Python/Flask; no build command is needed).
4. Click **Deploy** and wait 1–3 minutes.

### Step 3: Test it
- Open the link Vercel gives you (like `https://email-campaign-predictor-xxxx.vercel.app`).
- Visit `/api/health` on that link; it should show `{"status": "ok"}`.
- Click a sample on the page. The **first** prediction can take a few seconds while the server wakes up and loads the model; later ones are fast.
- Put the link on slide 13 of the presentation.

### Updating later
Change code → `git add . && git commit -m "message" && git push`. Vercel redeploys automatically.

### Alternative: deploy from the terminal
```bash
npm install -g vercel
vercel          # first time: answer the setup questions
vercel --prod   # publish to the production link
```

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| **500 error** on predict, logs mention `InconsistentVersionWarning` or unpickling | The model was trained with a different scikit-learn version. Make `requirements.txt` match your local version (`pip show scikit-learn`), or retrain with the pinned version. |
| **404 Not Found** on the home page | Check `vercel.json` is in the project root and was pushed to GitHub. |
| Page loads but every route returns the home page, or predict gives 405 | In Vercel → Project → Settings, set Framework Preset to **Flask**, delete `vercel.json`, push, and redeploy (Vercel finds `api/index.py` automatically). |
| **Bundle too large** | Make sure `.vercelignore` was pushed; it keeps the 5 MB dataset, notebooks and slides out of the deployment. |
| `ModuleNotFoundError: features` | Run the app from the project root: `python api/index.py`. |
| See what went wrong on Vercel | Project → **Logs** (or **Deployments → the deployment → Functions/Logs**). |

---

## 6. API reference

`POST /api/predict` with JSON:

```json
{
  "Email_Type": 1, "Subject_Hotness_Score": 4.9, "Email_Source_Type": 2,
  "Customer_Location": "E", "Email_Campaign_Type": 2,
  "Total_Past_Communications": 11, "Time_Email_sent_Category": 2,
  "Word_Count": 67, "Total_Links": 11, "Total_Images": 4
}
```

Response:

```json
{"prediction": 1, "label": "Read",
 "probabilities": {"Ignored": 0.15, "Read": 0.85, "Acknowledged": 0.0}}
```

`Total_Past_Communications`, `Total_Links`, `Total_Images` and `Customer_Location` may be left blank; the model fills them in the same way as during training.
