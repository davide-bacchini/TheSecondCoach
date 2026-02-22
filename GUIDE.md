# The Next Frontier of Football Analytics: Measuring Decision Quality
**An overview of the ML-Powered Football Analytics Platform**

---

## Executive Summary

In modern elite football, the margin between championship titles and early elimination is razor-thin. While traditional analytics have mastered the art of measuring *execution*—pass completion rates, Expected Goals (xG), and distance covered—they consistently fail to measure the most critical, yet elusive, component of elite performance: **Decision Quality.** 

Traditional metrics tell us what happened; they do not tell us *what should have happened*. Was a completed sideways pass actually the best option, or did the player ignore a line-breaking pass that would have resulted in a high-probability scoring chance? 

This platform bridges that critical gap. Built on top of StatsBomb's elite 360° freeze-frame tracking data, we have developed a machine-learning-powered engine capable of evaluating every on-ball action not just by its outcome, but against the universe of *alternative actions the player could have taken*. By quantifying the opportunity cost of every decision on the pitch, we empower managers, sporting directors, and scouts to evaluate players on their true footballing intelligence.

---

## The Business Problem: The Limits of Traditional Analytics

Football is a game of continuous, complex spatial geometry. When a midfielder receives the ball under pressure, they must process the positions of 21 other players, their momentum, and the trajectory of the ball in fractions of a second.

Current analytics frameworks (like Expected Goals or simple Possession Value models) are fundamentally biased toward safe possession. They reward players who complete 95% of their passes, even if those passes consistently move the team backward. They penalize creative players who attempt high-risk, high-reward actions. 

From a managerial perspective, this creates a blind spot. A club might invest €50m in a central midfielder with outstanding passing statistics, only to realize his risk-averse decision-making stifles the team's attacking transitions. Coaches lack objective tools to show a player: *"You chose Option A, resulting in a low-threat outcome. Option B was available and carried a 4x higher probability of scoring."*

---

## The Methodology: How We Measure the Unseen

To solve this, we engineered a pipeline that reconstructs the geometric reality of the pitch at the exact moment a player makes a decision, generates synthetic alternative realities, and scores them using a trained predictive model. 

### 1. The Threat Baseline: Expected Threat (xT)
We first establish a common currency for value: **Expected Threat (xT)**. By analyzing thousands of historical shot-ending sequences, we calculate the underlying probability of a team scoring within the next 5 actions from any specific coordinate on the pitch (a 16×12 spatial grid). Moving the ball from a low-xT zone (e.g., your own corner flag) to a high-xT zone (e.g., the opponent's penalty box) generates positive **ΔxT**.

### 2. The 360° Context: Spatial Pressure
Using StatsBomb 360° data, we extract the precise coordinates of visible teammates and opponents at the exact moment of the action. We engineer complex spatial features:
- How many opponents are within a 5m/10m radius?
- How many clear passing lanes exist (lanes not blocked by an opponent's geometric shadow)?
- How many teammates are available in advanced, open space?

### 3. Alternative Generation: Building the "Multiverse"
For every pass, carry, or shot taken, our engine pauses time and generates the options the player *didn't* take:
- Passes to every visible teammate.
- Carries into space across 8 directional vectors.
- A shot on goal (if within range).

### 4. Machine Learning Inference: Scoring the Alternatives
We pass every generated alternative (plus the actual action taken) through our trained Machine Learning model. The model predicts the expected ΔxT of each action based on the spatial constraints. The engine ranks these options and identifies the mathematically optimal decision. 

---

## Why Machine Learning?

Why not use a simple rules-based system (e.g., "always pass to the player closest to the goal")? 

Because football is highly non-linear. A pass to a striker on the edge of the box might seem optimal, but if the passing lane is heavily congested and the passer is under acute pressure, that pass represents a near-certain turnover. 

We deployed an **XGBoost Regressor** (eXtreme Gradient Boosting) for this task. XGBoost thrives on complex, tabular data with non-linear relationships and intricate feature interactions. 

### Model Architecture & Training Strategy
- **Features:** 28 engineered features encompassing State (pitch coordinates, score differential), Geometry (target coordinates, action vectors), Context (opponent density, passing lanes), and Categorical markers (player roles, action types).
- **Target Variable:** The *realized ΔxT* of historical actions. If an action led to an immediate turnover, the target value is heavily penalized. If it progressed play, it receives the destination's xT value.
- **Robust Cross-Validation:** We utilized `GroupKFold` cross-validation grouped by `match_id`. This prevents the model from memorizing the specific spatial dynamics of a single game, ensuring the model generalizes to completely unseen matches and tactical setups.

The result is a model that understands spatial trade-offs. It learns that carrying the ball into space is valuable, but carrying it into a cluster of 3 defenders is disastrous. It learns that long, sweeping passes are high-reward, but penalizes them if the target is heavily marked.

---

## Project Directory Structure

Our architecture is strictly modular and designed for hackathon evaluation, cleanly separating raw data, preprocessing (ETL), model training, trained artifacts, and the presentation layer. Below is the absolute directory structure.

```text
open-data/
├── data/
│   ├── competitions.json
│   ├── events/
│   ├── matches/
│   └── three-sixty/
│
├── pipeline/
│   ├── compute_metrics.py
│   ├── build_training_data.py
│   └── train_model.py
│
├── models/
│   ├── vaep_xgb.json
│   ├── xt_grid.csv
│   ├── cv_results.csv
│   └── feature_importance.csv
│
├── mvp/
│   ├── analyze_match.py
│   ├── dashboard_server.py
│   └── frontend/
│       ├── index.html
│       ├── style.css
│       └── app.js
│
├── GUIDE.md
└── LICENSE.pdf
```

---

## Technical Deep Dive: Explaining the Files

The following sections provide a precise, step-by-step breakdown of every custom engineered file in our pipeline and the exact operations they perform to surface decision quality metrics.

### 1. `pipeline/compute_metrics.py` (The Spatial ETL Framework)
* **What it does:** This script establishes the baseline valuation framework (xT) and extracts robust physical context from raw StatsBomb logs. It translates a list of discrete events into a mathematically connected spatial sequence.
* **Why it exists:** Machine Learning models require structural context. A raw pass coordinate has no value without understanding the space around it. This script calculates the base probability of a goal occurring (xT) and maps the geometric congestion of the pitch.
* **How it works:** 
  1. **xT Grid Generation:** It mathematically derives a 16x12 Expected Threat grid by analyzing thousands of historical shot-ending sequences and mapping the probability scalar values per pitch quadrant. Output: `models/xt_grid.csv`.
  2. **Spatial Engineering:** It calculates Euclidean distances (`math.hypot()`) between the ball carrier and opposing players leveraging the `three-sixty/` tracking data. It counts opponents within 5m/10m radii and identifies geometrically unblocked passing lanes.
  3. **Baseline DQ Scoring:** It computes a continuous heuristic Decision Quality (DQ) score based on weighted sub-scores for retention, progression (ΔxT), and risk.

### 2. `pipeline/build_training_data.py` (The Feature Vectorizer)
* **What it does:** Transforms the spatially enriched events into a structured tensor ready for supervised machine learning.
* **Why it exists:** XGBoost requires strictly formatted, tabular datasets without data leakage. The raw event stream contains variable lengths and nested JSON objects that must be flattened, encoded, and imputed correctly.
* **How it works:**
  1. **Label Generation:** Constructs the vital `realized_delta_xT` continuous target variable. It determines the true value generated (or lost) by evaluating the receiving endpoint's xT minus the originating endpoint's xT. Turnovers result in sharp negative penalties based on original location risk.
  2. **Feature Extraction:** It rigidly flattens the dataset into exactly 28 engineered features encompassing *State* (coords, score diff), *Context* (density metrics), *Geometry* (target coordinates, angles, vector distances), and *Categoricals*.
  3. **Data Imputation:** Handles the imputation of missing 360° frames (e.g., zeroing spatial density metrics where tracking cameras lost the players) to prevent feature breakage. Output: `ml_training.csv`.

### 3. `pipeline/train_model.py` (The Machine Learning Engine)
* **What it does:** Trains the predictive model that enables the "Multiverse" alternative-choice scoring.
* **Why it exists:** Football decisions are non-linear interactions. A pass to an open player is poor if the vector angle forces them backward under pressure. We use an eXtreme Gradient Boosting (`XGBRegressor`) architecture because it inherently segments complex, non-linear physical interactions.
* **How it works:**
  1. **Training Architecture:** Instantiates the XGBoost Regressor (500 trees, depth 6) on the tabular feature matrix.
  2. **GroupKFold Validation Strategy:** To prevent critical data leakage, it utilizes a rigorous 5-fold cross-validation scheme grouped by `match_id`. This guarantees the model does not "memorize" a specific team's idiosyncratic defensive shape from the first half and apply it artificially in the second half.
  3. **Artifact Serialization:** Serializes the trained weights to `models/vaep_xgb.json` and exports objective partial dependence/SHAP-like `feature_importance.csv` rankings.

### 4. `mvp/analyze_match.py` (The Multiverse Inference Script)
* **What it does:** Acts as the bridge between raw event ingestion and the predictive model, performing high-volume batch inference to determine optimal actions.
* **Why it exists:** The model only knows how to score what it is fed. In order to evaluate a decision, we must simulate the alternative decisions the player *could have made* but didn't. This script generates that synthetic candidate pool.
* **How it works:**
  1. **Candidate Generation:** Pauses time at every action event and creates synthetic data points based on player tracking: *"What if the player passed to visible Teammate C? What if they carried forward 8 meters? What if they shot?"*
  2. **Batch Scoring:** Batches this universe of alternative choices against the in-memory XGBoost model. The model yields a predicted `expected_delta_xT` for every generated edge case.
  3. **Optimization:** Compares the actual action realized against the ML's optimal choice array. Extreme negative deviations flag the event as a poor decision.

### 5. `mvp/dashboard_server.py` (The Full-Stack Orchestrator)
* **What it does:** The presentation layer bridge. It wraps the core ML inference engine into an accessible, low-latency REST API, serving the dynamic HTML/JS interface.
* **Why it exists:** Executives, sporting directors, and coaches cannot read command-line script outputs on the touchline. Data only carries value when properly socialized and visualized via consumer-grade interfaces.
* **How it works:**
  1. **Lazy Loading:** It loads the dense `vaep_xgb.json` model into RAM exactly once on startup to ensure sub-second API resolution times for match processing.
  2. **Tactical Role Extraction:** It parses the `Starting XI` and `Substitution` events inside the StatsBomb file to dynamically assign positional classifications (GK, DEF, MID, FWD) to players.
  3. **Role-Aware AI Coaching:** It integrates these tactical markers into the ML outputs to generate contextual coaching texts. *An attacker is actively encouraged to make risky progressive carries resulting in occasional turnovers; a center-back committing the exact same actions is heavily penalized in the text readout.*
  4. **Custom Serialization:** It utilizes a custom Numpy JSON encoder to serialize complex multi-dimensional mathematical arrays to be parsed by the client. This drives the DOM rendering in `frontend/index.html` and the Canvas manipulations housed inside `frontend/app.js`.

---

## How to Run the Environment

The system is designed for zero-cloud dependency execution on any laptop or workstation.

**Prerequisites:**
Ensure Python 3.9+ is installed along with the essential data science suite (`pandas`, `numpy`, `xgboost`, `scikit-learn`, `flask`, `matplotlib`).

**Startup Execution:**
1. Clone the repository and navigate into the project root directory.
2. Activate your virtual environment: `source .venv/bin/activate`
3. Launch the presentation layer:
   ```bash
   cd mvp
   python dashboard_server.py
   ```
4. Access the command center via your browser at `http://localhost:5050`.

---

## Conclusion

This platform represents a paradigm shift. We have moved from descriptive statistics ("What happened?") to prescriptive intelligence ("What is the optimal path?"). By marrying state-of-the-art computer vision data (360° frames) with non-linear machine learning architectures (XGBoost) and wrapping it in an intuitive, actionable managerial dashboard, we provide an end-to-end operational tool for elite football clubs to systematically evaluate and improve player intelligence.
