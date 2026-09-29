# Used Car Selling Price Prediction Using Regression Models

**Prepared by:** Upasana Mukherjee
**Programme:** B.Tech Computer Science and Engineering (AI & ML), Institute of Engineering & Management / University of Engineering & Management, Kolkata
**Assignment level:** 3rd-year academic Machine Learning project

---

## 1. Abstract

This project builds a regression system that estimates the resale price of a used car from its showroom price, age, mileage, fuel type, transmission, seller type and number of previous owners. Inspection of the supplied `car.csv` (301 rows) revealed that it mixes 200 cars with 101 motorcycles/scooters, identifiable by brand; since the assignment specifically concerns car pricing, the two-wheeler rows were excluded, leaving 198 cars after also removing two duplicate rows. Four regression models were compared by five-fold cross-validation on the training data: Linear Regression, Decision Tree, Random Forest and Gradient Boosting. Gradient Boosting had the lowest cross-validated error and, after a small grid search, was evaluated once on a held-out test set of 40 cars, reaching MAE 1.19 lakh, RMSE 2.79 lakh and R² 0.77. A closer look shows this test-set RMSE is dominated by a single very expensive, rarely-represented car; excluding it, RMSE falls to 1.24 lakh, close to the cross-validated estimate. `Present_Price`, the car's current showroom price, is by far the strongest predictor.

## 2. Introduction

Estimating a fair resale price for a used car is a common regression task: the price is a continuous number, and several observable facts about a car (its age, mileage, condition proxies) are expected to relate to it. This project follows a disciplined workflow: inspect the data before assuming its structure, clean it transparently, validate models on data they were not trained on, and report results including their weaknesses honestly.

## 3. Problem Statement

Develop an ML model for car selling price prediction and analysis. The deployed system should provide users with an approximate selling price for their cars based on several features, including fuel type, years of service, showroom price, number of previous owners, kilometers driven, whether the seller is a dealer or an individual, and transmission type.

## 4. Objectives

1. Inspect the dataset thoroughly and identify the target variable and any data-quality issues.
2. Clean the data and engineer a car-age feature, without assuming it matches a known reference dataset.
3. Explore relationships between the features and price.
4. Train and compare four regression models with cross-validation.
5. Evaluate with MAE, MSE, RMSE and R², select a final model on measured evidence, and save it.
6. Provide a working prediction function and a Streamlit deployment using the same pipeline as training.

## 5. Dataset Description

The dataset was supplied as `car.csv`: 301 rows, 9 columns, no missing values anywhere.

| Column | Meaning |
|---|---|
| Car_Name | Model name |
| Year | Year of manufacture |
| Selling_Price | Price the vehicle sold for, in lakhs of INR (target) |
| Present_Price | Current showroom price, in lakhs of INR |
| Kms_Driven | Kilometers driven |
| Fuel_Type | Petrol / Diesel / CNG |
| Seller_Type | Dealer / Individual |
| Transmission | Manual / Automatic |
| Owner | Number of previous owners |

**Target.** `Selling_Price` is the only column matching "selling price" in the problem statement: numeric, continuous, complete. It was chosen as the target.

**Critical data-quality finding.** Examining `Car_Name` closely shows that 101 of the 301 rows are motorcycles and scooters — brands such as Bajaj, Hero, Honda Activa/CB/CBR/Karizma, Hyosung, KTM, Mahindra Mojo, Royal Enfield, Suzuki Access, TVS, UM and Yamaha — not cars. This is not an isolated labelling error: the group is internally consistent and systematically different from the car rows. Two-wheeler `Selling_Price` ranges from ₹0.10L to ₹1.75L, against ₹0.35L to ₹35L for cars, and every two-wheeler runs on Petrol. Since the assignment is specifically about car price prediction, mixing the two populations would badly distort the target distribution and every relationship the model would learn. **The 101 two-wheeler rows were excluded** (saved to `outputs/results/excluded_two_wheelers.csv` for transparency), leaving 200 car rows.

**Duplicates.** Two exact duplicate rows were found among the 200 cars (an `ertiga` and a `fortuner`, each listed twice) and removed, leaving **198 rows** used for modelling.

**Outliers.** The interquartile-range rule flags a number of high `Selling_Price` and `Present_Price` values (e.g. a Land Cruiser at ₹35L, several Fortuners up to ₹33L) and a few very high-mileage old cars (an Innova at 197,176 km, a Camry at 142,000 km). These were inspected and are genuine listings, not data errors, so they were kept.

## 6. Methodology

The workflow was: load and inspect the raw data; filter out two-wheelers and duplicates; engineer `Car_Age`; explore the data; separate features and target; preprocess (encoding, optional scaling) inside a pipeline; split into training (80%, 158 cars) and test (20%, 40 cars) sets with `random_state=42`; compare four models by five-fold cross-validation on the training set; test whether a log-transformed target helps; select and tune the best model; evaluate once on the test set; analyse errors; save the model; build the prediction interface.

## 7. Data Preprocessing

**Categorical encoding.** `Fuel_Type`, `Seller_Type` and `Transmission` each take only 2–3 values, so One-Hot Encoding is appropriate: it does not create an excessive number of columns and avoids imposing an artificial order on categories that have none. `handle_unknown="ignore"` means a category never seen in training does not break a prediction.

**Numerical scaling.** Features differ in scale (`Present_Price` in single/double digits, `Kms_Driven` in the tens of thousands). Standardisation was applied only for Linear Regression, where scale affects the fit; the tree-based models are scale-invariant.

**Missing values.** None exist in the dataset, so no imputation step was needed.

**Unnecessary columns.** `Car_Name` (37 unique values across 198 rows) was excluded from the model — too high-cardinality relative to the sample size for reliable one-hot encoding, and not part of the assignment's stated feature list. Raw `Year` was replaced by the engineered `Car_Age` and then dropped to avoid redundancy.

**Data leakage.** All preprocessing (encoding, scaling) sits inside a single scikit-learn `Pipeline` together with the model, fit on the training data only and merely applied to the test data.

## 8. Exploratory Data Analysis

Figures are stored in `outputs/plots/`.

* **Target distribution** (`01_target_distribution.png`). Right-skewed (skewness 2.70): most cars sell for ₹3.6–7.5L, with a long tail from a handful of premium cars.
* **Numerical distributions** (`02_numeric_distributions.png`). `Present_Price` and `Kms_Driven` are both right-skewed; `Car_Age` is roughly bell-shaped, peaking around 3–6 years.
* **Correlation heatmap** (`03_correlation_heatmap.png`). `Present_Price` correlates strongly with `Selling_Price` (r = 0.82); `Car_Age` moderately (r = −0.35); `Kms_Driven` (−0.11) and `Owner` (−0.09) weakly.
* **Price vs Kilometers Driven** (`04_price_vs_kms.png`). No strong linear trend on its own; the few very high-mileage cars are all low-priced older cars.
* **Price vs Car Age** (`05_price_vs_age.png`) and **retained-value analysis** (`07_depreciation_vs_age.png`). Raw price vs age is noisy (r = −0.35), but the *ratio* of selling price to present price falls sharply and consistently with age (r = −0.86, mean retained value 63%) — depreciation is the clearer story.
* **Price vs Present Price** (`06_price_vs_present_price.png`). The clearest relationship in the dataset; every car sells below its current showroom price, as expected.
* **Price by Fuel Type** (`08_price_by_fuel.png`). Diesel cars average ₹10.10L (58 cars) against Petrol's ₹5.18L (138 cars). Only 2 CNG cars exist, so CNG conclusions are unreliable.
* **Price by Transmission** (`09_price_by_transmission.png`). Automatic cars average ₹11.69L (30 cars) against Manual's ₹5.70L (168 cars), likely reflecting that automatics are more common in premium segments.
* **Price by Seller Type** (`10_price_by_seller.png`). Dealers average ₹6.63L against Individuals' ₹5.52L, but only 5 of 198 cars are individually sold.
* **Price by Owner count** (`11_price_by_owner.png`). Price falls as owner count rises (0 owners ₹6.67L, 1 owner ₹4.34L, 3 owners ₹2.50L for a single car), though the 1- and 3-owner groups are very small.

None of these associations establish causation; several (fuel type, transmission) likely reflect car segment rather than a direct price effect of the factor itself.

## 9. Feature Engineering

**Car_Age** = reference_year − Year, where reference_year is the maximum Year present in the data (2018) — derived from the data itself rather than an assumed "today". This directly represents the "years of service" the assignment asks for, replacing the raw `Year` column.

**Interaction features** were considered but not added: with only 198 rows and 7 features, added complexity was judged unlikely to help and risks overfitting, so the simpler feature set was kept as instructed.

**Log-transforming the skewed target.** `Selling_Price` has a skewness of 2.70. A log1p transform (converted back with expm1 for evaluation) was tested with five-fold cross-validation for all four models:

| Model | CV RMSE, raw target | CV RMSE, log target |
|---|---|---|
| Linear Regression | 1.866 | 1.649 |
| Decision Tree | 1.841 | 1.501 |
| Random Forest | 1.239 | 1.221 |
| Gradient Boosting | 1.158 | 1.173 |

The log transform clearly helps the weaker models but is very slightly worse for Gradient Boosting, the eventual best model. **The final model therefore uses the raw target** — a decision based on the measured numbers above, not assumed from the target's skewness alone.

## 10. Feature / Target Separation

Features: `Present_Price`, `Kms_Driven`, `Car_Age`, `Owner` (numeric) and `Fuel_Type`, `Seller_Type`, `Transmission` (categorical). Target: `Selling_Price`.

## 11. Models Used

| Model | Principle | Strength | Limitation |
|---|---|---|---|
| Linear Regression | Least-squares straight-line fit | Simple, interpretable | Cannot capture non-linear effects |
| Decision Tree (depth 5) | Threshold-based splits | Captures non-linearity, no scaling needed | A single tree overfits easily |
| Random Forest (200 trees) | Averages many trees on random samples | More stable than one tree | Slower, less interpretable |
| Gradient Boosting | Sequential trees correcting earlier errors | Often most accurate on tabular data | More settings to tune, prone to overfitting on small data |

## 12. Results

**Cross-validation on the training set (five folds, raw target).**

| Model | CV RMSE (mean ± sd) | CV MAE | CV R² |
|---|---|---|---|
| Linear Regression | 1.866 ± 0.513 | 1.312 | 0.829 |
| Decision Tree | 1.841 ± 0.370 | 1.165 | 0.827 |
| Random Forest | 1.239 ± 0.457 | 0.806 | 0.929 |
| Gradient Boosting | 1.158 ± 0.491 | 0.770 | 0.938 |

**Test set (40 cars).**

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | 1.395 | 6.785 | 2.605 | 0.799 |
| Decision Tree | 1.475 | 12.354 | 3.515 | 0.634 |
| Random Forest | 1.133 | 7.186 | 2.681 | 0.787 |
| Gradient Boosting | 1.192 | 8.805 | 2.967 | 0.739 |
| Gradient Boosting (tuned) | 1.193 | 7.784 | 2.790 | 0.769 |

**Model selection was based on cross-validation, not the test table above.** By CV RMSE, Gradient Boosting (1.16) is clearly ahead of Random Forest (1.24), which is clearly ahead of the Decision Tree (1.84) and Linear Regression (1.87). Tuning (grid search over number of trees, learning rate and depth) improved CV RMSE slightly, from 1.158 to 1.136, with best settings of 400 trees, learning rate 0.05, depth 3.

**Training versus test.** The tuned model reaches R² 0.999 on training data (RMSE 0.16) against 0.769 on the test set (RMSE 2.79) — a sign of overfitting, expected given only 158 training rows and a flexible model.

**Why the test score looks worse than cross-validation suggests.** Investigating the individual test-set errors shows the single largest miss is a **Land Cruiser** — Present_Price ₹92.6L, by far the most expensive car in the dataset — with actual Selling_Price ₹35L predicted at ₹19.15L. This one car is a genuine listing, not an error, but the model has almost no comparable examples to learn from in that price bracket. Excluding just this one car, test RMSE falls from 2.79 to 1.24 lakh, close to the cross-validated estimate of 1.14–1.16. MAE (1.19 lakh), being less sensitive to one large error, is a fairer everyday summary of accuracy than RMSE here.

**Feature importance.** In the Random Forest, `Present_Price` accounts for about 0.80 of total importance, `Car_Age` about 0.14, and `Kms_Driven` about 0.05; the one-hot encoded categorical features contribute very little individually. The final Gradient Boosting model shows a similar pattern (Present_Price ≈0.76, Car_Age ≈0.20).

## 13. Model Comparison

Cross-validation and the test table agree that the Decision Tree is the weakest model. They disagree on Linear Regression, which looks competitive on the test set (RMSE 2.61) but is clearly weaker in cross-validation (RMSE 1.87 vs Gradient Boosting's 1.16) — a reminder that a single 40-row test split, especially with one high-leverage point, is not always representative. Model choice relied on cross-validation for this reason, with the test set used only once for final reporting.

**Actual vs predicted** (`actual_vs_predicted.png`). Most points sit tightly on the diagonal; the Land Cruiser is the one clear outlier, visibly separated from the rest of the cloud.

**Residual analysis** (`residual_analysis.png`). Residuals (actual − predicted) have mean 0.45 and are otherwise unremarkable aside from the Land Cruiser case (residual +15.85). No other strong pattern is visible.

## 14. Prediction System

The final pipeline (one-hot encoder, scaler where relevant, Gradient Boosting model) is saved as `models/final_model.pkl` with `joblib`, alongside `model_metadata.json` recording the feature list, categorical options and training ranges. `predict_price()` in `src/predict.py` loads the pipeline, validates that all seven features are supplied, applies the same preprocessing as training, and returns the predicted price. It is used from the command line, from Python, and by the Streamlit app.

## 15. Deployment

A Streamlit app (`app.py`) provides a form for the seven input features, a Predict button, the estimated price, and optional expandable sections for model information and performance charts. It loads the saved pipeline once (cached) rather than retraining on each prediction. It was verified with Streamlit's headless `AppTest` tool (page load and a simulated Predict click produced no exceptions); it has not been opened in an actual browser session during development.

## 16. Limitations

* The model overfits: training R² 0.999 versus test R² 0.769.
* The test set is small (40 cars), and a single high-value car dominates its RMSE.
* Only 2 CNG cars and 5 individually-sold cars exist in the whole dataset, so predictions for those categories are unreliable.
* The dataset spans 2003–2018 model years and is not current; prices and depreciation patterns will have shifted since.
* `Car_Name`/brand is not used as a feature, so the model cannot distinguish, say, a premium brand from a budget brand beyond what `Present_Price` already captures.
* Correlation and feature importance describe association, not causation.

## 17. Future Scope

Collect more high-value cars for better price-range coverage; test log-target Gradient Boosting with a wider hyper-parameter search; try XGBoost/LightGBM and simple stacking; extract car brand as a lower-cardinality categorical feature; add SHAP-based per-prediction explanations to the Streamlit app; revisit CNG and individual-seller pricing with more data; validate on a more current dataset.

## 18. Conclusion

A complete regression pipeline was built on the supplied dataset, beginning with a careful inspection that uncovered a significant data-quality issue (two-wheelers mixed into a "car" file) and correcting it transparently. After leak-free preprocessing and cross-validated comparison of four models, tuned Gradient Boosting was selected, reaching test MAE 1.19 lakh and R² 0.77 — a result shown to be limited mainly by one high-value car with few comparable training examples, not by a general failure of the approach. `Present_Price` is confirmed as the dominant driver of predicted resale value. The results are reasonable for an introductory project on a small (198-row) dataset, and the honest error analysis is offered as part of the deliverable rather than smoothed over.

## 19. References

1. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
3. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
4. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232.
5. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
6. scikit-learn documentation: https://scikit-learn.org/stable/ (user guide sections on pipelines, one-hot encoding, cross-validation).
7. Streamlit documentation: https://docs.streamlit.io/.
