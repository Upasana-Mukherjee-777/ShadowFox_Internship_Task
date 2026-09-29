# Boston House Price Prediction Using Regression Models

**Prepared by:** Upasana Mukherjee
**Programme:** B.Tech Computer Science and Engineering (AI & ML), Institute of Engineering & Management / University of Engineering & Management, Kolkata
**Assignment level:** Beginner Machine Learning project

---

## 1. Abstract

This project builds a regression system that predicts the median price of houses in Boston neighbourhoods from thirteen recorded characteristics such as crime rate, average number of rooms and pupil-teacher ratio. The supplied dataset contains 506 records, of which 112 have at least one missing value. After imputing the missing values inside a leak-free scikit-learn pipeline, five models were compared: Linear Regression, Ridge Regression, Decision Tree, Random Forest and Gradient Boosting. Models were compared by five-fold cross-validation on the training data, and the selected model was evaluated once on a held-out test set of 102 houses. Gradient Boosting gave the lowest cross-validated error and, after a small grid search, reached a test MAE of 1.87, RMSE of 2.69 and R² of 0.901 (prices in thousands of dollars, assuming the unit of the original dataset). The number of rooms (`RM`) and the share of lower-status population (`LSTAT`) were the most influential features. The final model shows noticeable overfitting (training R² 0.997), and the price variable appears to be capped at 50, so the model should be treated as a learning exercise rather than a valuation tool.

## 2. Introduction

Predicting house prices is a standard introductory problem in machine learning. The price is a continuous quantity, so the task is one of *regression*, a branch of supervised learning in which a model learns the relationship between input features and a known numerical target from examples. Once trained, the model can estimate the target for houses it has never seen.

Reliable evaluation matters as much as the model itself. A model must be judged on data it was not trained on, and every preprocessing step must be learned from the training data alone. This report follows that discipline throughout.

## 3. Problem Statement

Using the provided dataset containing features such as the number of rooms, crime rates and other relevant factors, design and implement a regression model to predict Boston house prices. The solution should involve data preprocessing, model selection, training and evaluation.

## 4. Objectives

1. Inspect the dataset and identify the target variable without assuming its structure.
2. Prepare the data by handling missing values and scaling features without data leakage.
3. Explore the data to understand its distribution and the relationships between features and price.
4. Train and compare several regression models.
5. Evaluate the models with MAE, MSE, RMSE and R², select a final model on sound evidence and save it.
6. Provide a working prediction function and a simple user interface.

## 5. Dataset Description

The dataset was supplied as `HousingData.csv`. It has 506 rows and 14 columns, no duplicate rows and no constant columns. All columns are numeric. Missing values are written as the text `NA`.

| Column | Meaning (standard definition of the Boston Housing variables) |
|---|---|
| CRIM | Per-capita crime rate |
| ZN | Share of residential land zoned for large lots |
| INDUS | Share of non-retail business acres |
| CHAS | 1 if the tract borders the Charles River, otherwise 0 |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms per dwelling |
| AGE | Share of owner-occupied units built before 1940 |
| DIS | Weighted distance to employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Property-tax rate |
| PTRATIO | Pupil-teacher ratio |
| B | Index derived from the proportion of Black residents |
| LSTAT | Percentage of lower-status population |
| MEDV | Median home value (target) |

The file contains no data dictionary. The meanings above are taken from the public documentation of the original dataset, and the unit of `MEDV` ($1000s) is assumed from that source rather than stated in the file.

**Target.** `MEDV` is the only numeric, complete column that represents the quantity to be predicted, so it was chosen as the target. Its mean is 22.53, median 21.2, minimum 5.0 and maximum 50.0. The distribution is right-skewed (skewness 1.11) and 16 records lie exactly at 50.0, which suggests that values were capped at that level.

**Missing values.** Six columns (`CRIM`, `ZN`, `INDUS`, `CHAS`, `AGE`, `LSTAT`) each have 20 missing entries (3.95 %). In total 112 of 506 rows (22.1 %) contain at least one missing value.

## 6. Methodology

The work followed this sequence: load and inspect the data; preprocess; explore; split into training (80 %, 404 rows) and test (20 %, 102 rows) sets with `random_state=42`; compare five models by five-fold cross-validation on the training set; select and tune the best model; evaluate it once on the test set; analyse errors; save the model; build the prediction interface. Preprocessing and each model were combined into a single scikit-learn `Pipeline`, so that medians and scaling parameters are learned from training data only.

## 7. Data Preprocessing

**Missing values.** Deleting incomplete rows would remove 22 % of the data. The mean price of rows with a missing value (23.14) is close to that of complete rows (22.36), so they do not appear to form a distinct group. Missing values were therefore imputed: with the training-set median for numeric columns (robust against extreme values such as those in `CRIM`) and with the most frequent value for the binary `CHAS`.

**Data types and duplicates.** No conversions were needed. `CHAS` is stored as a decimal only because of its missing entries. No duplicate rows or constant columns exist.

**Outliers.** The interquartile-range rule flags 65 values of `CRIM`, 63 of `ZN` (most towns have a value of 0) and 30 of `RM`, among others. These correspond to real neighbourhoods rather than evident errors, and tree-based models are not sensitive to extreme values, so they were kept.

**Feature scaling.** The features differ widely in scale (for example `NOX` about 0.5 and `TAX` about 400). Standardisation was applied for Linear and Ridge Regression, where scale affects the fit or the penalty, and not for tree-based models, which compare each feature only with thresholds.

**Sensitive column.** Column `B` is derived from the proportion of Black residents in each town. Using a racial attribute to estimate property value is ethically problematic, so `B` was excluded and twelve features were used. A comparison of the default Gradient Boosting model with and without `B` gave test RMSE 2.68 and 2.72 respectively (R² 0.902 and 0.899), so the exclusion has almost no effect on accuracy.

**Data leakage.** Leakage occurs when information from the test data influences training, for example when a median is computed from all rows before the split. To avoid it, the data was split first and all imputation and scaling values were learned inside the pipeline from the training rows only.

## 8. Exploratory Data Analysis

Figures are stored in `outputs/plots/`.

* **Distribution of the target** (`02_target_distribution.png`). Most prices lie between 17.0 and 25.0 (interquartile range). The mean exceeds the median, and a visible spike at 50 supports the capping observation.
* **Correlation heatmap** (`03_correlation_heatmap.png`). The strongest correlations with price are `LSTAT` (-0.74) and `RM` (+0.70), followed by `PTRATIO` (-0.51), `INDUS` (-0.48) and `TAX` (-0.47). Some features are strongly correlated with each other, most notably `RAD` and `TAX` (0.91).
* **Features against price** (`04_features_vs_price.png`). Price increases roughly linearly with `RM`. The relation with `LSTAT` is curved, which a straight-line model cannot fully capture. Areas priced below 15 have a median crime rate of 9.4 compared with 0.25 for the whole dataset.
* **Boxplots** (`05_feature_boxplots.png`). `CRIM` is heavily skewed with many high values; `RM` has outliers on both sides.

Correlation measures association only and does not show that one variable causes another.

## 9. Models Used

| Model | Principle | Strength | Limitation |
|---|---|---|---|
| Linear Regression | Fits a straight-line formula by least squares | Simple, interpretable | Cannot model curved relationships |
| Ridge Regression | Linear regression with a penalty on large coefficients | More stable with correlated features | Still linear |
| Decision Tree (max depth 5) | Splits the data with successive threshold questions | Captures non-linear effects; no scaling | Unstable; overfits when deep |
| Random Forest (200 trees) | Averages many trees trained on random samples | Accurate and robust; gives feature importance | Slower, less interpretable |
| Gradient Boosting | Adds trees one by one, each correcting earlier errors | Often the most accurate on tabular data | Sensitive to settings; can overfit |

## 10. Model Training

All models were built as pipelines (imputation, optional scaling, model). Five-fold cross-validation on the training data was used to estimate generalisation error, and the test set played no part in model selection. Each model was then fitted on the full training set. For the selected model, a grid search over the number of trees (100, 200, 400), learning rate (0.03, 0.05, 0.1) and tree depth (2, 3, 4) was run with the same cross-validation.

## 11. Evaluation Metrics

* **MAE** is the average absolute prediction error.
* **MSE** is the average squared error and penalises large mistakes strongly.
* **RMSE** is the square root of MSE and is in the same unit as the price.
* **R²** is the proportion of variance in the target explained by the model (1 is perfect; 0 is no better than predicting the mean).

No model was chosen on R² alone. RMSE, MAE, cross-validated scores and the gap between training and test performance were all considered.

## 12. Results

**Cross-validation on the training set (five folds).**

| Model | CV RMSE (mean ± sd) | CV MAE | CV R² |
|---|---|---|---|
| Linear Regression | 5.091 ± 0.687 | 3.602 | 0.693 |
| Ridge Regression | 5.088 ± 0.695 | 3.597 | 0.693 |
| Decision Tree | 6.141 ± 0.587 | 3.648 | 0.528 |
| Random Forest | 4.125 ± 0.496 | 2.621 | 0.794 |
| Gradient Boosting | 3.664 ± 0.512 | 2.449 | 0.840 |

**Test set (102 houses).**

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | 3.079 | 23.659 | 4.864 | 0.677 |
| Ridge Regression | 3.075 | 23.673 | 4.865 | 0.677 |
| Decision Tree | 2.467 | 9.625 | 3.102 | 0.869 |
| Random Forest | 2.049 | 9.038 | 3.006 | 0.877 |
| Gradient Boosting | 1.906 | 7.386 | 2.718 | 0.899 |
| Gradient Boosting (tuned) | 1.868 | 7.246 | 2.692 | 0.901 |

**Training versus test.** The tuned model reaches R² 0.997 (RMSE 0.483) on the training data and 0.901 (RMSE 2.692) on the test data. The large gap indicates overfitting. Linear and Ridge Regression score similarly on both sets (R² 0.732 and 0.677), which points to underfitting.

**Feature importance.** In the Random Forest, `RM` (0.562) and `LSTAT` (0.249) together account for about 81 % of the total importance; `DIS` (0.058) and `CRIM` (0.045) follow, and the remaining features contribute 0.02 or less each.

## 13. Model Comparison

The tree ensembles clearly outperformed the linear models: test RMSE was about 4.86 for Linear and Ridge Regression, 3.01 for Random Forest and 2.72 for Gradient Boosting. Ridge Regression was practically identical to ordinary Linear Regression, so its penalty added little. Gradient Boosting had the lowest cross-validated RMSE (3.66 against 4.13 for Random Forest); the difference is about one standard deviation of the fold-to-fold variation, so it is a modest rather than decisive advantage.

The Decision Tree is a reminder that a single test split can mislead: it scored R² 0.869 on the test set but only 0.528 in cross-validation, with a large spread between folds (0.18). The test set contains only 102 houses. For this reason model selection relied on cross-validation, and the test set was used only for the final report.

Gradient Boosting was selected because it had the best cross-validated RMSE, MAE and R², and also the best test scores among the default models. Tuning reduced the cross-validated RMSE from 3.66 to 3.60. The best settings (400 trees, learning rate 0.1, depth 3) lie at the edge of the grid, so a wider search might improve the result slightly.

**Actual against predicted** (`actual_vs_predicted.png`). Most points lie close to the diagonal. The largest miss is a house with an actual price of 50 predicted at 36.5; all three test houses priced at 50 were under-predicted, which is consistent with a capped target.

**Residual analysis** (`residual_analysis.png`). Residuals (actual minus predicted) have a mean of 0.44 and a standard deviation of 2.66. 78.4 % of predictions are within ±3 and 94.1 % within ±5 (in $1000s). Apart from the capped house (+13.5) the residuals lie between about -6 and +7. The seven test houses priced at 35 or more have a mean residual of +3.3, a sign that expensive areas may be under-predicted, although seven points are too few for a firm conclusion. No formal statistical assumptions are claimed.

## 14. Prediction System

The final pipeline (imputer and Gradient Boosting model) is saved as `models/best_model.pkl` with `joblib`, together with `model_metadata.json` listing the feature names and training ranges. The function `predict_price()` in `src/predict.py` loads the pipeline, checks that all twelve features are given, applies the same preprocessing as in training (missing values are imputed with the training median) and returns the predicted price. It can be used from the command line, from Python, or through the Streamlit application `app.py`. For a demonstration house from the test set, the model predicted 23.48 against an actual price of 23.60 (in $1000s).

## 15. Limitations

* The model overfits: training R² is 0.997 against a test R² of 0.901.
* The test set is small (102 houses), so the reported test scores carry uncertainty.
* Prices appear capped at 50, which limits accuracy for expensive areas.
* Imputation with medians is simple and may not preserve relationships between features.
* The dataset is small, old and from a single city, so the model does not generalise to other markets or to current prices.
* The unit of the target is assumed rather than documented in the supplied file.
* Feature importance and correlation describe association, not causation.
* Hyper-parameter tuning used a small grid, and the best values lay on its edge.

## 16. Future Scope

Use repeated cross-validation and a wider hyper-parameter search; test XGBoost, LightGBM and stacked models; apply model-based imputation; transform the skewed target and `CRIM`; add explanation methods such as SHAP and prediction intervals; validate on a modern, larger dataset; and deploy the application online.

## 17. Conclusion

A complete regression pipeline was built on the supplied dataset. After leak-free preprocessing and comparison of five models by cross-validation, tuned Gradient Boosting was chosen as the final model and achieved MAE 1.87, RMSE 2.69 and R² 0.901 on unseen test data. Tree-based ensembles handled the non-linear structure of the data far better than linear models, and the number of rooms and the proportion of lower-status population were the dominant predictors. The results are encouraging for an introductory project, but the overfitting, small test set and capped target mean that they should be read with care.

## 18. References

1. Harrison, D., & Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. *Journal of Environmental Economics and Management, 5*(1), 81-102.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
4. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55-67.
5. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
6. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189-1232.
7. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
8. scikit-learn documentation: https://scikit-learn.org/stable/ (user guide sections on pipelines, cross-validation and model evaluation).
