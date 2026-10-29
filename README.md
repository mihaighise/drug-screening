# drug-screening

Tree-based regression for drug screening: predicting how sensitive cancer cell lines are to 13 anti-cancer drugs from their gene expression, comparing six tree-based regressors per drug, and explaining the best model for each drug with SHAP.

All of the code is in `HW4_Code_Ghise.py`.

## Dataset

`hw4-drug-screening-data.csv` (not included in the repository because of its size, 331 MB) has 12,502 rows and 2,532 columns:

| Column(s) | Meaning |
|---|---|
| `CELL_LINE_NAME` | The cancer cell line tested (963 cell lines) |
| `DRUG_NAME` | The drug applied (13 drugs) |
| `LN_IC50` | **Target**: natural log of the IC50, the drug concentration needed to inhibit 50% of cell growth. Lower values mean the cell line is more sensitive to the drug. |
| 2,529 gene columns (`A2M`, `AATF`, …) | Gene expression levels of the cell line |

Each row is one cell line–drug pair. The dataset is balanced: every drug has between 961 and 963 samples. A cell line's gene expression is identical across all of its rows, since it describes the cell line, not the drug.

The 13 drugs are 5-Fluorouracil, Camptothecin, Docetaxel, MG-132, MK-2206, Navitoclax, Nutlin-3a (-), Olaparib, PD0325901, PD173074, Palbociclib, Pictilisib and Staurosporine.

## Pipeline

### 1. Exploration and cleaning

- Plots the number of samples per drug as a bar chart.
- Checks every gene column for missing values: no gene is entirely missing.
- Removes constant genes, which have the same value in every row and carry no information. One gene (`KREMEN1`) is dropped, leaving 2,528 genes as features.

### 2. Drug-specific regression

Because each drug responds to gene expression differently, a separate model is trained for each drug. For every drug:

1. Only that drug's rows (~962 cell lines) are selected.
2. The features are the 2,528 genes and the target is `LN_IC50`. The drug name is not a feature, since it is constant within one drug's data.
3. Each of the six models below is evaluated with **5-fold cross-validation** (`KFold`, shuffled, `random_state=42`). No feature scaling is applied, because tree-based models do not need it.
4. MAE, MSE, RMSE and R² are computed on each held-out fold and averaged over the 5 folds.

This gives 13 drugs × 6 models = 78 sets of metrics.

| Model | Library | Main settings |
|---|---|---|
| Decision Tree | scikit-learn | `max_depth=5`, `min_samples_leaf=10` |
| Random Forest | scikit-learn | 300 trees, `max_features="sqrt"`, `min_samples_leaf=5` |
| Gradient Boosting Machine | scikit-learn | 200 trees, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8` |
| XGBoost | xgboost | 300 trees, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.5` |
| LightGBM | lightgbm | 300 trees, `learning_rate=0.05`, `num_leaves=15`, `subsample=0.8`, `colsample_bytree=0.5` |
| CatBoost | catboost | 300 iterations, `learning_rate=0.05`, `depth=4` |

The best model for each drug is the one with the lowest cross-validated RMSE.

### 3. SHAP feature importance

For each drug, its best model is refit on all of that drug's data and explained with SHAP (`shap.TreeExplainer`, which computes exact SHAP values for tree-based models). A gene's importance is its mean absolute SHAP value over all cell lines: how much it moves the predicted `LN_IC50`, on average, in either direction. The 10 most important genes per drug are plotted and saved.

## Outputs

Running the script produces:

| Output | Contents |
|---|---|
| `drug_model_metrics.csv` | MAE, MSE, RMSE and R² for every drug and model (78 rows) |
| `figures/<drug>.png` | 13 grouped bar charts, one per drug: the four metrics on the x-axis, one bar per model, with value labels |
| `shap_top10_features.csv` | The top 10 genes per drug with their rank and mean \|SHAP\| value |
| `shap_plots/<drug>_shap.png` | 13 SHAP bar plots, one per drug, showing its top 10 genes |

The full cross-validation (390 model fits) takes several hours, mostly because of the Gradient Boosting Machine.

## Results

### Best regressor per drug

| Drug | Best model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|---|
| 5-Fluorouracil | CatBoost | 1.137 | 2.016 | 1.416 | 0.309 |
| Camptothecin | XGBoost | 1.162 | 2.204 | 1.484 | 0.344 |
| Docetaxel | CatBoost | 1.000 | 1.674 | 1.292 | 0.318 |
| MG-132 | CatBoost | 0.446 | 0.408 | 0.637 | 0.211 |
| MK-2206 | LightGBM | 0.991 | 1.579 | 1.256 | 0.271 |
| Navitoclax | XGBoost | 1.109 | 2.050 | 1.429 | 0.486 |
| Nutlin-3a (-) | LightGBM | 0.860 | 1.264 | 1.123 | 0.596 |
| Olaparib | Gradient Boosting Machine | 0.724 | 0.873 | 0.933 | 0.477 |
| PD0325901 | LightGBM | 1.148 | 2.354 | 1.526 | 0.491 |
| PD173074 | CatBoost | 0.913 | 1.404 | 1.181 | 0.280 |
| Palbociclib | CatBoost | 0.909 | 1.321 | 1.149 | 0.461 |
| Pictilisib | XGBoost | 0.904 | 1.335 | 1.154 | 0.152 |
| Staurosporine | Gradient Boosting Machine | 0.914 | 1.380 | 1.173 | 0.274 |

- A boosting model is best for every drug: CatBoost 5 times, XGBoost 3, LightGBM 3, Gradient Boosting Machine 2. The boosting models are often within a few hundredths of each other in RMSE.
- R² of the best models ranges from 0.15 (Pictilisib) to 0.60 (Nutlin-3a): gene expression explains part, but not all, of each drug's response.
- The single Decision Tree performs worst for every drug and has a negative R² for 7 drugs, meaning it predicts worse than the mean. With ~770 training rows and ~2,500 genes, one tree overfits to chance patterns; ensembles that average many trees (Random Forest) or build them gradually (boosting) generalize much better.

### Top 10 genes per drug (SHAP)

| Drug | Best model | Top 10 genes (most important first) |
|---|---|---|
| 5-Fluorouracil | CatBoost | VEGFB, WAS, VAMP3, BCL2L1, EEF2, RHEB, RPS27L, MYBL1, LYN, SOCS2 |
| Camptothecin | XGBoost | NQO1, TJP1, AGAP2, PHB2, HDAC11, THBS1, DLG4, APTX, TPT1, CTSD |
| Docetaxel | CatBoost | BCL2L1, BID, ABCB1, DFFB, APTX, ATP6AP2, PTPN1, RPTOR, PTMA, BTC |
| MG-132 | CatBoost | CES1, CDC37, TINF2, S1PR2, ASAP1, FZD4, APTX, GATAD2A, KRT8, IL12A |
| MK-2206 | LightGBM | FOXO3, KIFC3, DKK1, IRS2, IL13RA1, NCKAP1, MET, TH, SMURF2, GAB2 |
| Navitoclax | XGBoost | BCL2, MCL1, YAP1, PMAIP1, MACROH2A1, MYB, MYOF, WNT3, APPBP2, CTNNA1 |
| Nutlin-3a (-) | LightGBM | MDM2, RPS27L, AGAP2, MET, CDKN2A, BAX, CDKN1A, BCL2L1, KRT18, DDB2 |
| Olaparib | Gradient Boosting Machine | MYOF, ITGA3, SDC4, TP53, BCL2L1, MET, LMNA, CTSD, HDAC11, TCF4 |
| PD0325901 | LightGBM | ETV4, DUSP6, SPRY2, GNA15, PLA2G4A, TLR4, FCGR2A, RIN1, FZD1, ARAP3 |
| PD173074 | CatBoost | FGFR2, DOCK9, PEG10, OCLN, BCL2L1, FGF2, CLDN1, ARHGEF28, CYBB, DES |
| Palbociclib | CatBoost | TJP1, RB1, CDKN2A, WAS, EDN1, SREBF1, MAML2, RPL11, RHEB, OCLN |
| Pictilisib | XGBoost | RHEB, GAB2, FOXO3, PRKAR1A, DUSP6, TNF, NCF1, GYS1, TBC1D4, TBL1X |
| Staurosporine | Gradient Boosting Machine | RAC2, ABRAXAS1, LSP1, BCL2L1, MRTFA, DOCK2, COL7A1, ARHGAP4, PEG10, COL4A1 |

Many of the top genes match each drug's known mechanism, suggesting the models learned real biology rather than noise:

- **Navitoclax** inhibits BCL2, its top gene; MCL1 (second) is a known resistance factor.
- **Nutlin-3a** inhibits MDM2, its top gene; CDKN1A (p21) and BAX respond downstream of it.
- **PD0325901** is a MEK inhibitor; ETV4, DUSP6 and SPRY2 are standard markers of MEK/ERK pathway activity.
- **PD173074** is an FGFR inhibitor, and FGFR2 is its top gene.
- **Palbociclib** inhibits CDK4/6, which acts through RB1; RB1 and CDKN2A are in its top 3.
- **Docetaxel**: ABCB1 is the efflux pump that removes the drug from cells.
- **MK-2206** and **Pictilisib** target the PI3K/AKT pathway, and both rank AKT-pathway genes (FOXO3, RHEB, GAB2) highly.
- **BCL2L1**, a general cell-survival gene, is in the top 10 for 6 drugs.
