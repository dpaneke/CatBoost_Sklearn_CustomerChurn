# CatBoost_Sklearn_CustomerChurn

This repository contains two solutions for [customer churn problem](https://www.kaggle.com/c/telecom-clients-prediction2/overview/description) on Kaggle 

The first solution  `Sklearn_CustomerChurn_kaggle.ipynb` includes some base machine learning approaches based on use of sklearn library.

It contains, among other things:
- Data preparation
- Custom FrequencyEncoder
- Custom Transformer (with pipeline containing different data processing and encoding strategies: `OneHot`, `Ordinal`, `Frequency`)
- Different base algorithms comparison (`RidgeClassiifer`, `RandomForestClassifier`, `GradientBoostingClassifier`, `LGBMClassifier`)
- Final pipeline, containing data transformer, feature selectors, random oversampler and gradient boosting classiifer
- Grid search for the best parameters
- Final evaluation score of `0.707` (better than the other `35%` of participants)

The second solution  `CatBoost_CustomerChurn_kaggle.ipynb` deals with problem using `catboost` library.

It contains, among other things:
- Data preparation
- Feature content investigation
- Custom Transformer (with pipeline containing different scaling and NaN filling strategies)
- Сustom feature selector NullPercentageThreshold (depending on occupancy)
- Final model containing data transformer, feature selector, and CatBoost classiifer
- Grid search for the best parameters
- Final evaluation score of `0.722` (better than the other `75%` of participants)
CatBoost_CustomerChurn_kaggle.ipynb
