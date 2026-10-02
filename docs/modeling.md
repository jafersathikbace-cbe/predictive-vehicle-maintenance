# Modeling notes

The target is `Need_Maintenance`. The training workflow uses a stratified 80/20 split and compares XGBoost, LightGBM, and Random Forest. The selected artifact is the model with the highest recall for the positive maintenance class, while precision, F1, and ROC-AUC are reported for context.

This selection rule prioritizes identifying maintenance cases, but it should be revisited with domain-specific costs and a fresh validation set before production use.
