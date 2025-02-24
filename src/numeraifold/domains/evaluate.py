
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

def evaluate_domain_performance(train_df, val_df, domain_features, target_col):
    """
    Train a quick LGBM model on the given domain's features and return the ROC AUC on val_df.
    """
    model = LGBMClassifier(random_state=42)
    model.fit(train_df[domain_features], train_df[target_col])
    preds = model.predict_proba(val_df[domain_features])[:, 1]
    return roc_auc_score(val_df[target_col], preds)