from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def train_model(df):
    df = df.copy()

    X = df.drop(['math score', 'reading score', 'writing score', 'average', 'passed'], axis=1)
    y = df['passed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Importance des variables
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return model, X_test, y_test, y_pred, feature_importance_df

def evaluate_model(y_test, y_pred):
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    matrix = confusion_matrix(y_test, y_pred)
    return report_df, matrix
