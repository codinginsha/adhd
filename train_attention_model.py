import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def window_features(df, window=10):
    X, y = [], []
    subjects = df['subject'].unique()
    for s in subjects:
        sub = df[df['subject']==s].sort_values('t')
        vals = sub[['hr','eda','accel']].values
        labels = sub['attention'].values
        for i in range(len(vals)-window):
            win = vals[i:i+window].flatten()
            lab = 1 if labels[i:i+window].mean() > 0.5 else 0
            X.append(win)
            y.append(lab)
    return np.array(X), np.array(y)

def main():
    df = pd.read_csv('wearable_timeseries.csv')
    X, y = window_features(df, window=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, 'attention_model.joblib')
    print('Trained model saved to attention_model.joblib')

if __name__ == '__main__':
    main()
