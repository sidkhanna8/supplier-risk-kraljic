
#!/usr/bin/env python3
# Train a classifier to predict late deliveries (risk modeling).
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main(args):
    df = pd.read_csv(args.input_csv, parse_dates=['order_date'])
    df['late_flag'] = (df['actual_lt_days'] > df['promised_lt_days']).astype(int)
    X = df[['category','region','units_ordered','promised_lt_days','unit_cost_usd','defects_ppm','financial_rating','geo_risk','single_source_flag']]
    y = df['late_flag']
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['category','region'])], remainder='passthrough')
    clf = Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=1000))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    report = classification_report(yte, yhat, digits=3)
    Path('outputs').mkdir(parents=True, exist_ok=True)
    with open(args.report_output, 'w') as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input_csv', default='data/supplier_po_lines.csv')
    p.add_argument('--report_output', default='outputs/late_delivery_model.txt')
    args = p.parse_args()
    main(args)
