
#!/usr/bin/env python3
# Compute supplier/category KPIs and Kraljic matrix segmentation.
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def zscore(s):
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def main(args):
    df = pd.read_csv(args.input_csv, parse_dates=['order_date'])

    # Spend per PO
    df['line_spend'] = df['units_ordered'] * df['unit_cost_usd']

    # Supplier-level KPIs
    sup = df.groupby('supplier_id').agg(
        pos=('po_id','count'),
        spend=('line_spend','sum'),
        on_time_rate=('on_time_flag','mean'),
        avg_lt=('actual_lt_days','mean'),
        defects_ppm=('defects_ppm','mean'),
        fin=('financial_rating','mean'),
        geo=('geo_risk','mean'),
        single_src_rate=('single_source_flag','mean')
    ).reset_index()

    # Risk score (higher = riskier). Combine delivery, quality, geo, single-source, financial
    sup['risk_score'] = (
        (1 - sup['on_time_rate']).fillna(0) * 0.35 +
        (sup['defects_ppm'] / (sup['defects_ppm'].mean() + 1e-9)).clip(0,3) * 0.20 +
        sup['geo'].fillna(0) * 0.20 +
        sup['single_src_rate'].fillna(0) * 0.15 +
        (1 - sup['fin']).fillna(0) * 0.10
    )

    # Normalize for Kraljic axes
    sup['profit_impact'] = zscore(sup['spend'])  # proxy by spend (can be replaced with margin/criticality if known)
    sup['supply_risk']  = zscore(sup['risk_score'])

    # Thresholds by quantile (flexible)
    q_hi = 0.6
    q_lo = 0.4
    risk_cut  = sup['supply_risk'].quantile(q_hi)
    imp_cut   = sup['profit_impact'].quantile(q_hi)
    risk_low  = sup['supply_risk'].quantile(q_lo)
    imp_low   = sup['profit_impact'].quantile(q_lo)

    def kraljic_cell(risk, impact):
        hi_risk = risk >= risk_cut
        hi_imp  = impact >= imp_cut
        lo_risk = risk <= risk_low
        lo_imp  = impact <= imp_low
        # Default mid goes to whichever side it's closer to; use simple logic
        if hi_risk and hi_imp:
            return "Strategic"
        if (not hi_risk) and hi_imp:
            return "Leverage"
        if hi_risk and (not hi_imp):
            return "Bottleneck"
        return "Routine"

    sup['kraljic_segment'] = [kraljic_cell(r,i) for r,i in zip(sup['supply_risk'], sup['profit_impact'])]

    Path('outputs').mkdir(parents=True, exist_ok=True)
    sup.to_csv(args.supplier_kpis_csv, index=False)
    print("Wrote:", args.supplier_kpis_csv)

    # Category x supplier view (optional)
    cat_sup = df.groupby(['category','supplier_id']).agg(
        pos=('po_id','count'),
        spend=('line_spend','sum'),
        on_time_rate=('on_time_flag','mean'),
        defects_ppm=('defects_ppm','mean'),
        geo=('geo_risk','mean')
    ).reset_index()
    cat_sup.to_csv(args.category_supplier_csv, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input_csv', default='data/supplier_po_lines.csv')
    p.add_argument('--supplier_kpis_csv', default='outputs/supplier_kpis_kraljic.csv')
    p.add_argument('--category_supplier_csv', default='outputs/category_supplier_summary.csv')
    args = p.parse_args()
    main(args)
