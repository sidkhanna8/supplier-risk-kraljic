# Supplier Risk + Kraljic Matrix

End-to-end analytics that (1) computes supplier KPIs and a composite risk score, (2) segments the portfolio on the **Kraljic Matrix** (profit impact × supply risk), and (3) trains a simple classifier to predict late deliveries.

**Dataset size:** 70,000 PO lines across 650 suppliers (2023–2025).  
**Stack:** Python (pandas, numpy, scikit-learn, matplotlib).

## Repo Structure
```
data/supplier_po_lines.csv           # synthetic but realistic PO-level dataset
scripts/compute_kraljic.py           # KPIs, composite risk, Kraljic segments
scripts/train_risk_classifier.py     # predicts late deliveries
scripts/visualize_kraljic.py         # PNG Kraljic plot
outputs/                             # created at runtime
```

## Quickstart
```bash
pip install -r requirements.txt

# 1) Compute KPIs + Kraljic segmentation
python3 scripts/compute_kraljic.py --input_csv data/supplier_po_lines.csv --supplier_kpis_csv outputs/supplier_kpis_kraljic.csv

# 2) Visualize the matrix (creates outputs/kraljic_matrix.png)
python3 scripts/visualize_kraljic.py --supplier_kpis_csv outputs/supplier_kpis_kraljic.csv --output_png outputs/kraljic_matrix.png

# 3) Train risk model for late deliveries
python3 scripts/train_risk_classifier.py --input_csv data/supplier_po_lines.csv --report_output outputs/late_delivery_model.txt
```

## Interpretation
- **Risk score** blends delivery reliability (on-time rate), quality (defects PPM), geo factors, financial health, and single-sourcing exposure.
- **Profit impact** proxies with normalized spend. If you have margin/criticality, swap it in.
- **Segments:**
  - Strategic (high impact, high risk): partnership, dual sourcing, contracts/SLA enforcement.
  - Leverage (high impact, low risk): competitive bidding, target pricing.
  - Bottleneck (low impact, high risk): safety stock, supplier development, alternative specs.
  - Routine (low impact, low risk): automation, eProcurement.
