
#!/usr/bin/env python3
# Visualize Kraljic matrix as a scatter plot and save to PNG.
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main(args):
    df = pd.read_csv(args.supplier_kpis_csv)
    fig, ax = plt.subplots(figsize=(9,7))
    seg_colors = {'Strategic':'red','Leverage':'green','Bottleneck':'orange','Routine':'blue'}
    for seg, g in df.groupby('kraljic_segment'):
        ax.scatter(g['profit_impact'], g['supply_risk'], label=seg, alpha=0.7, s=30, c=seg_colors.get(seg, 'gray'))
    ax.axvline(df['profit_impact'].quantile(0.6), linestyle='--', alpha=0.5)
    ax.axhline(df['supply_risk'].quantile(0.6), linestyle='--', alpha=0.5)
    ax.set_xlabel('Profit Impact (normalized spend)')
    ax.set_ylabel('Supply Risk (composite)')
    ax.set_title('Kraljic Matrix – Supplier Portfolio')
    ax.legend()
    Path('outputs').mkdir(parents=True, exist_ok=True)
    outpath = Path(args.output_png)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print("Saved figure to:", outpath)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--supplier_kpis_csv', default='outputs/supplier_kpis_kraljic.csv')
    p.add_argument('--output_png', default='outputs/kraljic_matrix.png')
    args = p.parse_args()
    main(args)
