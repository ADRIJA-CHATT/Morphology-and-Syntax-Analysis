#!/usr/bin/env python3
"""Regenerate the LaTeX table fragments from ../data.

Usage: python regenerate_tables.py
Requires: pandas, scipy.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA / "language_features.csv")
fam = pd.read_csv(DATA / "language_family_assignments.csv")

def esc(x):
    s = str(x)
    for a,b in [('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#'), ('$', r'\$')]:
        s = s.replace(a,b)
    return s

def num(x, d=3):
    return '--' if pd.isna(x) else f'{float(x):.{d}f}'

with open(OUT/'summary_metrics.tex','w') as f:
    rows = [
        ('Languages', len(df)),
        ('Locality estimates', int(df.locality_lambda_per_char.notna().sum())),
        ('Morphology index values in current result', int(df.morphology_index.notna().sum())),
        ('Morphology index NA in current result', int(df.morphology_index.isna().sum())),
        ('Languages with matching UD treebank', int((df.ud_status=='loaded').sum())),
        ('Broad genealogical families', int(fam.family.nunique())),
        ('Median lambda_char', df.locality_lambda_per_char.median()),
        ('Median lambda_time', df.locality_lambda_per_s.median()),
        ('Median exponential fit R2', df.locality_lambda_r2.median()),
        ('Languages with negative Hilberg R2', int((df.hilberg_r2<0).sum())),
    ]
    f.write(r'\begin{tabular}{lr}\toprule Quantity & Value\\ \midrule'+'\n')
    for k,v in rows:
        f.write(f'{esc(k)} & {num(v,3) if isinstance(v,float) else v}\\\\\n')
    f.write(r'\bottomrule\end{tabular}'+'\n')

fc=fam.assign(measured=fam.morphology_index.notna()).groupby('family').agg(Total=('language','size'),Measured=('measured','sum'))
fc['Missing']=fc.Total-fc.Measured
fc=fc.sort_values(['Total','Measured'],ascending=[False,False])
with open(OUT/'family_coverage.tex','w') as f:
    f.write(r'\begin{tabular}{lrrr}\toprule Family & Total & Measured & Missing\\ \midrule'+'\n')
    for k,r in fc.iterrows(): f.write(f'{esc(k)} & {int(r.Total)} & {int(r.Measured)} & {int(r.Missing)}\\\\\n')
    f.write(r'\bottomrule\end{tabular}'+'\n')

corr=[]
for label,x in [('Speech character rate','speech_char_rate_cps'),('Memory cost','memory_cost_bits_char'),('Morphology index','morphology_index'),('Word-order redundancy','word_order_redundancy'),('Word-structure redundancy','word_structure_redundancy'),('UD morphology-feature density','ud_morph_feature_density'),('UD mean dependency distance','ud_mean_dependency_distance'),('Lexical TTR','lexical_ttr'),('Mean word length','mean_word_length_chars')]:
    z=df[[x,'locality_lambda_per_char']].dropna(); p=pearsonr(z[x],z.locality_lambda_per_char); s=spearmanr(z[x],z.locality_lambda_per_char)
    corr.append((label,len(z),p.statistic,p.pvalue,s.statistic,s.pvalue))
with open(OUT/'correlations.tex','w') as f:
    f.write(r'\begin{tabular}{lrrrrr}\toprule Predictor & $n$ & Pearson $r$ & $p$ & Spearman $\rho$ & $p$\\ \midrule'+'\n')
    for r in corr: f.write(f'{esc(r[0])} & {r[1]} & {num(r[2])} & {r[3]:.3g} & {num(r[4])} & {r[5]:.3g}\\\\\n')
    f.write(r'\bottomrule\end{tabular}'+'\n')

rc=pd.read_csv(DATA/'regression/regression_character_metrics.csv').iloc[0]
rt=pd.read_csv(DATA/'regression/regression_time_domain_metrics.csv').iloc[0]
sc=pd.read_csv(DATA/'regression/regression_character_sensitivity_metrics.csv')
st=pd.read_csv(DATA/'regression/regression_time_domain_sensitivity_metrics.csv')
with open(OUT/'regression_metrics.tex','w') as f:
    f.write(r'\begin{tabular}{lrrrr}\toprule Model & $n$ & LOCO-CV $R^2$ & MAE & RMSE\\ \midrule'+'\n')
    for name,r in [(r'RidgeCV / $\log\lambda_{char}$',rc),(r'RidgeCV / $\log\lambda_{time}$',rt)]:
        f.write(f'{name} & {int(r.n_languages)} & {r.cv_r2_log_outcome:.3f} & {r.cv_mae_log_outcome:.3f} & {r.cv_rmse_log_outcome:.3f}\\\\\n')
    for _,r in sc.iterrows(): f.write(f'{esc(str(r.model).replace("_"," ")+" / char")} & {int(r.n_languages)} & {r.cv_r2_log_outcome:.3f} & {r.cv_mae_log_outcome:.3f} & --\\\\\n')
    for _,r in st.iterrows(): f.write(f'{esc(str(r.model).replace("_"," ")+" / time")} & {int(r.n_languages)} & {r.cv_r2_log_outcome:.3f} & {r.cv_mae_log_outcome:.3f} & --\\\\\n')
    f.write(r'\bottomrule\end{tabular}'+'\n')

for fn,out in [('regression_character_coefficients.csv','character_coefficients.tex'),('regression_time_domain_coefficients.csv','time_coefficients.tex')]:
    c=pd.read_csv(DATA/'regression'/fn); a=c.columns[0]; b=c.columns[-1]
    with open(OUT/out,'w') as f:
        f.write(r'\begin{longtable}{lr}\toprule Feature & Coefficient\\ \midrule\endfirsthead\toprule Feature & Coefficient\\ \midrule\endhead'+'\n')
        for _,r in c.iterrows(): f.write(f'{esc(r[a])} & {float(r[b]):.6f}\\\\\n')
        f.write(r'\bottomrule\end{longtable}'+'\n')

missing=df[df.morphology_index.isna()][['language','iso639_3']].merge(fam[['language','family']],on='language').sort_values(['family','language'])
with open(OUT/'missing_44.tex','w') as f:
    f.write(r'\begin{longtable}{lll}\toprule Language & ISO-639-3 & Family\\ \midrule\endfirsthead\toprule Language & ISO-639-3 & Family\\ \midrule\endhead'+'\n')
    for _,r in missing.iterrows(): f.write(f'{esc(r.language)} & {esc(r.iso639_3)} & {esc(r.family)}\\\\\n')
    f.write(r'\bottomrule\end{longtable}'+'\n')

lang=df[['language','iso639_3','locality_lambda_per_char','locality_lambda_per_s','morphology_index']].merge(fam[['language','family']],on='language').sort_values('language')
with open(OUT/'language_summary.tex','w') as f:
    f.write(r'\begin{longtable}{p{3.1cm}p{0.9cm}p{2.7cm}rrr}\toprule Language & ISO & Family & $\lambda_{char}$ & $\lambda_{time}$ & $M$\\ \midrule\endfirsthead\toprule Language & ISO & Family & $\lambda_{char}$ & $\lambda_{time}$ & $M$\\ \midrule\endhead'+'\n')
    for _,r in lang.iterrows(): f.write(f'{esc(r.language)} & {esc(r.iso639_3)} & {esc(r.family)} & {num(r.locality_lambda_per_char,4)} & {num(r.locality_lambda_per_s,4)} & {num(r.morphology_index,4)}\\\\\n')
    f.write(r'\bottomrule\end{longtable}'+'\n')
print('Regenerated table fragments in', OUT)
