#!/usr/bin/env python3
"""Generate the 42 colourblind-friendly figures used for the supplied results.

Figures 41 and 42 extend the original 102-language family visualization into two
explicit views: morphology index versus character-domain decay and morphology
index versus speech-time decay. Both views retain all 102 languages. The 44
languages without a matching UD treebank are shown in a dedicated NA strip rather
than being silently dropped or assigned an imputed morphology value.

Usage:
  python plot_results.py --results-dir results --output-dir plots
  python plot_results.py --results-dir results --output-dir plots --pdf

The script expects the result files produced by the information-locality pipeline.
It also writes derived_metrics_used_in_plots.csv and PLOT_CATALOG.md.
"""
from pathlib import Path
import argparse, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

# Okabe-Ito colourblind-friendly palette + redundant marker encoding.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
BLUE, ORANGE, GREEN, PURPLE, GOLD, CYAN, YELLOW, BLACK = PALETTE
MID, LIGHT = "#6B7280", "#D1D5DB"
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 240,
    "font.size": 10.5, "axes.titlesize": 15,
    "axes.labelsize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.7,
    "legend.frameon": False,
})

def wrap(s, width=125):
    return "\n".join(textwrap.wrap(str(s), width=width))

def savefig(fig, path, caption):
    path = Path(path)
    fig.subplots_adjust(bottom=0.22, top=0.89, left=0.10, right=0.98)
    fig.text(0.10, 0.035, "Caption — " + wrap(caption), ha="left", va="bottom",
             fontsize=9, color="#374151", linespacing=1.35)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def corr_text(d, x, y):
    z = d[[x, y]].dropna()
    if len(z) < 3:
        return f"n={len(z)}"
    pr = pearsonr(z[x], z[y]); sr = spearmanr(z[x], z[y])
    return f"n={len(z)}; Pearson r={pr.statistic:.3f} (p={pr.pvalue:.3g}); Spearman ρ={sr.statistic:.3f} (p={sr.pvalue:.3g})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output-dir", default="plots")
    ap.add_argument("--pdf", action="store_true", help="Also assemble all PNGs into a multipage PDF.")
    args = ap.parse_args()

    r = Path(args.results_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lf = pd.read_csv(r / "language_features.csv")
    lc = pd.read_csv(r / "locality_curves.csv")
    rcm = pd.read_csv(r / "regression" / "regression_character_metrics.csv")
    rtm = pd.read_csv(r / "regression" / "regression_time_domain_metrics.csv")
    rcs = pd.read_csv(r / "regression" / "regression_character_sensitivity_metrics.csv")
    rts = pd.read_csv(r / "regression" / "regression_time_domain_sensitivity_metrics.csv")
    rcc = pd.read_csv(r / "regression" / "regression_character_coefficients.csv")
    rtc = pd.read_csv(r / "regression" / "regression_time_domain_coefficients.csv")
    rcp = pd.read_csv(r / "regression" / "regression_character_predictions.csv")
    rtp = pd.read_csv(r / "regression" / "regression_time_domain_predictions.csv")
    family_path = r / "language_family_assignments.csv"
    if not family_path.exists():
        raise FileNotFoundError(f"Missing required family-assignment file: {family_path}")
    family_df = pd.read_csv(family_path)
    required_family_cols = {"language", "iso639_3", "family"}
    missing_family_cols = required_family_cols - set(family_df.columns)
    if missing_family_cols:
        raise ValueError(f"Family-assignment file is missing columns: {sorted(missing_family_cols)}")
    if family_df["language"].duplicated().any():
        dup = family_df.loc[family_df["language"].duplicated(), "language"].tolist()
        raise ValueError(f"Duplicate language rows in family-assignment file: {dup}")
    family_df = family_df[["language", "iso639_3", "family"]].copy()
    if len(family_df) != len(lf) or set(family_df["language"]) != set(lf["language"]):
        raise ValueError("language_family_assignments.csv does not match the 102 language rows in language_features.csv")
    lf = lf.merge(family_df, on=["language", "iso639_3"], how="left", validate="one_to_one")

    lf["D_order"] = lf.H_order - lf.H_original
    lf["D_structure"] = lf.H_structure - lf.H_original
    lf["half_life_chars"] = np.log(2) / lf.locality_lambda_per_char
    lf["half_life_seconds"] = np.log(2) / lf.locality_lambda_per_s
    d = lf.dropna(subset=["D_order", "D_structure", "morphology_index"])

    regions = sorted(lf.region_group.dropna().unique())
    rc = {rr: PALETTE[i % len(PALETTE)] for i, rr in enumerate(regions)}
    rm = {rr: MARKERS[i % len(MARKERS)] for i, rr in enumerate(regions)}
    captions = {}

    def region_scatter(ax, x, y, data=None):
        data = lf if data is None else data
        for rr in sorted(data.region_group.dropna().unique()):
            g = data[data.region_group == rr]
            ax.scatter(g[x], g[y], s=46, c=rc[rr], marker=rm[rr], alpha=.86,
                       edgecolors="white", linewidths=.5, label=rr)

    def region_legend(ax, ncol=2):
        handles = [Line2D([0], [0], marker=rm[rr], color="none",
                          markerfacecolor=rc[rr], markeredgecolor="white",
                          markersize=8, label=rr) for rr in sorted(regions)]
        ax.legend(handles=handles, loc="best", ncol=ncol, fontsize=8.5)

    def emit(num, stem, fig, cap):
        fn = f"{num:02d}_{stem}.png"
        savefig(fig, out / fn, cap)
        captions[fn] = cap

    # 1–3: locality surface
    q25 = lc.groupby("lag_characters").information_bits.quantile(.25)
    q75 = lc.groupby("lag_characters").information_bits.quantile(.75)
    mean = lc.groupby("lag_characters").information_bits.mean()
    med = lc.groupby("lag_characters").information_bits.median()
    fig, ax = plt.subplots(figsize=(10.8, 7))
    for _, g in lc.groupby("language"):
        ax.plot(g.lag_characters, g.information_bits, color=LIGHT, alpha=.24, lw=.7)
    ax.fill_between(q25.index, q25.values, q75.values, color=CYAN, alpha=.25, label="Language IQR")
    ax.plot(mean.index, mean.values, color=BLUE, lw=2.6, label="Mean")
    ax.plot(med.index, med.values, color=ORANGE, lw=2.2, ls="--", label="Median")
    ax.axhline(0, color=BLACK, lw=1); ax.set(xlabel="Lag (characters)", ylabel="Information retained, Iₜ (bits)", title="Information locality across 102 FLEURS languages"); ax.legend()
    emit(1, "global_locality_curves_characters", fig, "Each faint line is one language's locality curve; the mean, median and interquartile band summarize the cross-language distribution.")

    q25 = lc.groupby("lag_seconds").information_bits.quantile(.25); q75 = lc.groupby("lag_seconds").information_bits.quantile(.75)
    mean = lc.groupby("lag_seconds").information_bits.mean(); med = lc.groupby("lag_seconds").information_bits.median()
    fig, ax = plt.subplots(figsize=(10.8, 7))
    for _, g in lc.groupby("language"):
        ax.plot(g.lag_seconds, g.information_bits, color=LIGHT, alpha=.24, lw=.7)
    ax.fill_between(q25.index, q25.values, q75.values, color=GREEN, alpha=.22, label="Language IQR")
    ax.plot(mean.index, mean.values, color=BLUE, lw=2.6, label="Mean"); ax.plot(med.index, med.values, color=ORANGE, lw=2.2, ls="--", label="Median")
    ax.axhline(0, color=BLACK, lw=1); ax.set(xlabel="Lag (seconds)", ylabel="Information retained, Iₜ (bits)", title="Information locality after speech-time calibration"); ax.legend()
    emit(2, "global_locality_curves_seconds", fig, "The same locality curves in seconds, showing how speech-rate calibration changes temporal persistence relative to character persistence.")

    pv = lc.pivot(index="language", columns="lag_characters", values="information_bits")
    pv = pv.reindex(lf.sort_values("locality_lambda_per_char", ascending=False).language)
    fig, ax = plt.subplots(figsize=(11.5, 16)); im = ax.imshow(pv.values, aspect="auto", cmap="cividis", vmin=0, vmax=np.nanpercentile(pv.values, 98))
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns); ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index, fontsize=7.2)
    ax.set(xlabel="Lag (characters)", ylabel="Language", title="Where does information persist?"); fig.colorbar(im, ax=ax, pad=.02, label="Iₜ (bits)")
    emit(3, "information_retention_heatmap", fig, "A language-by-lag heatmap that makes the complete 102-language locality surface visible without relying on 102 overlapping line colours.")

    # 4–6: distributions / derived persistence
    specs = [(4, "locality_lambda_per_char", CYAN, "λ per character", "Distribution of character-domain locality decay rates", ".3f"),
             (5, "locality_lambda_per_s", GREEN, "λ per second", "Distribution of time-domain locality decay rates", ".2f")]
    for no, col, color, xlab, title, fmt in specs:
        fig, ax = plt.subplots(figsize=(10.4, 6.5)); v = lf[col].dropna(); ax.hist(v, bins=18, color=color, edgecolor="white")
        ax.axvline(v.median(), color=ORANGE, lw=2.4, ls="--", label=f"Median = {v.median():{fmt}}")
        ax.axvline(v.mean(), color=BLUE, lw=2.4, label=f"Mean = {v.mean():{fmt}}")
        ax.set(xlabel=xlab, ylabel="Number of languages", title=title); ax.legend()
        emit(no, "lambda_character_distribution" if no==4 else "lambda_time_distribution", fig,
             "The language-level distribution of the exponential decay parameter in character space." if no==4 else "The language-level distribution of the exponential decay parameter after speech-time calibration.")
    fig, ax = plt.subplots(figsize=(10.4, 6.5)); q = ax.scatter(lf.half_life_chars, lf.half_life_seconds, s=46, c=lf.locality_lambda_r2, cmap="cividis", edgecolors="white")
    fig.colorbar(q, ax=ax, pad=.02, label="Locality fit R²"); ax.set(xlabel="Half-life (characters)", ylabel="Half-life (seconds)", title="Information half-life")
    emit(6, "locality_half_life_map", fig, "Half-life is ln(2)/λ and translates the fitted decay parameter into an interpretable persistence distance.")

    # 7–12: extremes, regional distributions, speech rate
    for no, col, color, xlabel, stem, title in [
        (7, "locality_lambda_per_char", BLUE, "λ per character", "lambda_character_extremes", "Languages at the two ends of character-domain locality decay"),
        (8, "locality_lambda_per_s", GREEN, "λ per second", "lambda_time_extremes", "Languages at the two ends of time-domain locality decay")]:
        b = pd.concat([lf.nsmallest(12, col), lf.nlargest(12, col)]).drop_duplicates("language").sort_values(col)
        fig, ax = plt.subplots(figsize=(10.8, 9)); ax.barh(b.language, b[col], color=[color if v < lf[col].median() else PURPLE for v in b[col]])
        ax.axvline(lf[col].median(), color=BLACK, ls=":", lw=1.2); ax.set(xlabel=xlabel, title=title)
        emit(no, stem, fig, "Descriptive extremum plot showing the 12 lowest- and 12 highest-rate languages; it is not an overall ranking of languages.")

    for no, col, title, xlabel, stem in [
        (9, "locality_lambda_per_char", "Character-domain locality by broad region group", "λ per character", "region_lambda_character"),
        (10, "locality_lambda_per_s", "Time-domain locality by broad region group", "λ per second", "region_lambda_time"),
        (11, "speech_char_rate_cps", "Speech-rate calibration by broad region group", "Speech character rate (characters/second)", "speech_rate_region")]:
        order = lf.groupby("region_group")[col].median().sort_values().index
        fig, ax = plt.subplots(figsize=(11.5, 7.4)); data = [lf.loc[lf.region_group==rr, col].dropna().values for rr in order]
        ax.boxplot(data, orientation="horizontal", tick_labels=order, showfliers=False)
        for j,(rr,arr) in enumerate(zip(order,data),1):
            jit=np.linspace(-.12,.12,len(arr)) if len(arr)>1 else [0]; ax.scatter(arr, np.full(len(arr),j)+jit, s=28, c=rc[rr], marker=rm[rr], alpha=.72, edgecolors="white", linewidths=.35)
        ax.set(xlabel=xlabel, title=title)
        emit(no, stem, fig, "Boxes summarize within-region distributions and overlaid points show individual languages; sparse groups should be interpreted descriptively.")

    fig, ax = plt.subplots(figsize=(10,7)); region_scatter(ax,"speech_char_rate_cps","locality_lambda_per_s"); z=lf[["speech_char_rate_cps","locality_lambda_per_s"]].dropna(); sl,ic=np.polyfit(z.speech_char_rate_cps,z.locality_lambda_per_s,1); xx=np.linspace(z.speech_char_rate_cps.min(),z.speech_char_rate_cps.max(),100); ax.plot(xx,sl*xx+ic,color=BLACK,ls="--")
    ax.set(xlabel="Speech character rate",ylabel="λ per second",title="Time-domain decay is strongly coupled to speech rate"); region_legend(ax)
    emit(12,"lambda_time_vs_speech_rate",fig,"The positive association between speech-character rate and λ/second is partly algebraically induced because λ/second = λ/character × speech-character-rate.")

    # 13–26: morphology, controls and fit diagnostics
    fig, ax=plt.subplots(figsize=(9.8,8)); q=ax.scatter(d.D_order,d.D_structure,c=d.morphology_index,cmap="cividis",s=62,edgecolors="white"); lim=max(d.D_order.max(),d.D_structure.max())*1.05; ax.plot([0,lim],[0,lim],color=BLACK,ls="--"); fig.colorbar(q,ax=ax,pad=.02,label="Morphology index"); ax.set(xlabel="D_order",ylabel="D_structure",title="The analytic–synthetic information trade-off")
    emit(13,"morphology_tradeoff_map",fig,"D_order and D_structure show the relative contributions of word order and within-word structure; morphology index is the structural share of their sum.")

    pairs=[
        (14,"morphology_index","locality_lambda_per_char","Morphological balance vs character-domain locality"),
        (15,"morphology_index","locality_lambda_per_s","Morphological balance vs time-domain locality"),
        (16,"word_order_redundancy","locality_lambda_per_char","Word-order redundancy vs locality decay"),
        (17,"word_structure_redundancy","locality_lambda_per_char","Word-structure redundancy vs locality decay"),
        (18,"memory_cost_bits_char","locality_lambda_per_char","Memory cost and locality decay"),
        (19,"ud_morph_feature_density","locality_lambda_per_char","UD morphological feature density vs locality"),
        (20,"ud_mean_dependency_distance","locality_lambda_per_char","Dependency distance vs locality decay"),
        (21,"lexical_ttr","locality_lambda_per_char","Lexical diversity vs locality decay"),
        (22,"mean_word_length_chars","locality_lambda_per_char","Mean word length vs locality decay"),
        (23,"n_chars","locality_lambda_per_char","Corpus size is not obviously driving λ")]
    for no,x,y,title in pairs:
        z=lf.dropna(subset=[x,y]); fig,ax=plt.subplots(figsize=(9.8,7)); region_scatter(ax,x,y)
        if no==23: ax.set_xscale("log")
        ax.set(xlabel=x,ylabel=y,title=title); region_legend(ax)
        stem_map={14:"morphology_index_vs_lambda_character",15:"morphology_index_vs_lambda_time",16:"word_order_redundancy_vs_lambda",17:"word_structure_redundancy_vs_lambda",18:"memory_cost_vs_lambda",19:"ud_morph_feature_density_vs_lambda",20:"dependency_distance_vs_lambda",21:"lexical_ttr_vs_lambda",22:"mean_word_length_vs_lambda",23:"corpus_size_vs_lambda"}
        stem=stem_map[no]
        emit(no,stem,fig,f"Bivariate diagnostic for {x} versus {y}. The plot is descriptive and should not be interpreted causally; it checks whether a simple language-level covariate visibly tracks locality.")

    fig,ax=plt.subplots(figsize=(9.6,7)); q=ax.scatter(lf.locality_lambda_r2,lf.locality_fit_rmse_bits,s=54,c=lf.n_positive_information_points,cmap="cividis",edgecolors="white"); ax.axvline(.9,color=ORANGE,ls="--"); ax.set(xlabel="Locality fit R²",ylabel="Fit RMSE (bits)",title="Exponential locality fit quality"); fig.colorbar(q,ax=ax,pad=.02,label="Positive decay points")
    emit(24,"locality_fit_quality",fig,"Most exponential locality fits are strong; the two weakest are Cantonese Chinese and Mandarin Chinese, which merit sensitivity attention.")

    fig,ax=plt.subplots(figsize=(9.6,7)); region_scatter(ax,"hilberg_alpha","locality_lambda_per_char"); ax.set(xlabel="Hilberg exponent α",ylabel="λ per character",title="Alternative scaling exponent vs exponential locality decay"); region_legend(ax)
    emit(25,"hilberg_alpha_vs_lambda",fig,"Hilberg α and exponential locality λ summarize different aspects of information structure and show only a weak bivariate association in these results.")

    fig,ax=plt.subplots(figsize=(10.4,6.5)); ax.hist(lf.hilberg_r2.dropna(),bins=20,color=PURPLE,edgecolor="white"); ax.axvline(0,color=BLACK,ls="--"); ax.set(xlabel="Hilberg fit R²",ylabel="Number of languages",title="The Hilberg fit is substantially weaker than the locality fit")
    emit(26,"hilberg_fit_distribution",fig,"83 of 102 Hilberg R² values are negative, contrasting with the high R² of the primary locality exponential fits.")

    # 27–33: regression
    perf=[("Ridge · character",rcm.cv_r2_log_outcome.iloc[0])]+[(f"{m} · character",v) for m,v in zip(rcs.model,rcs.cv_r2_log_outcome)]+[("Ridge · time",rtm.cv_r2_log_outcome.iloc[0])]+[(f"{m} · time",v) for m,v in zip(rts.model,rts.cv_r2_log_outcome)]
    pp=pd.DataFrame(perf,columns=["model","R2"]); fig,ax=plt.subplots(figsize=(11,7.4)); xx=np.arange(len(pp)); ax.bar(xx,pp.R2,color=[BLUE if "character" in m else GREEN for m in pp.model],width=.68); ax.axhline(0,color=BLACK,lw=1); ax.set_xticks(xx); ax.set_xticklabels(pp.model,rotation=25,ha="right",fontsize=9.3); ax.set_ylabel("Leave-one-language-out CV R² on log outcome"); ax.set_title("Cross-language regression generalization is weak",pad=14); ax.margins(x=.04)
    for b,v in zip(ax.patches,pp.R2): ax.text(b.get_x()+b.get_width()/2,v+(.018 if v>=0 else -.028),f"{v:.3f}",ha="center",va="bottom" if v>=0 else "top",fontsize=9)
    emit(27,"regression_cv_performance",fig,"All reported cross-validated R² values are negative, so the current multivariable feature set does not generalize to held-out languages better than the baseline.")

    for no,df,xlab,ylab,title,color in [
        (28,rcp,"Observed log λ/character","Predicted log λ/character","Character-domain regression: observed vs cross-validated predicted",BLUE),
        (29,rtp,"Observed log λ/second","Predicted log λ/second","Time-domain regression: observed vs cross-validated predicted",GREEN)]:
        fig,ax=plt.subplots(figsize=(8.8,8)); ax.scatter(df.observed_log_outcome,df.predicted_log_outcome,s=54,c=color,alpha=.78,edgecolors="white"); lim=[min(df.observed_log_outcome.min(),df.predicted_log_outcome.min())-.1,max(df.observed_log_outcome.max(),df.predicted_log_outcome.max())+.1]; ax.plot(lim,lim,color=BLACK,ls="--"); ax.set(xlabel=xlab,ylabel=ylab,title=title)
        emit(no,"character_observed_vs_predicted" if no==28 else "time_observed_vs_predicted",fig,"Each point is one held-out language and the dashed diagonal is perfect prediction; visible scatter around the diagonal explains the negative cross-validated R².")

    for no,df,title,color in [(30,rcp,"Largest LOCO-CV errors · character domain",ORANGE),(31,rtp,"Largest LOCO-CV errors · time domain",PURPLE)]:
        q=df.copy(); q["absres"]=q.cv_residual.abs(); q=q.nlargest(20,"absres").sort_values("cv_residual"); fig,ax=plt.subplots(figsize=(10.8,9)); ax.barh(q.language,q.cv_residual,color=color); ax.axvline(0,color=BLACK); ax.set_xlabel("Residual in log outcome"); ax.set_title(title)
        emit(no,"character_residuals" if no==30 else "time_residuals",fig,"The 20 largest absolute leave-one-language-out residuals identify where the supplied predictor set misses language-specific locality variation.")

    for no,df,color,title in [(32,rcc,BLUE,"Largest fitted character-domain coefficients"),(33,rtc,GREEN,"Largest fitted time-domain coefficients")]:
        q=df.sort_values("coefficient_log_outcome"); q=pd.concat([q.head(10),q.tail(10)]).drop_duplicates().sort_values("coefficient_log_outcome"); fig,ax=plt.subplots(figsize=(11,9)); ax.barh(q.feature,q.coefficient_log_outcome,color=[ORANGE if v<0 else color for v in q.coefficient_log_outcome]); ax.axvline(0,color=BLACK); ax.set_xlabel("Coefficient on log outcome"); ax.set_title(title)
        emit(no,"character_regression_coefficients" if no==32 else "time_regression_coefficients",fig,"The largest positive and negative fitted coefficients are descriptive model parameters; the negative LOCO-CV R² means they should not be read as validated causal or predictive effects.")

    # 34–40: coverage / sampling / structural summaries
    cov=lf.groupby("region_group").agg(n=("language","size"),ud=("ud_status",lambda s:(s=="loaded").sum())); cov["missing"]=cov.n-cov.ud; cov=cov.sort_values("n",ascending=False)
    fig,ax=plt.subplots(figsize=(11.5,7.5)); y=np.arange(len(cov)); ax.barh(y,cov.ud,color=GREEN,label="UD available"); ax.barh(y,cov.missing,left=cov.ud,color=LIGHT,label="No matching UD treebank"); ax.set_yticks(y); ax.set_yticklabels(cov.index); ax.set_xlabel("Languages"); ax.set_title("Structural-data coverage is uneven across the 102-language sample"); ax.legend()
    emit(34,"ud_coverage_by_region",fig,"FLEURS locality is available for all 102 languages, but only 58 have matching UD annotations, so morphology analyses are necessarily a 58-language subset.")

    q=lf.sort_values("locality_effective_train_chars"); fig,ax=plt.subplots(figsize=(11,11)); ax.barh(q.language,q.locality_effective_train_chars,color=CYAN); ax.axvline(100000,color=ORANGE,ls="--"); ax.set_xlabel("Effective KN training characters"); ax.set_title("Effective locality-estimation sample size by language")
    emit(35,"effective_kn_training_chars",fig,"Effective KN training characters show where the adaptive estimator used less text than the original 100k upper cap, preventing low-resource languages from being rejected solely because they are smaller.")

    fig,ax=plt.subplots(figsize=(9.6,7)); q=ax.scatter(lf.locality_effective_train_chars,lf.locality_effective_valid_chars,c=lf.locality_eval_positions,cmap="cividis",s=58,edgecolors="white"); ax.axvline(100000,color=ORANGE,ls="--"); ax.axhline(20000,color=ORANGE,ls="--"); ax.set(xlabel="Effective train characters",ylabel="Effective validation characters",title="Adaptive train/validation allocation"); fig.colorbar(q,ax=ax,pad=.02,label="Evaluation positions")
    emit(36,"effective_train_vs_validation",fig,"The dashed lines mark the originally requested 100k/20k caps; points below them show where actual language-level text availability forced a smaller but valid allocation.")

    for no,col,color,title in [(37,"word_order_redundancy",GOLD,"Distribution of word-order redundancy"),(38,"word_structure_redundancy",PURPLE,"Distribution of word-structure redundancy")]:
        fig,ax=plt.subplots(figsize=(10.3,6.5)); ax.hist(d[col],bins=16,color=color,edgecolor="white"); ax.set(xlabel=col.replace("_"," ").title(),ylabel="Number of languages",title=title); emit(no,f"{col}_distribution",fig,"A direct distribution of one component of the analytic–synthetic trade-off, useful for seeing whether a few languages dominate the cross-language range.")

    order=d.groupby("region_group").morphology_index.median().sort_values().index; data=[d.loc[d.region_group==rr,"morphology_index"].dropna().values for rr in order]
    fig,ax=plt.subplots(figsize=(11.5,7.4)); ax.boxplot(data,orientation="horizontal",tick_labels=order,showfliers=False)
    for j,(rr,arr) in enumerate(zip(order,data),1):
        jit=np.linspace(-.12,.12,len(arr)) if len(arr)>1 else [0]; ax.scatter(arr,np.full(len(arr),j)+jit,s=28,c=rc[rr],marker=rm[rr],alpha=.75,edgecolors="white")
    ax.set_xlabel("Morphology index"); ax.set_title("Morphology index varies across region groups")
    emit(39,"morphology_index_by_region",fig,"The morphology-index distribution differs across broad region groups, but this is an annotated 58-language subset with uneven UD coverage.")

    fig,ax=plt.subplots(figsize=(10.2,7.4)); region_scatter(ax,"memory_cost_bits_char","half_life_chars"); ax.set(xlabel="Memory cost (bits/character)",ylabel="Locality half-life (characters)",title="Memory burden and locality persistence"); region_legend(ax)
    emit(40,"memory_cost_vs_half_life",fig,"A synthesis map combining cumulative memory cost with information half-life, giving a compact view of the study's efficiency–locality theme.")

    # 41-42: all-102 family-coded morphology/locality views.
    # The morphology index exists for only 58 languages because it requires a matching
    # UD treebank. Locality lambda exists for all 102. The 44 missing morphology values
    # are therefore visualized explicitly in an NA strip rather than omitted.
    fam_colors = {
        "Indo-European": "#0072B2", "Niger-Congo": "#D55E00", "Afro-Asiatic": "#009E73",
        "Austronesian": "#CC79A7", "Austroasiatic": "#E69F00", "Dravidian": "#56B4E9",
        "Turkic": "#8C6BB1", "Uralic": "#F0E442", "Sino-Tibetan": "#009E9A",
        "Kra-Dai": "#B15928", "Japonic": "#6A3D9A", "Koreanic": "#1B9E77",
        "Kartvelian": "#E7298A", "Mongolic": "#66A61E", "Nilotic": "#A6761D",
        "Creole": "#7570B3"
    }
    fam_markers = {
        "Indo-European": "o", "Niger-Congo": "s", "Afro-Asiatic": "^", "Austronesian": "D",
        "Austroasiatic": "P", "Dravidian": "X", "Turkic": "v", "Uralic": "<",
        "Sino-Tibetan": ">", "Kra-Dai": "*", "Japonic": "h", "Koreanic": "8",
        "Kartvelian": "p", "Mongolic": "H", "Nilotic": "d", "Creole": "o"
    }
    fam_order = [
        "Indo-European", "Niger-Congo", "Afro-Asiatic", "Austronesian", "Turkic", "Dravidian",
        "Sino-Tibetan", "Uralic", "Austroasiatic", "Kra-Dai", "Japonic", "Kartvelian",
        "Koreanic", "Mongolic", "Nilotic", "Creole"
    ]
    fam_df = lf.copy()
    fam_df["code"] = fam_df["iso639_3"].astype(str).str.lower()
    measured = fam_df[fam_df["morphology_index"].notna()].copy()
    missing_morph = fam_df[fam_df["morphology_index"].isna()].copy()
    if len(fam_df) != 102 or len(measured) != 58 or len(missing_morph) != 44:
        raise ValueError(
            f"Expected 102 total / 58 measured / 44 missing morphology values; "
            f"got {len(fam_df)} / {len(measured)} / {len(missing_morph)}"
        )

    def family_handles():
        return [Line2D([0], [0], marker=fam_markers[f], color="none",
                       markerfacecolor=fam_colors[f], markeredgecolor="white",
                       markersize=8, label=f) for f in fam_order if f in set(fam_df["family"])]

    def family_plot(no, xcol, stem, xlab, title, caption):
        fig, ax = plt.subplots(figsize=(18, 13))
        ax.set_facecolor("#FBFBFC")
        xpad = (fam_df[xcol].max() - fam_df[xcol].min()) * 0.035
        x0 = fam_df[xcol].min() - xpad
        x1 = fam_df[xcol].max() + xpad
        yb, yt = -0.29, -0.012
        ax.set_xlim(x0, x1)
        ax.set_ylim(yb, 1.08)
        ax.axhspan(yb, yt, facecolor="#EEF0F3", alpha=.72, hatch="///", edgecolor="#C7CBD1", linewidth=.0, zorder=0)
        for fam in fam_order:
            g = measured[measured["family"] == fam]
            if len(g):
                ax.scatter(g[xcol], g["morphology_index"], s=68, c=fam_colors[fam], marker=fam_markers[fam],
                           edgecolors="white", linewidths=.7, alpha=.95, zorder=4)

        offsets = [(5,5),(-5,5),(7,-6),(-7,-6),(9,0),(-9,0),(0,9),(0,-9),
                   (11,6),(-11,6),(12,-7),(-12,-7)]
        ordered = measured.sort_values([xcol, "morphology_index"]).reset_index(drop=True)
        for k, r0 in ordered.iterrows():
            x, y = float(r0[xcol]), float(r0["morphology_index"])
            local = ((np.abs(measured[xcol]-x) < (0.025 if xcol.endswith("char") else 0.15)) &
                     (np.abs(measured["morphology_index"]-y) < .025)).sum()
            dx, dy = offsets[k % len(offsets)]
            scale = 1 + min(max(int(local)-1, 0), 3) * .45
            dx, dy = dx*scale, dy*scale
            ax.annotate(
                r0["code"], (x,y), xytext=(dx,dy), textcoords="offset points",
                fontsize=7.9, fontweight="bold", color="#30343B", ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=.13", fc="white", ec="none", alpha=.82),
                arrowprops=(dict(arrowstyle="-", color="#9AA1AA", lw=.45, alpha=.50, shrinkA=2, shrinkB=2)
                            if abs(dx)+abs(dy)>13 else None)
            )

        ax.axhline(float(measured["morphology_index"].median()), color=BLACK, ls="--", lw=1.15, alpha=.75)
        ax.text(x0 + .012*(x1-x0), float(measured["morphology_index"].median()) + .016,
                f"Median measured morphology index = {measured['morphology_index'].median():.3f}",
                fontsize=9.1, color="#4B5563", va="bottom")

        # Four staggered rows preserve each missing language's exact x-coordinate.
        for j, r0 in enumerate(missing_morph.sort_values(xcol).itertuples(index=False)):
            yy = yt - .048 - (j % 4) * .058
            ax.text(float(getattr(r0, xcol)), yy, str(getattr(r0, "code")),
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color=fam_colors[getattr(r0, "family")],
                    bbox=dict(boxstyle="round,pad=.10", fc="white", ec="none", alpha=.88), zorder=6)
        ax.text(x0 + .012*(x1-x0), yb + .19,
                f"NA morphology index - {len(missing_morph)} languages with no matching UD treebank",
                fontsize=9.4, fontweight="bold", color="#4B5563", va="center")
        ax.text(x1 - .012*(x1-x0), yb + .030,
                "No morphology value is imputed for the 44 NA cases.",
                fontsize=8.7, color="#6B7280", ha="right", va="center")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Morphology index  M = D_structure / (D_order + D_structure)")
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.965)
        fig.text(0.5, 0.925,
                 "All 102 FLEURS languages | 58 measured morphology values + 44 UD-missing values | colour + marker = broad genealogical family",
                 ha="center", va="center", fontsize=10.2, color="#4B5563")
        ax.legend(handles=family_handles(), loc="upper center", bbox_to_anchor=(0.5, .99), ncol=4,
                  fontsize=8.4, frameon=True, framealpha=.93, title="Language family", title_fontsize=9.2)
        emit(no, stem, fig, caption)

    family_plot(
        41, "locality_lambda_per_char", "morphology_index_vs_lambda_character_102_family",
        "Information-locality decay rate, λ/character (1/character)",
        "Morphology-locality landscape across all 102 FLEURS languages: character domain",
        "All 102 language rows are retained. The 58 languages with a matching UD treebank are plotted at their measured morphology index; the 44 languages without a matching UD treebank are shown in the hatched NA strip at their observed character-domain decay coordinate. No morphology value is imputed."
    )
    family_plot(
        42, "locality_lambda_per_s", "morphology_index_vs_lambda_time_102_family",
        "Information-locality decay rate, λ/time (1/s)",
        "Morphology-locality landscape across all 102 FLEURS languages: speech-time domain",
        "All 102 language rows are retained. The 58 languages with a matching UD treebank are plotted at their measured morphology index; the 44 languages without a matching UD treebank are shown in the hatched NA strip at their observed speech-time decay coordinate. No morphology value is imputed."
    )

    lf[["language","iso639_3","family","region_group","locality_lambda_per_char","locality_lambda_per_s","half_life_chars","half_life_seconds","memory_cost_bits_char","speech_char_rate_cps","morphology_index","ud_status","D_order","D_structure","ud_morph_feature_density","ud_mean_dependency_distance","locality_lambda_r2","locality_fit_rmse_bits","hilberg_alpha","hilberg_r2","locality_effective_train_chars","locality_effective_valid_chars","locality_eval_positions"]].to_csv(out/"derived_metrics_used_in_plots.csv",index=False)
    (out/"PLOT_CATALOG.md").write_text("# Plot catalog\n\n"+"\n\n".join(f"### {k}\n{v}" for k,v in sorted(captions.items(), key=lambda kv:int(kv[0].split("_")[0]))),encoding="utf-8")

    if args.pdf:
        with PdfPages(out/"multilingual_locality_plots.pdf") as pdf:
            pngs = [
                p for p in out.glob("*.png")
                if p.name.split("_", 1)[0].isdigit()
            ]
            for p in sorted(pngs, key=lambda q: int(q.name.split("_", 1)[0])):
                img = plt.imread(p)
                fig = plt.figure(figsize=(11.7, 8.3))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Generated {len(list(out.glob('*.png')))} PNG figures in {out}")
    if args.pdf: print(f"Generated {out/'multilingual_locality_plots.pdf'}")

if __name__ == "__main__":
    main()
