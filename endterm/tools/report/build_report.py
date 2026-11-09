"""Build the Markdown report and PDF from actual experiment outputs.

Run from any directory with Python and ReportLab installed. No API requests.
"""
from pathlib import Path
import html
import json
import re
import os
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'
DOCS=ROOT/'docs'
def read(name): return pd.read_csv(OUT/name)
def info(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
comparison=read('model_comparison.csv')
indexed=comparison.set_index('model')
selection=info('model_selection.json')
winner=selection['winner']
selected=indexed.loc[winner]
paired=info('paired_bootstrap.json')
gain=paired['selected_vs_submitted_midterm_ridge']
manifest=info('experiment_manifest.json')
nn=info('neural_network_diagnostics.json')
cluster=info('clustering_manifest.json')
pred=read('test_predictions.csv')
tail=pred.actual_hours.gt(100)
tail_share=100*pred.loc[tail,'absolute_error'].sum()/pred.absolute_error.sum()

def table(frame):
    def fmt(v):
        if pd.isna(v): return 'NA'
        return str(v).replace('|',', ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])

def short(name):
    return name.replace('original features','original').replace('expanded IGDB','IGDB').replace('Tuned neural network','MLP').replace('Tuned gradient boosting','Boosting')

rows=[]
for r in comparison.itertuples(index=False):
    rows.append({'Model / features':short(r.model),'CV MAE +/- SD':f'{r.train_cv_MAE_hours:.3f} +/- {r.train_cv_MAE_std_hours:.3f}',
        'Test MAE':f'{r.test_MAE_hours:.2f}','RMSE':f'{r.test_RMSE_hours:.2f}',
        'R2':f'{r.test_R2:.3f}','Median AE':f'{r.test_MedAE_hours:.2f}'})
model_table=table(pd.DataFrame(rows))
settings=[]
for name in ['Random forest','Tuned gradient boosting','Tuned neural network']:
    r=indexed.loc[name]
    settings.append({'Search':short(name),'Grid size':8 if name!='Tuned neural network' else 4,
        'Selected parameters':str(json.loads(r.best_parameters)).replace('regressor__model__','').replace('model__',''),
        'CV MAE +/- SD':f'{r.train_cv_MAE_hours:.3f} +/- {r.train_cv_MAE_std_hours:.3f}'})

errors=read('all_model_error_by_duration.csv')
focus=['Median baseline','Archived Midterm Ridge','Random forest','Tuned gradient boosting','Tuned neural network']
bands=['0-5 h','>5-20 h','>20-50 h','>50-100 h','>100 h']
wide=errors[errors.model.isin(focus)].pivot(index='model',columns='duration_band',values='MAE_hours').reindex(focus)[bands].round(2).reset_index()
wide['model']=wide.model.map(short)
wide.columns=['Model']+bands
case_records=read('all_model_error_examples.csv')
failure_comments=[]
for name in focus:
    band=errors[errors.model.eq(name)].set_index('duration_band')
    case=case_records[(case_records.model==name) & case_records.selection.eq('largest error among 5-100-hour games')].iloc[0]
    failure_comments.append(f"{short(name)} has MAE {band.loc['0-5 h','MAE_hours']:.2f} h for 0-5-hour games and {band.loc['>100 h','MAE_hours']:.2f} h above 100 hours. Among 5-100-hour games, its largest error is {case.title}: actual {case.actual_hours:.2f} h, predicted {case.predicted_hours:.2f} h.")
failure_text='\n\n'.join(failure_comments)
band_wins=[]
for band_name in ['0-5 h','>5-20 h','>20-50 h','>100 h']:
    candidates=errors.loc[errors.model.isin(focus[2:]) & errors.duration_band.eq(band_name)].sort_values('MAE_hours')
    r=candidates.iloc[0]
    band_wins.append(f"{short(r.model)} in the {band_name} band ({r.MAE_hours:.2f} h MAE)")
tradeoff_text='The lowest observed band errors among the two ensembles and MLP belong to '+', '.join(band_wins)+'. Different models lead in different bands. These are descriptive test comparisons; they do not override the training-CV choice or establish statistically significant band-wise gains.'
classical_comments=[]
for name in ['Linear regression original features','Decision tree original features','KNN original features','ExtraTrees IGDB + Steam']:
    b=errors.loc[errors.model.eq(name)].set_index('duration_band')
    classical_comments.append(f"{short(name)} has MAE {b.loc['0-5 h','MAE_hours']:.2f} h for 0-5-hour games and {b.loc['>20-50 h','MAE_hours']:.2f} h for >20-50-hour games")
classical_text='. '.join(classical_comments)+'. The tree and neighbour references improve some medium-duration errors while retaining substantial short-game errors. Expanded ExtraTrees reduces both bands, but its long-tail failures remain visible in the complete duration table. The fixed-feature comparisons separate nonlinear interactions from the additional metadata.'
ids=case_records.loc[case_records.model.isin(focus[2:]) & case_records.selection.eq('largest loss against archived Ridge'),'game_id'].drop_duplicates().head(3).tolist()
ids+=[int(pred.loc[pred.absolute_error.idxmax(),'game_id'])]
cases=pred.set_index('game_id').loc[list(dict.fromkeys(ids))]
original=pd.read_csv(ROOT/'reference/midterm/outputs/test_predictions.csv').set_index('game_id')
case_table=pd.DataFrame({'Game':cases.title,'Actual h':cases.actual_hours.round(2),
    'Midterm Ridge':original.loc[cases.index,'predicted_hours'].round(2),
    'RF':cases['Random forest'].round(2),'Boosting':cases['Tuned gradient boosting'].round(2),
    'MLP':cases['Tuned neural network'].round(2)}).reset_index(drop=True)
ce=read('cluster_error_summary.csv')
cw=ce[ce.model.isin(['Archived Midterm Ridge',winner])].pivot(index='cluster',columns='model',values='MAE_hours')
counts=ce[ce.model.eq(winner)].set_index('cluster').rows
cluster_table=pd.DataFrame({'Cluster':cw.index,'Test games':counts.reindex(cw.index).values,
    'Midterm Ridge MAE':cw['Archived Midterm Ridge'].round(2).values,'Selected MAE':cw[winner].round(2).values})
examples=read('cluster_error_examples.csv')
typical=examples[examples.selection.eq('near cluster median duration')]
example_table=typical[['cluster','title','igdb_themes','actual_hours','midterm_absolute_error','selected_absolute_error']].copy()
example_table.columns=['Cluster','Game','Source themes','Actual h','Ridge error','Selected error']
for col in ['Actual h','Ridge error','Selected error']: example_table[col]=example_table[col].round(2)
worst_cluster=int(cw['Archived Midterm Ridge'].idxmax())
worst_case=examples.loc[examples.cluster.eq(worst_cluster) & examples.selection.eq('largest archived Ridge error')].iloc[0]
typical_case=examples.loc[examples.cluster.eq(worst_cluster) & examples.selection.eq('near cluster median duration')].iloc[0]
rf_params=json.loads(indexed.loc['Random forest','best_parameters'])
nnrow=indexed.loc['Tuned neural network']
earlier=comparison[(comparison.feature_set=='full') & (comparison.model!='Tuned neural network')].sort_values('train_cv_MAE_hours').iloc[0]
ranked=comparison.loc[comparison.feature_set.eq('full')].sort_values('train_cv_MAE_hours')
gap=float(ranked.iloc[1].train_cv_MAE_hours-ranked.iloc[0].train_cv_MAE_hours)
peak=int(np.argmax(nn['internal_validation_R2_curve']))+1
curve=nn['internal_validation_R2_curve']

report=f'''# Predicting video game completion time

Machine Learning Algorithms Endterm

Syrym Shadiyarbek and Yerdaulet Akhanov, SE2424

## Research question and answer

Can structured game metadata help a player choose a game that fits a time budget? We predict a positive number of main-story hours. On the same 1,046 test games as the Midterm, the model selected by training CV is {winner}, with MAE {selected.test_MAE_hours:.2f} hours, RMSE {selected.test_RMSE_hours:.2f} hours, R2 {selected.test_R2:.3f} and median absolute error {selected.test_MedAE_hours:.2f} hours. The result supports rough catalog estimates, while large errors for long and open-ended games remain.

The gain against the exact submitted Midterm Ridge is {gain['MAE_reduction_hours']:.2f} hours MAE, with a conditional paired 95% interval of {gain['lower_95_hours']:.2f} to {gain['upper_95_hours']:.2f} hours. Against the median dummy, MAE is {100*(indexed.loc['Median baseline','test_MAE_hours']-selected.test_MAE_hours)/indexed.loc['Median baseline','test_MAE_hours']:.1f}% lower. These are measured stage comparisons on an already observed test, with the limitations explained below.

## Changes responding to the teacher

The teacher's feedback supplied from 2 November 2026 identified missing collection code and conclusions, empty columns, and the need to upload every file in a ZIP because GitHub would not be checked. It also commented on the defence ("защита получше"). The complete ZIP now contains collection code, raw responses, cleaned data, saved split, an executed notebook, model checks and outputs, the report and slides. The notebook displays collection source and ends with a computed conclusion. The final slide states the answer and next experiment. The seven-minute defence guide includes both speakers and five minutes of Q&A preparation.

The new data types are genuine observations on the same games. A short change log in docs/CHANGELOG.md links each feedback point to before/after evidence. Historical empty fields remain clearly identified audit columns and are excluded from predictors. Presentation and report build scripts are retained under tools/ so their construction is reproducible.

## Data, collection and cleaning

One row describes one of 5,229 real distinct games. The original target is IGDB game_time_to_beats.hastily seconds divided by 3,600, recorded on 8 October 2026: reaching credits without notable extras. The old raw field is named hltb_main_hours, but its recorded source is IGDB. No HLTB targets are mixed in.

Metadata collected on 2 November 2026 adds IGDB themes, gameplay keywords, modes, perspectives, platforms, engines and game type. Explicit linked Steam app IDs add tags, store categories, supported platforms/languages and current US USD prices. The saved snapshot contains 2,861 observed regular prices and 3,333 rows with Steam tags. Missing quotes remain missing. Prices describe current store listings and cannot support launch-day forecasting. All observed price fields and tag names are verified against saved raw responses. Timestamps, URLs, dictionaries and hash manifests accompany those responses.

src/collect.py contains the enrichment requests, batched requests/caching, bounded retries, explicit matching and environment-based credentials. reference/midterm_source/src/collect_igdb.py preserves the original target collector. Both files are inside the ZIP and visible from the notebook. Developer secrets are excluded. The saved IGDB API terms and Steam terms/robots checks document access. The IGDB website robots request returned 403, so collection used the official authenticated API. No private user profiles or individual reviews enter the data.

The supplied Midterm raw snapshot has 5,273 rows. Its inherited rules remove 28 durations below 0.1 hours, five descriptions with fewer than eight words, one title with fewer than two characters, and ten repeated normalized-title/year records, leaving 5,229 games. The target inclusion range is 0.1-1,000 hours. These short durations are real observations outside the inclusion rule, rather than demonstrated source errors. Text readiness and duplicate filters can also exclude real short games or different editions. The row-level cleaning audit reconstructs every decision. Endterm adds columns by left join and changes no games, targets or split assignments.

The main modelling dataset is data/clean/games_model_inputs.csv: 5,229 rows and 43 columns, consisting of 39 populated raw predictor fields plus game ID, title, target and saved partition for reference. No predictor is entirely empty. The data card and CSV dictionary also define the full 70-column provenance/audit view. IDs, titles, descriptions, source URLs, timestamps, other completion-time fields and explicit duration tags are excluded from predictors. Empty old price/tag fields remain audit-only. Numeric missing-value medians and missingness indicators enter the model pipeline without turning imputed values into observations.

## Exploration and modelling decisions

Three training-only summaries guide the choices below.

| Evidence | Interpretation | Decision |
| --- | --- | --- |
| Median duration 5 h, 95th percentile 40 h, 99th percentile 100 h | A small tail can dominate squared error. | Select by hour MAE, report RMSE/R2, and test log-target models. |
| RPG median 14.25 h over 246 training games; Indie 3 h over 374 | Broad genres carry information but conceal variation. | Add detailed themes, modes, perspectives and gameplay tags. |
| 2,305 observed training prices; Spearman price/duration correlation 0.618 | Price and duration vary together in the store-linked subset. | Test price jointly, report coverage and compare fixed-estimator feature sets. |

![Training duration distribution](../outputs/figures/target_distribution.png)

These associations do not establish causal effects. Selective Steam coverage and correlated predictors limit interpretations of price and tag importance.

## Fair workflow and tuning

The saved split is byte-identical to Midterm: 4,183 training and 1,046 test games. Every candidate uses the same three shuffled training folds with seed 42. Numeric imputation/scaling, one-hot encoding, multi-hot vocabularies and developer catalog-frequency tiers fit each training fold only. Catalog frequency is a proxy, not verified studio size. Twenty-three test rows share Steam app IDs with training. Grouped training CV and the test subset without overlap describe that limitation.

The median dummy provides the absolute-error baseline. OLS tests linear relationships; Ridge adds regularization; a depth-10 tree with at least ten rows per leaf limits tree complexity; 25-neighbour distance-weighted KNN tests local similarity with numeric scaling. Fixed Ridge and ExtraTrees configurations repeat across original, IGDB and IGDB-plus-Steam features. Full model settings and reasons are in outputs/model_specifications.csv.

Random forest searches n_estimators 150/250, minimum leaf rows 3/8 and feature fraction 0.5/0.7: eight configurations. Tree count checks averaging stability, minimum leaf size limits complexity, and feature fraction controls diversity. Boosting searches 15/31 leaves, 20/50 minimum leaf rows and L2 1/10, holding 250 iterations and learning rate 0.07 fixed. It fits absolute-error loss directly in hours. The MLP searches hidden layers (32,) or (64,32) and L2 0.01/0.1. These small networks and penalties control complexity for the available training sample.

{table(pd.DataFrame(settings))}

Each search uses training CV MAE with fold-fitted preprocessing. The winner is locked before test predictions. The selected CV score is a search estimate and can be optimistic because CV is not nested. OLS, Ridge, trees, KNN and forests fit log1p hours. The MLP standardizes that log target. Every reported metric uses inverse-transformed hour predictions. Exact archived Midterm predictions remain a separate historical reference; current original-feature rows are refits through the current pipeline.

## Results on the saved split

MAE is the primary metric because a player interprets error in hours and occasional long labels strongly affect squared error. CV spread is the sample standard deviation over three folds. All baseline and current model comparisons appear in one table.

{model_table}

The exact submitted Midterm Ridge test MAE was 11.81593 h. Its current original-feature refit is {indexed.loc['Ridge original features','test_MAE_hours']:.5f} h; the small difference reflects the current encoding pipeline. Selection uses the new candidates' training CV, while the historical gain above uses the archived predictions. Steam-ID-grouped CV for the selected model is {manifest['grouped_cv_MAE_mean_hours']:.2f} +/- {manifest['grouped_cv_MAE_std_hours']:.2f} h.

The top two expanded CV means differ by only {gap:.5f} h, compared with fold standard deviations around {ranked.iloc[0].train_cv_MAE_std_hours:.2f} h. The minimum-mean rule resolves this near-tie mechanically. It does not establish a meaningful advantage over the runner-up. A different test ranking does not change that predeclared training-CV selection.

## Value of the additional columns

With unchanged Ridge settings, IGDB fields move test MAE from {indexed.loc['Ridge original features','test_MAE_hours']:.2f} to {indexed.loc['Ridge expanded IGDB','test_MAE_hours']:.2f} h, and adding Steam gives {indexed.loc['Ridge IGDB + Steam','test_MAE_hours']:.2f} h. With unchanged ExtraTrees settings, the corresponding values are {indexed.loc['ExtraTrees original features','test_MAE_hours']:.2f}, {indexed.loc['ExtraTrees expanded IGDB','test_MAE_hours']:.2f} and {indexed.loc['ExtraTrees IGDB + Steam','test_MAE_hours']:.2f} h. Steam has model-dependent additional value after IGDB metadata. The paired intervals and a training-validation permutation check document uncertainty and correlated-feature effects.

![Fixed-estimator feature comparison](../outputs/figures/feature_ablation.png)

Bootstrap resampling uses the same test games in pairs 5,000 times. It describes these fitted models on this cohort and does not cover all training variability or create a fresh independent test.

## Errors across models

The following table shows each principal model's test MAE by actual-duration band. Complete duration and genre tables cover all 13 current models plus the exact archived Ridge. Positive residual means underprediction.

{table(wide)}

![Residuals across the earlier and Endterm models](../outputs/figures/cross_model_residuals.png)

The linear reference smooths coarse metadata relationships. Forests and boosting exploit nonlinear interactions, while the MLP learns a regularized transformed-target representation. The measured bands reveal where those mechanisms help and where every model underpredicts the long tail. Log-target fits favour relative distinctions and common short games. Raw-hour absolute-error boosting also favours conditional medians, so it does not resolve open-ended-game labels.

{tradeoff_text}

{classical_text}

{failure_text}

The next examples include the largest loss against archived Ridge for each ensemble/MLP, plus the selected model's largest failure. They expose regressions as well as aggregate improvement. Values are predictions in hours, not absolute errors.

{table(case_table)}

Only {int(tail.sum())} test games exceed 100 hours, but they contribute {tail_share:.1f}% of selected-model absolute error. Online titles such as Gorilla Tag and Grand Theft Auto Online have very large labels whose interpretation as main-story completion is uncertain. Their raw source values remain unchanged. Large error alone does not justify dropping a row or inventing a correction. Source-coverage, single-player and overlap-free sensitivity tables supplement the full-cohort score and never choose the model.

## PCA, clusters and the earlier model's failures

The full-feature diagnostic mainly separates Steam availability. The main analysis therefore uses IGDB genres, themes, keywords, modes and perspectives, with L2 row normalization to reduce the influence of annotation volume. PCA reduces {cluster['pca_input_features']} semantic features to {cluster['pca_components']} components, retaining {100*cluster['retained_variance']:.1f}% of variance. Training-only KMeans compares k=2..6; k={cluster['selected_k']} has the highest sampled silhouette, {cluster['selected_silhouette']:.3f}. Another seed gives adjusted Rand index {cluster['adjusted_rand_index_second_seed']:.3f}. Neither duration nor test errors determine these settings.

![Training-only PCA projection](../outputs/figures/pca_clusters.png)

The first two components retain {100*cluster['pc1_pc2_variance']:.1f}% of variance. The low silhouette indicates substantial overlap, and stable initialization does not establish natural game classes. Profiles use source genres, themes and modes. The named near-median-duration examples below show actual objects behind the profiles, together with their held-out absolute errors.

{table(example_table)}

{table(cluster_table)}

Cluster {worst_cluster} has the highest archived Ridge MAE in this test comparison ({cw.loc[worst_cluster,'Archived Midterm Ridge']:.2f} h). Its selected-model MAE is {cw.loc[worst_cluster,winner]:.2f} h. This connects the unsupervised grouping to the earlier model's failures. Counts and named cases matter because unequal group sizes and rare extreme labels can dominate a cluster mean. The complete examples also show each cluster's largest earlier-model failure. This is descriptive error analysis after evaluation, with no cluster-based model selection.

In that cluster, {worst_case.title} has a source duration of {worst_case.actual_hours:.2f} h and archived Ridge error {worst_case.midterm_absolute_error:.2f} h. The near-median example {typical_case.title} has duration {typical_case.actual_hours:.2f} h. Their shared cluster membership permits a wide duration range. Coarse genre and semantic groups cannot identify how long an individual campaign takes, particularly when an open-ended-game label dominates the group's mean error.

## MLP learning and comparison

The chosen MLP uses hidden layers {nn['hidden_layer_sizes']}, ReLU/Adam and L2 {nn['alpha']}. A 15% holdback inside the training partition controls early stopping. Training loss uses standardized log-target space. Internal validation R2 is dimensionless and computed on that transformed target. Validation peaks at iteration {peak} with R2 {max(curve):.3f}; by iteration {nn['iterations']}, R2 is {curve[-1]:.3f}. Training loss continues to fall while validation weakens, indicating overfitting after the peak. Early stopping restores the best validation weights. The run stops after {nn['iterations']} of at most {nn['max_iter']} iterations, so the evidence is a validation stopping check rather than proof of a globally converged optimum.

![MLP training and validation learning curves](../outputs/figures/neural_learning.png)

The MLP has CV MAE {nnrow.train_cv_MAE_hours:.2f} +/- {nnrow.train_cv_MAE_std_hours:.2f} h and test MAE {nnrow.test_MAE_hours:.2f} h. The best earlier expanded model by CV is {earlier.model}, with CV MAE {earlier.train_cv_MAE_hours:.3f} h and test MAE {earlier.test_MAE_hours:.2f} h. MLP minus earlier-model test MAE is {nnrow.test_MAE_hours-earlier.test_MAE_hours:+.2f} h. The common table also compares the Midterm-family references. R2, RMSE and median error expose trade-offs, but the prespecified selection rule remains hour MAE.

## Conclusions and Final source plan

Real additional feature types improve this stage comparison, and {winner} has the lowest training CV MAE among expanded candidates. The result answers the player-time-budget question with rough catalog estimates. Low R2 and large errors for long/open-ended titles limit precise per-game predictions. Selective Steam coverage, current metadata, repeated store IDs, inherited inclusion rules and prior test exposure remain concrete limitations. The weak cluster separation also limits any interpretation as distinct game categories.

For Final, use real IGDB descriptions already stored for the same games. If more text is needed, collect summary/storyline from the official /v4/games API and link by the same game_id. Retain attribution, source snapshots and lawful-access evidence. Keep developer credentials in environment variables, observe rate limits and recheck current terms before any new request. This adds text representations while preserving the object cohort.

Descriptions may distinguish campaign structure and open-ended online play inside broad genres. Check and remove explicit duration statements before encoding to prevent target disclosure. Compare structured-only, text-only and combined features, with fold-fitted TF-IDF/IDF and regularized Ridge using the same training folds and hour MAE. Choose vocabulary and regularization by training CV, then score the locked comparison on the saved test and inspect the same long-game/cluster errors. A local app can load the selected pipeline. No text-model gain is claimed before that experiment runs.

## Reproduction, contributions and tools

README provides installation and offline commands. The notebook defaults to RETRAIN=True for Restart/Run All. tools/notebook/execute_notebook.py executes a fresh kernel and compares a complete repeated fit against the command-line experiment. outputs/reproducibility_check.json records numeric differences and source hashes. Raw-source verification, data integrity tests and artifact checks accompany the results. Live API recollection is separate from this fixed-snapshot reproduction.

Syrym is assigned collection/preparation, exploration and PCA/clustering review. Yerdaulet is assigned modelling, tuning, MLP and evaluation review. Both must review conclusions and practise the entire defence. These are assigned tasks; completed individual contributions require factual student confirmation. CONTRIBUTIONS.md records the review tasks.

OpenAI Codex assisted with code, actual experiments, verification, notebook/report/slide preparation and editing. Both students remain responsible for explaining the submitted work independently. The complete ZIP supplies everything shown, because the teacher will not inspect GitHub. Each partner must upload the same complete contents by 23:59 on the day before defence, using the required Endterm_Surname_Name_Group.zip filename format. The spoken presentation is seven minutes, followed by five minutes of questions. Rehearsal and upload are student actions.

## Sources

Supplied Endterm Rubric, Project Guide and Midterm Rubric are included under reference/. The original submission and collection code remain there for comparison.

IGDB API documentation: https://api-docs.igdb.com/

Steam API overview: https://partner.steamgames.com/doc/webapi_overview

Steam API terms: https://steamcommunity.com/dev/apiterms

Saved raw responses, source checks, manifests, parameter-search tables, predictions and error examples support the measured results.
'''
(DOCS/'ENDTERM_REPORT.md').write_text(report,encoding='utf-8')

font='Helvetica'; bold='Helvetica-Bold'
fontdir=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
if (fontdir/'arial.ttf').exists():
    pdfmetrics.registerFont(TTFont('ReportBody',str(fontdir/'arial.ttf')))
    pdfmetrics.registerFont(TTFont('ReportBold',str(fontdir/'arialbd.ttf')))
    font='ReportBody';bold='ReportBold'
pdfmetrics.registerFontFamily(font,normal=font,bold=bold,italic=font,boldItalic=bold)
styles={
 'title':ParagraphStyle('title',fontName=bold,fontSize=23,leading=28,spaceAfter=16,textColor=colors.HexColor('#17343C')),
 'heading':ParagraphStyle('heading',fontName=bold,fontSize=15,leading=19,spaceBefore=14,spaceAfter=8,keepWithNext=True,textColor=colors.HexColor('#176B68')),
 'body':ParagraphStyle('body',fontName=font,fontSize=10,leading=14,spaceAfter=9,allowWidows=0,allowOrphans=0),
 'cell':ParagraphStyle('cell',fontName=font,fontSize=8,leading=10.5),
 'head':ParagraphStyle('head',fontName=bold,fontSize=8,leading=10.5,textColor=colors.white),
 'caption':ParagraphStyle('caption',fontName=font,fontSize=8,leading=11,spaceAfter=10,textColor=colors.HexColor('#53666C'))}
width=A4[0]-84
def rich(value):
    s=html.escape(str(value))
    return re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
story=[]
lines=report.splitlines();i=0
while i<len(lines):
    line=lines[i].strip();i+=1
    if not line: continue
    if line.startswith('|'):
        batch=[line]
        while i<len(lines) and lines[i].strip().startswith('|'):
            batch.append(lines[i].strip());i+=1
        raw=[[s.strip() for s in r.strip('|').split('|')] for r in batch]
        raw=[raw[0]]+raw[2:]
        n=len(raw[0]); weights=[1.0]*n
        if n==6 and raw[0][0]=='Model / features':weights=[2.5,1.6,1,1,.8,1]
        elif n==6 and raw[0][1]=='Game':weights=[.55,1.5,1.9,.65,.85,.85]
        elif n==4 and raw[0][0]=='Search':weights=[1.1,.55,2.9,1.2]
        elif n==3:weights=[1.5,1.7,1.7]
        elif n==6:weights=[2.3,1,1,1,1,1]
        data=[[Paragraph(rich(c),styles['head' if j==0 else 'cell']) for c in row] for j,row in enumerate(raw)]
        t=Table(data,colWidths=[width*w/sum(weights) for w in weights],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#176B68')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),
            ('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D0DADB'))]))
        story.append(KeepTogether([t,Spacer(1,10)]));continue
    image_match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
    if image_match:
        path=(DOCS/image_match[2]).resolve()
        pic=Image(str(path));ratio=width/pic.imageWidth;height=pic.imageHeight*ratio
        pic.drawWidth=width;pic.drawHeight=height
        story.extend([pic,Paragraph(rich(image_match[1]),styles['caption'])]);continue
    if line.startswith('# '):story.append(Paragraph(rich(line[2:]),styles['title']));continue
    if line.startswith('## '):story.append(Paragraph(rich(line[3:]),styles['heading']));continue
    paragraph=line
    while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![')):
        paragraph+=' '+lines[i].strip();i+=1
    story.append(Paragraph(rich(paragraph),styles['body']))
def decorate(canvas,doc):
    canvas.saveState();canvas.setFont(font,8);canvas.setFillColor(colors.HexColor('#60747B'))
    canvas.drawString(42,A4[1]-28,'Video game completion time prediction   Endterm')
    canvas.drawString(42,24,'Syrym Shadiyarbek and Yerdaulet Akhanov')
    canvas.drawRightString(A4[0]-42,24,str(doc.page));canvas.restoreState()
SimpleDocTemplate(str(DOCS/'Endterm_Report.pdf'),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=49,bottomMargin=42,
    title='Video game completion time prediction Endterm',author='Syrym Shadiyarbek and Yerdaulet Akhanov').build(story,onFirstPage=decorate,onLaterPages=decorate)
print('Markdown report and PDF built from measured results.')
