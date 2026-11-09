import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const HERE=path.dirname(fileURLToPath(import.meta.url));
const ROOT=path.resolve(HERE,'../..');
const BUILD=path.join(HERE,'.build');
const OUTPUT=path.join(BUILD,'validated');
const RUNTIME=process.env.RUNTIME_NODE_MODULES || process.env.CODEX_RUNTIME_NODE_MODULES;
if(!process.env.PRESENTATION_SKILL_DIR || !process.env.RUNTIME_PYTHON)throw new Error('Set PRESENTATION_SKILL_DIR and RUNTIME_PYTHON, or use build.ps1, so chart workbooks and validation are included.');
const api=RUNTIME?await import(pathToFileURL(path.join(RUNTIME,'@oai/artifact-tool/dist/artifact_tool.mjs')).href):await import('@oai/artifact-tool');
const {Presentation,PresentationFile,FileBlob}=api;
await fs.mkdir(OUTPUT,{recursive:true});
await fs.mkdir(path.join(BUILD,'rendered'),{recursive:true});
function parseCSV(s){let rows=[],row=[],cell='',quoted=false;for(let i=0;i<s.length;i++){const c=s[i];if(c==='"'){if(quoted&&s[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}else if(c===','&&!quoted){row.push(cell);cell='';}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&s[i+1]==='\n')i++;row.push(cell);if(row.some(v=>v!==''))rows.push(row);row=[];cell='';}else cell+=c;}if(cell||row.length){row.push(cell);rows.push(row);}const heads=rows.shift();return rows.map(r=>Object.fromEntries(heads.map((h,i)=>[h,r[i]??''])));}
const csv=async name=>parseCSV(await fs.readFile(path.join(ROOT,name),'utf8'));
const json=async name=>JSON.parse(await fs.readFile(path.join(ROOT,name),'utf8'));
const compare=await csv('outputs/model_comparison.csv');
const coverage=await csv('data/clean/feature_coverage.csv');
const experiment=await json('outputs/experiment_manifest.json');
const clusters=await json('outputs/clustering_manifest.json');
const neural=await json('outputs/neural_network_diagnostics.json');
const quality=await json('data/clean/enrichment_report.json');
const selection=await json('outputs/model_selection.json');
const row=n=>{const x=compare.find(r=>r.model===n);if(!x)throw new Error('Missing model '+n);return x;};
const winner=row(selection.winner);
if(row('Random forest').tuned.toLowerCase()!=='true' || row('Tuned gradient boosting').tuned.toLowerCase()!=='true')throw new Error('Both ensemble searches must finish before building the slides');
if(experiment.best_cv_model!==selection.winner)throw new Error('Experiment manifest and selected model disagree');
const ensembleGap=Math.abs(+row('Random forest').train_cv_MAE_hours - +row('Tuned gradient boosting').train_cv_MAE_hours);
const rankingNote=ensembleGap<0.01?' (near tie)':'';
const cv=r=>`${(+r.train_cv_MAE_hours).toFixed(3)} ± ${(+r.train_cv_MAE_std_hours).toFixed(3)}`;
const f=(n,d=2)=>Number(n).toFixed(d);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#17343C',teal:'#176B68',orange:'#B76A37',muted:'#70838C',grid:'#E4E9EB',pale:'#F3F7F7'};
function text(s,t,x,y,w,h,size=28,color=C.ink,bold=false){const v=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});v.text=t;v.text.style={typeface:'Arial',fontSize:size,color,bold};return v;}
function slide(title){const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,65,43,1160,75,44,C.ink,true);text(s,`${String(p.slides.items.length).padStart(2,'0')} / 09`,1140,675,100,30,18,C.muted);return s;}
function note(s,body,source){s.speakerNotes.text=body+'\n\nProject evidence: '+source+'. Included in the submission ZIP.';}
function table(s,values,x,y,w,h,widths,font=24){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});t.borders.assign({style:'solid',fill:C.grid,width:0.7});for(let i=0;i<values.length;i++)for(let j=0;j<values[0].length;j++){const c=t.getCell(i,j);c.fill=i===0?C.teal:'#FFFFFF';c.text.style={typeface:'Arial',fontSize:font,color:i===0?'#FFFFFF':C.ink,bold:i===0};}return t;}
function chart(s,type,cfg){for(const series of cfg.series||[]){for(const key of ['values','xValues']){if(series[key])series[key]=series[key].map(n=>n===null?null:Number(Number(n).toPrecision(12)));}}const q=s.charts.add(type,{chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',hasLegend:false,...cfg});for(const a of [q.xAxis,q.yAxis]){a.textStyle.typeface='Arial';a.textStyle.fontSize=20;a.textStyle.fill=C.ink;if(a.title?.textStyle){a.title.textStyle.typeface='Arial';a.title.textStyle.fontSize=21;}}q.legend.textStyle.typeface='Arial';q.legend.textStyle.fontSize=20;q.legend.textStyle.fill=C.ink;q.dataLabels.textStyle.typeface='Arial';q.dataLabels.textStyle.fontSize=18;q.dataLabels.textStyle.fill=C.ink;return q;}
const axis={majorGridlines:{style:'solid',fill:C.grid,width:1},min:0};
function errorValue(rows,predicate){const r=rows.find(predicate);if(!r||r.MAE_hours===''||!Number.isFinite(+r.MAE_hours))throw new Error('Missing observed error summary');return +r.MAE_hours;}
const sourceTerms='data/raw/source_checks; docs/DATA_CARD.md';
// 1. Research question and intended use.
{
const s=slide('');s.background.fill=C.ink;
text(s,'Predicting game\ncompletion time',78,125,1120,180,64,'#FFFFFF',true);
text(s,'Can game metadata help a player plan a time budget?',82,342,1110,90,34,'#93D2C3');
text(s,'Machine Learning Algorithms Endterm, SE2424',82,447,1110,48,28,'#FFFFFF');
text(s,'Syrym Shadiyarbek\nYerdaulet Akhanov',82,553,800,88,27,'#FFFFFF');
text(s,'09 November 2026',978,619,245,32,21,'#C0CFD1');
note(s,'Syrym. 25 seconds. We predict the main-story hours of a game, so a player can choose one that fits their available time. We test whether real new metadata and the Endterm methods improve our Midterm estimate.','docs/ENDTERM_REPORT.md; data/clean/games_enriched.csv');
}
// 2. Observed feature coverage and the teacher feedback.
{
const s=slide('Real feature expansion on the same 5,229 games');
const fields=[['USD regular price','steam_regular_price_usd'],['Steam user tags','steam_tags'],['Game modes','igdb_game_modes'],['Player perspectives','igdb_player_perspectives'],['Gameplay keywords','igdb_keywords'],['Gameplay themes','igdb_themes'],['Genres','genres_text'],['Release year','release_year']];
const full=fields.map(([_,name])=>coverage.find(r=>r.column===name));
chart(s,'bar',{position:{left:80,top:138,width:1110,height:370},categories:fields.map(r=>r[0]),series:[{name:'Endterm',values:full.map(r=>+r.coverage_percent),fill:C.teal},{name:'Midterm',values:fields.map((r,i)=>i<6?0:+full[i].midterm_coverage_percent),fill:C.muted}],barOptions:{direction:'bar',grouping:'clustered',gapWidth:60},hasLegend:true,legend:{position:'bottom'},xAxis:{majorGridlines:null},yAxis:{...axis,max:100,majorUnit:20,title:'Observed coverage (%)'}});
text(s,`${quality.usd_price_rows.toLocaleString('en-US')} observed USD prices and ${quality.steam_tag_rows.toLocaleString('en-US')} games with Steam tags`,82,536,1120,45,28,C.teal,true);
text(s,'The ZIP contains collection code, raw responses, conclusions and defence notes.',82,599,1120,60,26);
note(s,'Syrym. 45 seconds. The Midterm feedback identified empty columns and missing collection code and conclusions. We kept the same games, added real feature types and included the full source snapshots and code in the archive. Missing prices stay missing. Prices are current US-store quotes, so they do not represent launch prices.','data/clean/feature_coverage.csv; src/collect.py; reference/midterm_source/src/collect_igdb.py; '+sourceTerms);
}
// 3. Continuity and train-only validation.
{
const s=slide('Saved labels, split and collection code');
text(s,'4,183 training games',82,143,550,62,39,C.teal,true);
text(s,'1,046 test games',690,143,510,62,39,C.teal,true);
table(s,[['Step','Evidence in the ZIP'],['Collect real observations','src/collect.py and reference/midterm_source/src/'],['Preserve game IDs and target','IGDB main-story hours from the Midterm snapshot'],['Preserve the saved split','data/clean/fixed_split.csv matches the original bytes'],['Fit preprocessing inside CV','Imputation, scaling, vocabularies and studio frequency']],75,235,1130,270,[370,760],24);
text(s,'Three training folds, seed 42. Both ensemble searches use training CV MAE.',83,550,1120,70,29);
text(s,'The test was already seen at Midterm. It supports comparison between stages.',83,623,1120,38,23,C.muted);
note(s,'Syrym. 45 seconds. The regression target and the saved split are unchanged. Collection code is included directly in the ZIP, including the original IGDB collector. Three training folds fit every preprocessing step again. We exclude IDs, descriptions and alternative completion times. Test outcomes were already observed at Midterm, so the same test measures stage comparisons.','src/collect.py; reference/midterm_source/src/; data/clean/fixed_split.csv; src/features.py; outputs/experiment_plan.json');
}
// 4. Controlled feature ablation.
{
const s=slide('What the additional feature types contribute');
const groups=['Original features','Expanded IGDB','IGDB + Steam'];
chart(s,'bar',{position:{left:82,top:147,width:1110,height:411},categories:groups,series:[{name:'Ridge',values:['Ridge original features','Ridge expanded IGDB','Ridge IGDB + Steam'].map(n=>+row(n).test_MAE_hours),fill:C.teal},{name:'ExtraTrees',values:['ExtraTrees original features','ExtraTrees expanded IGDB','ExtraTrees IGDB + Steam'].map(n=>+row(n).test_MAE_hours),fill:C.orange}],barOptions:{direction:'column',grouping:'clustered',gapWidth:80},hasLegend:true,legend:{position:'bottom'},yAxis:{...axis,max:14,title:'Test MAE (hours)'},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00'}});
text(s,'Estimator settings stay fixed within each series. Lower MAE is better.',83,594,1120,43,27);
text(s,'Steam helps ExtraTrees here. Its additional gain for Ridge is unclear.',83,639,1120,31,23,C.muted);
note(s,'Yerdaulet. 45 seconds. These comparisons separate new features from changing the model. Ridge and ExtraTrees keep their settings across all three feature sets. IGDB fields help both. Steam further improves ExtraTrees, while Ridge shows no clear additional gain. Real useful columns can overlap in the information they carry.','outputs/model_comparison.csv; outputs/paired_bootstrap.json');
}
// 5. Model comparison and the two tuned ensembles.
{
const s=slide('Training-CV selection and tuned ensembles');
const names=['Median baseline','Ridge original features','Ridge IGDB + Steam','ExtraTrees IGDB + Steam','Random forest','Tuned gradient boosting','Tuned neural network'];
const labels=['Median baseline','Ridge original (refit)','Ridge expanded','ExtraTrees expanded','Tuned random forest','Tuned boosting','Tuned MLP'];
table(s,[['Model','CV MAE ± SD (h)','Test MAE (h)','Test R²'],...names.map((n,i)=>{const r=row(n);return[labels[i],cv(r),f(r.test_MAE_hours),f(r.test_R2,3)];})],65,145,1150,392,[440,280,210,220],24);
const plan=await json('outputs/experiment_plan.json');
const count=n=>Object.values(plan.grids[n]||{}).reduce((a,b)=>a*b.length,1);
text(s,`RF: ${count('Random forest')} settings. Boosting: ${count('Tuned gradient boosting')} settings. MLP: ${count('Tuned neural network')} settings.`,82,565,1120,42,25);
text(s,`Selected by training CV: ${selection.winner}${rankingNote}`,82,613,1120,43,29,C.teal,true);
note(s,'Yerdaulet. 55 seconds. Random forest and boosting each have a documented training-only search. We compare them with the original Ridge and the neural network using MAE in hours. The selected model has the lowest mean training CV MAE. The report and notebook contain every reference model and CV fold result. Test scores support evaluation, not model selection.','outputs/experiment_plan.json; outputs/search/; outputs/cv_fold_scores.csv; outputs/model_comparison.csv');
}
// 6. Cross-model errors.
{
const s=slide('Where different models fail');
const errors=await csv('outputs/all_model_error_by_duration.csv');
const names=['Archived Midterm Ridge','Random forest','Tuned gradient boosting','Tuned neural network'];
const colors=[C.muted,'#389993',C.teal,C.orange];
const bands=[...new Set(errors.map(r=>r.duration_band))];
const field='MAE_hours';
chart(s,'bar',{position:{left:75,top:145,width:1120,height:385},categories:bands,series:names.map((name,i)=>({name:['Midterm Ridge','Tuned RF','Boosting','MLP'][i],values:bands.map(b=>errorValue(errors,r=>r.model===name&&r.duration_band===b)),fill:colors[i]})),barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:true,legend:{position:'bottom'},yAxis:{...axis,title:'MAE by actual duration (hours)'}});
const shortRF=errorValue(errors,r=>r.model==='Random forest'&&r.duration_band==='0-5 h');
const shortGB=errorValue(errors,r=>r.model==='Tuned gradient boosting'&&r.duration_band==='0-5 h');
const middleRF=errorValue(errors,r=>r.model==='Random forest'&&r.duration_band==='>5-20 h');
const middleGB=errorValue(errors,r=>r.model==='Tuned gradient boosting'&&r.duration_band==='>5-20 h');
text(s,'RF vs boosting: RF has lower MAE at 0-5 h, boosting at >5-20 h.',82,566,1120,45,29,C.teal,true);
text(s,`RF / boosting MAE: ${f(shortRF)} / ${f(shortGB)} h (short), ${f(middleRF)} / ${f(middleGB)} h (5-20 h).`,82,624,1120,43,24);
note(s,'Yerdaulet. 50 seconds. This compares errors across models on the same test games. It shows why the average error remains much higher than the typical error. We retain the long labels and inspect the real game examples instead of deleting difficult cases. The notebook compares the actual predictions of each model on those games.','outputs/all_model_error_by_duration.csv; outputs/all_model_error_examples.csv; outputs/test_predictions.csv; outputs/figures/cross_model_residuals.png');
}
// 7. Unsupervised structure connected to model failure.
{
const s=slide('PCA groups help describe model errors');
const errors=await csv('outputs/cluster_error_summary.csv');
const names=['Archived Midterm Ridge',selection.winner];
const groups=[...new Set(errors.map(r=>r.cluster))];
const field='MAE_hours';
chart(s,'bar',{position:{left:65,top:152,width:750,height:390},categories:groups.map(c=>'Cluster '+c),series:names.map((name,i)=>({name:i===0?'Midterm Ridge':'Selected Endterm',values:groups.map(c=>errorValue(errors,r=>r.model===name&&r.cluster===c)),fill:i===0?C.muted:C.teal})),barOptions:{direction:'column',grouping:'clustered',gapWidth:80},hasLegend:true,legend:{position:'bottom'},yAxis:{...axis,title:'Test MAE (hours)'}});
text(s,`${clusters.pca_components} PCA components\n${f(100*clusters.retained_variance,1)}% variance retained`,857,180,345,110,29,C.teal,true);
text(s,`k = ${clusters.selected_k}\nSilhouette ${f(clusters.selected_silhouette,3)}\nSeed ARI ${f(clusters.adjusted_rand_index_second_seed,3)}`,857,330,345,160,27);
const examples=await csv('outputs/cluster_error_examples.csv');
const example=examples.find(e=>e.game_id==='10733');
if(!example)throw new Error('Missing real cluster example');
text(s,`${example.title} (cluster ${example.cluster}): ${example.igdb_themes.replace(/\|/g,', ')}, ${example.igdb_game_modes}.`,82,572,1120,42,27);
text(s,`Actual ${f(example.actual_hours)} h. Midterm ${f(example['Archived Midterm Ridge'])} h. Endterm ${f(example.selected_model_prediction)} h. Groups overlap (silhouette ${f(clusters.selected_silhouette,3)}).`,82,627,1120,43,23,C.muted);
note(s,'Syrym. 50 seconds. PCA and KMeans fit only the training predictors. L2 normalization reduces the effect of annotation counts. We now connect the resulting groups to held-out errors of the original Ridge and the selected model. The notebook names actual games within each cluster. Low silhouette means the groups overlap, even though the second seed is stable. The error comparison is descriptive and does not choose a model.','src/cluster.py; outputs/clustering_manifest.json; outputs/cluster_error_summary.csv; outputs/cluster_error_examples.csv');
}
// 8. Both learning curves are visible.
{
const s=slide('MLP training and validation progress');
const categories=neural.loss_curve.map((_,i)=>String(i+1));
chart(s,'scatter',{position:{left:65,top:170,width:550,height:285},categories,series:[{name:'Training loss',xValues:categories.map(Number),values:neural.loss_curve,line:{fill:C.teal,width:3}}],yAxis:{...axis,title:'Training loss'},xAxis:{min:0,max:neural.iterations,majorUnit:5,title:'Iteration'},scatterOptions:{style:'line'}});
chart(s,'scatter',{position:{left:660,top:170,width:550,height:285},categories,series:[{name:'Internal validation R²',xValues:categories.map(Number),values:neural.internal_validation_R2_curve,line:{fill:C.orange,width:3}}],yAxis:{title:'Validation R²',majorGridlines:axis.majorGridlines},xAxis:{min:0,max:neural.iterations,majorUnit:5,title:'Iteration'},scatterOptions:{style:'line'}});
const peak=neural.internal_validation_R2_curve.indexOf(Math.max(...neural.internal_validation_R2_curve))+1;
text(s,`Hidden layers ${neural.hidden_layer_sizes.join(' / ')}. L2 ${neural.alpha}. Best validation at iteration ${peak}.`,82,496,1120,45,27,C.teal,true);
text(s,`Training stops at ${neural.iterations} iterations and restores the best validation weights.`,82,548,1120,42,26);
text(s,`MLP test MAE ${f(row('Tuned neural network').test_MAE_hours)} h. Selected model ${f(winner.test_MAE_hours)} h.`,82,599,1120,40,28);
text(s,'Training loss uses standardized log targets. Validation R² uses the same transformed target.',82,646,1050,27,21,C.muted);
note(s,'Yerdaulet. 50 seconds. The network uses ReLU and Adam, two hidden layers and L2 regularization. Training loss keeps falling, but internal validation peaks earlier and then deteriorates. This is evidence of overfitting after the peak. Early stopping restores the best validation weights. Both curves use transformed-target units. The final comparison uses MAE in hours and the same test games.','outputs/neural_network_diagnostics.json; outputs/search/neural_network.csv; outputs/model_comparison.csv');
}
// 9. Conclusions and a concrete source plan.
{
const s=slide('Conclusions and the Final experiment');
const reduction=100*(+row('Median baseline').test_MAE_hours- +winner.test_MAE_hours)/+row('Median baseline').test_MAE_hours;
text(s,`${f(winner.test_MAE_hours)} h mean absolute test error\n${f(reduction,1)}% lower than the median baseline`,82,143,1110,115,42,C.teal,true);
text(s,'Real metadata improves rough estimates. Long and open-ended labels remain difficult.',82,295,1110,82,31);
text(s,'Final source: IGDB descriptions linked by the same game ID.\nCompare structured inputs, TF-IDF text and their combination using training CV MAE.',82,400,1110,119,29);
text(s,'Limits: current metadata, selective Steam coverage, uncertain labels and prior test exposure.',82,556,1110,75,26,C.muted);
note(s,'Both. 55 seconds. The added metadata improves estimates, but the errors do not support a precise prediction for every game. For Final we will encode the existing IGDB descriptions, linked by game ID, using TF-IDF fitted within training folds. The saved source-access checks and API collection code support access. We will compare structured-only, text-only and combined inputs under the same CV MAE rule. Descriptions may capture campaign detail missing from broad genres. Both partners should describe their own reviewed work. OpenAI Codex assisted with code, experiments and preparation of these materials.','docs/ENDTERM_REPORT.md; docs/CONTRIBUTIONS.md; docs/DEFENSE_GUIDE.md; '+sourceTerms);
}
if(p.slides.items.length!==9)throw new Error('Slide count');
const spoken=await json('docs/defense_notes.json');
if(spoken.length!==9 || spoken.reduce((n,r)=>n+r.seconds,0)!==420)throw new Error('Speaker timing must total seven minutes');
for(let i=0;i<9;i++){
 if(/\[\[|\]\]/.test(spoken[i].text))throw new Error('Unresolved speaker note on slide '+(i+1));
 const saved=String(p.slides.items[i].speakerNotes.text);
 const evidence=saved.slice(saved.indexOf('Project evidence:'));
 p.slides.items[i].speakerNotes.text=`${spoken[i].speaker}. ${spoken[i].seconds} seconds.\n\n${spoken[i].text}\n\n${evidence}`;
}
const archived=await csv('reference/midterm/outputs/model_comparison.csv');
p.slides.items[4].speakerNotes.append('\nArchived Midterm Ridge test MAE: '+archived.find(r=>r.model==='Ridge Regression').MAE_hours+' h. The table original Ridge row is the controlled Endterm refit.');
p.slides.items[4].speakerNotes.append('\nSelected RF parameters: '+row('Random forest').best_parameters+'\nSelected boosting parameters: '+row('Tuned gradient boosting').best_parameters);
p.slides.items[8].speakerNotes.append('\nAI use: OpenAI Codex assisted with collection and modelling code, experiments, checks and preparation. Individual review and defence remain the students\' responsibility. See docs/CONTRIBUTIONS.md.');
// Make every chart independently editable with real literal data and a workbook snapshot.
const candidate=path.join(BUILD,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,'rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
const final=path.join(OUTPUT,`Endterm_Defense-${Date.now()}.pptx`);
if(process.env.PRESENTATION_SKILL_DIR){
 const skill=process.env.PRESENTATION_SKILL_DIR;
 const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
 const reference=path.join(HERE,'reference/Previous_Endterm_Defense.pptx');
 await finalizePresentation({workspaceDir:HERE,candidatePath:candidate,finalPath:final,pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','3','--require-native-table-slide','5'],explicitTotalSlideCount:9,requiredNativeTableOwnerSlides:[3,5],requiredNativeChartOwnerSlides:[2,4,6,7,8],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:['Arial'],referencePath:reference,referenceSha256:createHash('sha256').update(await fs.readFile(reference)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,path.basename(final)+'.validation.json')});
}else await fs.copyFile(candidate,final);
await fs.copyFile(final,path.join(ROOT,'docs/Endterm_Defense.pptx'));
const reviewed=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<reviewed.slides.items.length;i++){
 const b=await reviewed.export({slide:reviewed.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,'rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
console.log('Saved and rendered the validated nine-slide deck. Build directory:',BUILD);
