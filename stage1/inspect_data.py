from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
from collections import defaultdict
from scipy.stats import ks_2samp
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--data-dir', default=str(Path(__file__).resolve().parent.parent/'data'))
p.add_argument('--output-dir', default=str(Path(__file__).parent))
a = p.parse_args()
src, out = Path(a.data_dir), Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
names = ['train.csv','test.csv','txs_edgelist.csv','sample_submission.csv']
train, test, edges, sample = [pd.read_csv(src/n) for n in names]
raw_labels = pd.read_csv(src/'train.csv', usecols=['label'], keep_default_na=False)['label']
features = [c for c in train if c.startswith('feat_')]
result = {'files': {}, 'raw_label_values': raw_labels.value_counts().to_dict()}
for name, df in zip(names, [train,test,edges,sample]):
    result['files'][name] = {'shape':list(df.shape), 'sha256':hashlib.sha256((src/name).read_bytes()).hexdigest(), 'missing_by_column':{k:int(v) for k,v in df.isna().sum().items() if v}, 'duplicate_rows':int(df.duplicated().sum())}
for name, df in [('train',train),('test',test)]:
    result[name] = {'duplicate_txIds':int(df.txId.duplicated().sum()), 'time_steps':sorted(df.time_step.unique().tolist()), 'missing_features':int(df[features].isna().sum().sum()), 'infinite_features':int(np.isinf(df[features].to_numpy()).sum()), 'constant_features':df[features].columns[df[features].nunique()<=1].tolist(), 'duplicate_feature_rows':int(df.duplicated(features).sum())}
result['feature_count'] = len(features)
result['schema_difference'] = {'train_only':sorted(set(train)-set(test)), 'test_only':sorted(set(test)-set(train))}
result['overlap_txIds'] = len(set(train.txId)&set(test.txId))
result['labels'] = {'licit':int(train.label.eq(0).sum()),'illicit':int(train.label.eq(1).sum()),'unknown':int(train.label.isna().sum()),'illicit_rate_labeled':float(train.label.dropna().mean())}
times = train.assign(licit=train.label.eq(0), illicit=train.label.eq(1), unknown=train.label.isna()).groupby('time_step').agg(total=('txId','size'),licit=('licit','sum'),illicit=('illicit','sum'),unknown=('unknown','sum'))
times['labeled'] = times.licit + times.illicit
times['illicit_rate_labeled'] = times.illicit / times.labeled
times.to_csv(out/'train_by_time_step.csv')
test.groupby('time_step').size().rename('total').to_csv(out/'test_by_time_step.csv')
allnodes = pd.concat([train[['txId','time_step']],test[['txId','time_step']]])
node_ids = set(allnodes.txId)
graph_ids = set(edges.txId1)|set(edges.txId2)
indeg, outdeg = edges.txId2.value_counts(), edges.txId1.value_counts()
result['graph'] = {'edges':len(edges),'unique_nodes':len(graph_ids),'unavailable_endpoint_nodes':len(graph_ids-node_ids),'edges_both_endpoints_available':int((edges.txId1.isin(node_ids)&edges.txId2.isin(node_ids)).sum()),'self_loops':int(edges.txId1.eq(edges.txId2).sum())}
for name,df in [('train',train),('test',test)]:
    covered = df.txId.isin(graph_ids)
    result['graph'][name] = {'covered':int(covered.sum()),'total':len(df),'coverage':float(covered.mean()),'absent':int((~covered).sum())}
mapping = allnodes.set_index('txId').time_step
t1,t2 = edges.txId1.map(mapping), edges.txId2.map(mapping)
known = t1.notna()&t2.notna()
result['graph']['known_endpoint_time_comparison'] = {'same_step':int((known&t1.eq(t2)).sum()),'forward_step':int((known&(t1<t2)).sum()),'backward_step':int((known&(t1>t2)).sum()),'unknown':int((~known).sum())}
result['submission'] = {'columns':sample.columns.tolist(),'same_length_as_test':len(sample)==len(test),'index_exactly_matches_test':bool(sample['index'].equals(test['index'])),'unique_test_index':bool(test['index'].is_unique),'sequential_index':bool(np.array_equal(test['index'],np.arange(len(test))))}
result['degree_summary'] = {}
for name,df in [('train',train),('test',test)]:
    degrees = pd.DataFrame({'in_degree':df.txId.map(indeg).fillna(0),'out_degree':df.txId.map(outdeg).fillna(0)})
    result['degree_summary'][name] = degrees.describe(percentiles=[.5,.9,.99]).to_dict()
dups = train[train.duplicated(features, keep=False)]
result['duplicate_feature_details'] = {'rows_in_groups':len(dups),'conflicting_labeled_groups':int((dups.groupby(features).label.nunique()>1).sum())}
result['graph']['train_to_test_edges'] = int((edges.txId1.isin(train.txId)&edges.txId2.isin(test.txId)).sum())
result['graph']['test_to_train_edges'] = int((edges.txId1.isin(test.txId)&edges.txId2.isin(train.txId)).sum())
result['graph']['train_edges_with_missing_neighbor'] = int(((edges.txId1.isin(train.txId)&~edges.txId2.isin(node_ids)) | (edges.txId2.isin(train.txId)&~edges.txId1.isin(node_ids))).sum())
result['graph']['test_edges_with_missing_neighbor'] = int(((edges.txId1.isin(test.txId)&~edges.txId2.isin(node_ids)) | (edges.txId2.isin(test.txId)&~edges.txId1.isin(node_ids))).sum())

# ---------------------------------------------------------------------------
# Additional checks (added after the first review)
# ---------------------------------------------------------------------------
missing_ids = graph_ids - node_ids
train_ids, test_ids = set(train.txId), set(test.txId)
labeled_ids = set(train.txId[train.label.notna()])

# (a) How missing nodes connect: direct test adjacency, missing-missing edges, and
#     which weakly connected components they belong to.
e1_miss, e2_miss = edges.txId1.isin(missing_ids), edges.txId2.isin(missing_ids)
direct = set(edges.txId1[e1_miss & edges.txId2.isin(test_ids)]) | set(edges.txId2[e2_miss & edges.txId1.isin(test_ids)])
adj = defaultdict(list)
for u, v in zip(edges.txId1, edges.txId2):
    adj[u].append(v); adj[v].append(u)
seen, comp_rows = set(), []
for start in missing_ids:
    if start in seen: continue
    stack, comp = [start], []
    seen.add(start)
    while stack:
        x = stack.pop(); comp.append(x)
        for y in adj[x]:
            if y not in seen: seen.add(y); stack.append(y)
    comp = set(comp)
    comp_rows.append({'n_missing': len(comp & missing_ids), 'has_test': bool(comp & test_ids),
                      'has_train': bool(comp & train_ids),
                      'n_test_steps': int(test.time_step[test.txId.isin(comp)].nunique())})
comps = pd.DataFrame(comp_rows)
result['graph']['missing_nodes'] = {
    'total': len(missing_ids),
    'directly_adjacent_to_test': len(direct),
    'missing_to_missing_edges': int((e1_miss & e2_miss).sum()),
    'in_component_with_test': int(comps.loc[comps.has_test, 'n_missing'].sum()),
    'in_component_with_train': int(comps.loc[comps.has_train, 'n_missing'].sum()),
    'in_component_spanning_multiple_test_steps': int(comps.loc[comps.n_test_steps > 1, 'n_missing'].sum()),
    'note': 'Hypothesis only: consistent with unlabeled test-period transactions removed from the files; labels and time steps of missing nodes are not observable.'}

# (b) Neighbour availability: share of each node's neighbours with feature rows, and
#     share that are labeled (train) or test rows.
und = pd.concat([edges.rename(columns={'txId1': 'n', 'txId2': 'o'}), edges.rename(columns={'txId2': 'n', 'txId1': 'o'})])
und['available'] = und.o.isin(node_ids)
und['labeled_or_test'] = und.o.isin(labeled_ids | test_ids)
share = und.groupby('n')[['available', 'labeled_or_test']].mean()
def _share(ids, col): return float(share.loc[share.index.isin(ids), col].mean())
result['graph']['neighbour_availability'] = {
    'labeled_train_mean_share_available': _share(labeled_ids, 'available'),
    'labeled_train_mean_share_labeled_or_test': _share(labeled_ids, 'labeled_or_test'),
    'test_mean_share_available': _share(test_ids, 'available'),
    'note': 'Matching mean coverage does not imply matching missingness patterns; treat labeled-only aggregation as an experiment.'}

# (c) Rows per step: test vs labeled/total train (context for the missing-node hypothesis)
result['rows_per_step'] = {'train_total_mean': float(times.total.mean()), 'train_labeled_mean': float(times.labeled.mean()),
                           'test_mean': float(test.groupby('time_step').size().mean())}

# (d) Feature diagnostics: duplicate columns, train-vs-test drift (labeled train rows vs test),
#     and monotonic relationship of per-step means with time_step.
lab = train[train.label.notna()]
all_x = pd.concat([train[['time_step'] + features], test[['time_step'] + features]])
step_means = all_x.groupby('time_step')[features].mean()
steps = pd.Series(step_means.index, index=step_means.index)
fd = pd.DataFrame({
    'ks_labeled_train_vs_test': [ks_2samp(lab[c], test[c]).statistic for c in features],
    'ks_labeled_vs_unknown_train': [ks_2samp(lab[c], train.loc[train.label.isna(), c]).statistic for c in features],
    'spearman_step_mean_vs_time': step_means.corrwith(steps, method='spearman').values,
    'train_min': train[features].min().values, 'train_max': train[features].max().values,
    'test_share_above_train_max': [(test[c] > train[c].max()).mean() for c in features],
}, index=pd.Index(features, name='feature')).sort_values('ks_labeled_train_vs_test', ascending=False)
fd['time_encoding_candidate'] = (fd.ks_labeled_train_vs_test > 0.9) & (fd.spearman_step_mean_vs_time.abs() > 0.99)
fd.round(4).to_csv(out / 'feature_diagnostics.csv')
dup_cols = pd.concat([train[features], test[features]]).T.duplicated()
result['feature_diagnostics'] = {
    'duplicate_columns': dup_cols[dup_cols].index.tolist(),
    'time_encoding_candidates': fd.index[fd.time_encoding_candidate].tolist(),
    'n_features_ks_above_0_3': int((fd.ks_labeled_train_vs_test > 0.3).sum()),
    'median_ks_labeled_vs_unknown': float(fd.ks_labeled_vs_unknown_train.median()),
    'note': 'Candidates for exclusion only; compare with/without on identical chronological splits in Stage 2.'}

(out/'diagnostics.json').write_text(json.dumps(result,indent=2))
fig,axes = plt.subplots(2,1,figsize=(11,7),layout='constrained')
axes[0].plot(times.index,100*times.illicit_rate_labeled,marker='o',ms=4,color='#b34532')
axes[0].set(ylabel='Illicit among labeled (%)',title='Observed training labels by time step')
axes[1].bar(times.index,times.licit,label='Licit',color='#427b9e')
axes[1].bar(times.index,times.illicit,bottom=times.licit,label='Illicit',color='#b34532')
axes[1].bar(times.index,times.unknown,bottom=times.labeled,label='Unknown',color='#c5ccd0')
counts=test.groupby('time_step').size()
axes[1].bar(counts.index,counts,label='Test (labels withheld)',color='#8066a6')
axes[1].set(xlabel='Time step',ylabel='Transactions',title='Available rows: training and future test periods')
axes[1].legend(ncol=4,fontsize=9)
for ax in axes: ax.grid(axis='y',alpha=.2); ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'time_step_overview.png',dpi=170)
print(json.dumps(result,indent=2))
print(times.to_string())
