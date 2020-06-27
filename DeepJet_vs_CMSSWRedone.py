#!/usr/bin/env python
import ROOT as rt
from DataFormats.FWLite import Events,Handle
import itertools as it
from ROOT import btagbtvdeep
from root_numpy import root2array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file evaluated with DeepJet framework
deepjet_predict = "test_sample/prediction_new.npy"
deepjet_features = "test_sample/features.root"
deepjet_features = "test_sample/output_1.root"

stop = 2000

df_deepjet = pd.DataFrame()
l = np.load(deepjet_predict)
probs = ['probQCD', 
         'probHbb']
for i in range(len(probs)):
    df_deepjet[probs[i]] = l[:stop,i]

features = ["fj_pt", "fj_eta", "fj_phi", "fj_sdmass", 'npv','n_tracks', 'n_sv','event_no']
features_arr = root2array(deepjet_features, stop=stop, branches=features, treename="deepntuplizer/tree")

for feat in features:
    df_deepjet[feat] = features_arr[feat]
    
cpf_vars = ['track_ptrel', 'track_erel','track_phirel',  'track_etarel', 'track_deltaR', 'track_drminsv', 
            'track_drsubjet1', 'track_drsubjet2', 'track_dz', 'track_dzsig', 'track_dxy', 'track_dxysig',
           'track_normchi2', 'track_quality', 'track_dptdpt', 'track_detadeta', 'track_dphidphi', 'track_dxydxy',
           'track_dzdz','track_dxydz', 'track_dphidxy', 'track_dlambdadz', 'trackBTag_EtaRel', 'trackBTag_PtRatio',
            'trackBTag_PParRatio', 'trackBTag_Sip2dVal', 'trackBTag_Sip2dSig', 'trackBTag_Sip3dVal', 
            'trackBTag_Sip3dSig', 'trackBTag_JetDistVal']

sv_vars = ['sv_ptrel', 'sv_erel', 'sv_phirel', 'sv_etarel', 'sv_deltaR', 'sv_pt', 'sv_mass', 'sv_ntracks', 
           'sv_normchi2','sv_dxy', 'sv_dxysig', 'sv_d3d', 'sv_d3dsig', 'sv_costhetasvpv']

features = [(var,0,60) for var in cpf_vars]
features += [(var,0,5) for var in sv_vars]

features_arr = root2array(deepjet_features, stop=stop, branches=features, treename="deepntuplizer/tree")

maxvar = 60
for var in cpf_vars:
    arr_2d = features_arr[var]
    for c_n in range(0,maxvar):
        df_deepjet['{}_{}'.format(var,c_n)] = arr_2d[:,c_n]        
      
maxvar = 5
for var in sv_vars:
    arr_2d = features_arr[var]
    for c_n in range(0,maxvar):
        df_deepjet['{}_{}'.format(var,c_n)] = arr_2d[:,c_n]
        
 
df_deepjet.sort_values(['event_no', 'fj_pt'], ascending=[True, False], inplace=True)
df_deepjet.reset_index(drop=True)
print(df_deepjet)

# jets redone from AOD using CMSSW TF modules
cmssw_miniaod = "test_higgs_in_MINIAODSIM.root"

jetsLabel = "selectedUpdatedPatJets"

from RecoBTag.ONNXRuntime.pfHiggsInteractionNet_cff import _pfHiggsInteractionNetTagsProbs as disc_names

jet_pt = "fj_pt"
jet_eta = "fj_eta"

c_numbers = ['event_n']

c_cmssw = { d_name : []  for d_name in disc_names + [jet_pt, jet_eta] + c_numbers }
jetsHandle = Handle("std::vector<pat::Jet>")
cmssw_evs = Events(cmssw_miniaod)

max_n_jets = 1000000
max_n_events = 500000
n_jets = 0

for i, ev in enumerate(cmssw_evs):
    event_number = ev.object().id().event()
    #if event_number > max_n_events: continue    
    if (n_jets >= max_n_jets): break
    ev.getByLabel(jetsLabel, jetsHandle)
    jets = jetsHandle.product()
    for i_j,j in enumerate(jets):
        uncorr = j.jecFactor("Uncorrected")
        ptRaw = j.pt()*uncorr
        if ptRaw < 200.0 or abs(j.eta()) > 2.4: continue
        if (n_jets >= max_n_jets): break
        c_cmssw["event_n"].append(event_number)
        c_cmssw[jet_pt].append(ptRaw)
        c_cmssw[jet_eta].append(j.eta())
        discs = j.getPairDiscri()
        for d in discs:
            if d.first in disc_names:
                c_cmssw[d.first].append(d.second)
        n_jets +=1
        
df_cmssw = pd.DataFrame(c_cmssw)
df_cmssw.sort_values(['event_n', jet_pt], ascending=[True, False], inplace=True)
df_cmssw.reset_index(drop=True)
print(df_cmssw)

# merging the data frames efficiently
mergeDf = pd.merge(df_deepjet, df_cmssw, left_on= ['event_no','fj_eta'], right_on= ['event_n','fj_eta'], how='inner')
mergeDf.sort_values(['event_n', 'fj_pt_x'], ascending=[True, False], inplace=True)
mergeDf.reset_index(drop=True)
print(mergeDf)

branch_names_deepjet = probs
branch_names_cmssw = disc_names

fig, axs = plt.subplots(1,2,figsize=(20,7))

cmap = plt.cm.viridis
cmap.set_under('w',1)

n_bins = 100
bins = np.linspace(0.,1.,n_bins)
for i,ax in enumerate(axs.flatten()):
    deepjet_col = branch_names_deepjet[i]
    cmssw_col = branch_names_cmssw[i]
    
    ax.set_title(cmssw_col+" - {} bins - jet pt > 200".format(n_bins))
    ax.set_xlabel("DeepJet Framework over MINIAOD", color = "green")
    ax.set_ylabel("CMSSW over MINIAOD", color = "red")
    ax.tick_params(axis='x', colors='green')
    ax.tick_params(axis='y', colors='red') 

    hist_res = ax.hist2d(mergeDf[deepjet_col],mergeDf[cmssw_col],bins=bins,
                         vmin=1,cmap=cmap)

#fig.savefig('corr.pdf')
fig.savefig('corr.png')

fig, ax = plt.subplots(figsize=(10, 7))
plt.hist((mergeDf[deepjet_col]-mergeDf[cmssw_col]), bins=np.linspace(-0.5, 0.5, 100))
#fig.savefig('res.pdf')
fig.savefig('res.png')



