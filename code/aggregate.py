#!/usr/bin/env python3
"""Aggregate joint benchmark cases into tables, summaries, and a heatmap."""
from __future__ import annotations
import argparse,csv,json,platform,sys
from pathlib import Path
import numpy as np
import scipy
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

L_VALUES=(4,8,12); M_VALUES=(1,2,4)
ELECTRON_FAMILIES=("cosine","smoothstep"); PHOTON_FAMILIES=("gauss","midpoint")
MIXED_CORNERS=((4,1),(4,4),(12,1),(12,4))

def load_npz(path:Path):
    with np.load(path,allow_pickle=False) as z:
        r=json.loads(str(z['metadata']))
        for k in z.files:
            if k!='metadata': r[k]=np.asarray(z[k])
    return r

def texnum(x:float)->str:
    m,e=f"{x:.3e}".split('e');return rf"{m}\times10^{{{int(e)}}}"

def write_csv(path,rows,fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases-dir',type=Path,required=True);ap.add_argument('--checks-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--figure-dir',type=Path,required=True);a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True);a.figure_dir.mkdir(parents=True,exist_ok=True)
    R={}
    for ef in ELECTRON_FAMILIES:
      for pf in PHOTON_FAMILIES:
       for L in L_VALUES:
        for M in M_VALUES:
         r=load_npz(a.cases_dir/f'{ef}_{pf}_L{L}_M{M}.npz');R[ef,pf,L,M]=r
    rows=[]
    fields=['electronic_family','photon_family','L','M','E_v','E_j','E_src','argmax_v','argmax_j','min_kappa','max_density_tracking_error','max_centered_q_error','max_norm_error','max_top_sector_population','electronic_dimension','boson_dimension','total_dimension','boson_cutoff','stiffness_min','stiffness_max','time_grid_points','time_grid_step']
    for _,r in R.items(): rows.append({k:r[k] for k in fields})
    rows.sort(key=lambda r:(r['electronic_family'],r['photon_family'],int(r['L']),int(r['M'])))
    write_csv(a.output_dir/'joint_analytic_inverse_grid.csv',rows,fields)

    cross=[]
    for ef in ELECTRON_FAMILIES:
      for pf in PHOTON_FAMILIES:
        x=R[ef,pf,4,1]['delta_v_trace']-R[ef,pf,4,4]['delta_v_trace']-R[ef,pf,12,1]['delta_v_trace']+R[ef,pf,12,4]['delta_v_trace']
        cross.append({'electronic_family':ef,'photon_family':pf,'L0':4,'L1':12,'M0':1,'M1':4,'cross_v':float(np.max(np.abs(x)))})
    write_csv(a.output_dir/'joint_analytic_inverse_cross.csv',cross,['electronic_family','photon_family','L0','L1','M0','M1','cross_v'])

    esp=[{'L':L,'M':4,'spread_v':float(np.max(np.abs(R['cosine','gauss',L,4]['delta_v_trace']-R['smoothstep','gauss',L,4]['delta_v_trace'])))} for L in L_VALUES]
    psp=[]
    for M in M_VALUES:
        psp.append({'L':12,'M':M,'spread_v':float(np.max(np.abs(R['cosine','gauss',12,M]['delta_v_trace']-R['cosine','midpoint',12,M]['delta_v_trace']))),'spread_j':float(np.max(np.abs(R['cosine','gauss',12,M]['j_finite_trace']-R['cosine','midpoint',12,M]['j_finite_trace'])))})

    corner_checks=[]
    for L,M in MIXED_CORNERS:
        base=R['cosine','gauss',L,M]
        cut=load_npz(a.checks_dir/f'cutoff_L{L}_M{M}_N5.npz')
        tight=load_npz(a.checks_dir/f'tight_L{L}_M{M}.npz')
        corner_checks.append({
            'L':L,'M':M,
            'cutoff_max_v_trace_difference':float(np.max(np.abs(base['delta_v_trace']-cut['delta_v_trace']))),
            'cutoff_density_trace_difference':float(np.max(np.abs(base['density_trace']-cut['density_trace']))),
            'top_population_N4':float(base['max_top_sector_population']),
            'top_population_N5':float(cut['max_top_sector_population']),
            'tight_max_v_trace_difference':float(np.max(np.abs(base['delta_v_trace']-tight['delta_v_trace']))),
        })
    cross_cutoff=max(c['cutoff_max_v_trace_difference'] for c in corner_checks)
    cross_tight=max(c['tight_max_v_trace_difference'] for c in corner_checks)
    cross_floor=max(cross_cutoff,cross_tight)
    fine_corner=next(c for c in corner_checks if c['L']==12 and c['M']==4)
    cutoff={'L':12,'M':4,'Nmax':4,'Nmax_check':5,'max_v_trace_difference':fine_corner['cutoff_max_v_trace_difference'],'density_trace_difference':fine_corner['cutoff_density_trace_difference'],'top_population_N4':fine_corner['top_population_N4'],'top_population_N5':fine_corner['top_population_N5']}
    tol={'L':12,'M':4,'production_rtol':2e-9,'tight_rtol':5e-10,'max_v_trace_difference':fine_corner['tight_max_v_trace_difference']}
    cross_floor_check={'corners':corner_checks,'max_cutoff_v_trace_difference':cross_cutoff,'max_tight_v_trace_difference':cross_tight,'max_numerical_floor':cross_floor}

    primary=[r for r in rows if r['electronic_family']=='cosine' and r['photon_family']=='gauss']
    primary.sort(key=lambda r:(int(r['L']),int(r['M'])))
    tab=[r'\begin{tabular}{ccccc}',r'\toprule',r'$L$ & $M$ & $E^v_{L,M}$ & $E^j_M$ & $E^{\rm src}_{L,M}$\\',r'\midrule']
    for r in primary: tab.append(rf"{r['L']} & {r['M']} & ${texnum(float(r['E_v']))}$ & ${texnum(float(r['E_j']))}$ & ${texnum(float(r['E_src']))}$\\")
    tab += [r'\bottomrule',r'\end{tabular}'];(a.output_dir/'joint_analytic_inverse_table.tex').write_text('\n'.join(tab)+'\n')

    near=R['cosine','gauss',4,1];far=R['cosine','gauss',12,4];pc=next(r for r in cross if r['electronic_family']=='cosine' and r['photon_family']=='gauss')
    macros=''.join([
      f"\\newcommand{{\\AnalyticJointNear}}{{\\ensuremath{{{texnum(float(near['E_src']))}}}}}\n",
      f"\\newcommand{{\\AnalyticJointFar}}{{\\ensuremath{{{texnum(float(far['E_src']))}}}}}\n",
      f"\\newcommand{{\\AnalyticJointCross}}{{\\ensuremath{{{texnum(float(pc['cross_v']))}}}}}\n",
      f"\\newcommand{{\\AnalyticJointElectronSpreadCoarse}}{{\\ensuremath{{{texnum(esp[0]['spread_v'])}}}}}\n",
      f"\\newcommand{{\\AnalyticJointElectronSpreadFine}}{{\\ensuremath{{{texnum(esp[-1]['spread_v'])}}}}}\n",
      f"\\newcommand{{\\AnalyticJointPhotonSpreadFineV}}{{\\ensuremath{{{texnum(psp[-1]['spread_v'])}}}}}\n",
      f"\\newcommand{{\\AnalyticJointPhotonSpreadFineJ}}{{\\ensuremath{{{texnum(psp[-1]['spread_j'])}}}}}\n",
      f"\\newcommand{{\\AnalyticJointKappaMin}}{{\\ensuremath{{{min(float(r['min_kappa']) for r in rows):.3f}}}}}\n",
      f"\\newcommand{{\\AnalyticJointDensityMax}}{{\\ensuremath{{{texnum(max(float(r['max_density_tracking_error']) for r in rows))}}}}}\n",
      f"\\newcommand{{\\AnalyticJointQMax}}{{\\ensuremath{{{texnum(max(float(r['max_centered_q_error']) for r in rows))}}}}}\n",
      f"\\newcommand{{\\AnalyticJointNormMax}}{{\\ensuremath{{{texnum(max(float(r['max_norm_error']) for r in rows))}}}}}\n",
      f"\\newcommand{{\\AnalyticJointFockMax}}{{\\ensuremath{{{texnum(max(float(r['max_top_sector_population']) for r in rows))}}}}}\n",
      f"\\newcommand{{\\AnalyticJointCutoffV}}{{\\ensuremath{{{texnum(cutoff['max_v_trace_difference'])}}}}}\n",
      f"\\newcommand{{\\AnalyticJointTightV}}{{\\ensuremath{{{texnum(tol['max_v_trace_difference'])}}}}}\n",
      f"\\newcommand{{\\AnalyticJointCrossCutoffV}}{{\\ensuremath{{{texnum(cross_cutoff)}}}}}\n",
      f"\\newcommand{{\\AnalyticJointCrossTightV}}{{\\ensuremath{{{texnum(cross_tight)}}}}}\n",
      f"\\newcommand{{\\AnalyticJointCrossFloor}}{{\\ensuremath{{{texnum(cross_floor)}}}}}\n",
    ])
    (a.output_dir/'joint_analytic_inverse_macros.tex').write_text(macros)

    vals=np.array([[float(R['cosine','gauss',L,M]['E_src']) for L in L_VALUES] for M in M_VALUES]);fig,ax=plt.subplots(figsize=(3.45,2.45),constrained_layout=True);im=ax.imshow(np.log10(vals),origin='lower',aspect='auto');ax.set_xticks(range(3),map(str,L_VALUES));ax.set_yticks(range(3),map(str,M_VALUES));ax.set_xlabel(r'electronic regulator $L$');ax.set_ylabel(r'photon regulator $M$');thr=np.mean(np.log10(vals))
    for i in range(3):
      for j in range(3): ax.text(j,i,f'{vals[i,j]:.1e}',ha='center',va='center',fontsize=6.2,color='white' if np.log10(vals[i,j])<thr else 'black')
    cb=fig.colorbar(im,ax=ax,pad=.02);cb.set_label(r'$\log_{10} E^{\rm src}_{L,M}$');fig.savefig(a.figure_dir/'joint_analytic_inverse.pdf',bbox_inches='tight');fig.savefig(a.figure_dir/'joint_analytic_inverse.png',dpi=250,bbox_inches='tight');plt.close(fig)

    checks={'electronic_family_spreads':esp,'photon_family_spreads':psp,'cutoff_check':cutoff,'tolerance_check':tol,'mixed_defect_numerical_floor':cross_floor_check}
    if min(float(r['min_kappa']) for r in rows)<=0.75: raise RuntimeError('gap too small')
    if max(float(r['max_density_tracking_error']) for r in rows)>=2e-9: raise RuntimeError('density tracking failed')
    if max(float(r['max_centered_q_error']) for r in rows)>=2e-9: raise RuntimeError('Q tracking failed')
    if float(pc['cross_v']) <= 1000*cross_floor: raise RuntimeError('cross signal unresolved against four-corner numerical floor')
    payload={'benchmark':'joint_analytic_inverse_replacement','endpoint':'exact closed-form continuum target and source; no numerical reference trajectory','parameters':{'hopping':1.0,'selected_frequency':1.20,'dipole':0.50,'q_initial':0.20,'q_amplitude':0.15,'target_frequency':0.50,'bath_strength':0.60,'bath_low':0.75,'bath_high':1.65,'boundary_gate_amplitude':0.20,'final_time':2.0,'time_points':401,'boson_cutoff':4,'rtol':2e-9,'atol':2e-11,'L_values':list(L_VALUES),'M_values':list(M_VALUES),'electronic_families':list(ELECTRON_FAMILIES),'photon_families':list(PHOTON_FAMILIES)},'analytic_target':{'n':'1/2','Q':'Q0+A[1-cos(nu t)]','v':'omega*d*Q-d^2/2','j':'Qdd+omega^2 Q-omega*d/2-R_infty(t)','R_infty':'A nu^2 eta/Z [(sin(Omega_plus t)-sin(Omega_minus t))/t-(Omega_plus-Omega_minus)cos(nu t)]'},'metadata':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__},'grid':rows,'cross_defects':cross,'checks':checks}
    (a.output_dir/'joint_analytic_inverse_benchmark.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps({'near':near['E_src'],'far':far['E_src'],'cross':pc['cross_v'],'cross_floor':cross_floor,'checks':checks},indent=2))
if __name__=='__main__':main()
