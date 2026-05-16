"""Duration-aware replay dynamics for partial replay bins."""
from __future__ import annotations
import itertools
import numpy as np
from scipy.special import logsumexp

_REG={}; _COUNT=itertools.count(1)
class DurationFloat(float):
    def __new__(cls,value,durations):
        ds=tuple(float(x) for x in durations); base=float(value)
        eps=(next(_COUNT)%1_000_000)*max(abs(base),1.0)*1e-15
        obj=float.__new__(cls,base+eps); obj.base=base; obj.transition_durations=ds
        _REG[float(obj)]=ds
        return obj
    def __hash__(self): return hash((float(self),self.transition_durations))
    def __mul__(self,o): return self.first()*o
    def __rmul__(self,o): return o*self.first()
    def first(self): return self.transition_durations[0] if self.transition_durations else self.base

def _dur_from_dt(dt):
    ds=getattr(dt,'transition_durations',None) or _REG.get(float(dt))
    return None if ds is None else np.asarray(ds,dtype=float)

def transition_durations_s(em):
    ds=getattr(em,'transition_durations',None)
    n=max(em.n_time-1,0)
    if ds is not None: return _check(ds,n,'transition_durations')
    ds=_dur_from_dt(em.dt)
    if ds is not None: return _check(ds,n,'dt.transition_durations')
    t=np.asarray(getattr(em,'times',[]),dtype=float)
    if t.shape==(em.n_time,) and em.n_time>1:
        d=np.diff(t)
        if np.all(np.isfinite(d)) and np.all(d>0): return d
    return np.full(n,float(em.dt),dtype=float)

def _check(v,n,name):
    a=np.asarray(v,dtype=float)
    if a.shape!=(n,): raise ValueError(f'{name} must have shape {(n,)}, got {a.shape}')
    if not np.all(np.isfinite(a)) or np.any(a<=0): raise ValueError(f'{name} must contain finite positive durations')
    return a

def attach_duration_metadata(em):
    ds=transition_durations_s(em)
    em.transition_durations=ds
    em.dt=DurationFloat(float(em.dt),ds)
    return em

def _ps(sig,dt): return max(float(sig)*np.sqrt(max(float(dt),np.finfo(float).tiny)),np.finfo(float).eps)
def _pss(sig,ds,dt): return np.asarray([_ps(sig,d) for d in ds],float) if len(ds) else np.empty(0)
def _rep(sig,ds,dt): return _ps(sig,float(np.median(ds)) if len(ds) else dt)
def _decays(v,ds,ref):
    if not len(ds): return np.empty(0)
    r=max(float(ref),np.finfo(float).tiny); b=max(float(v),np.finfo(float).tiny)
    return np.asarray([b**(float(d)/r) for d in ds],float)
def _scales(ds):
    s=np.ones_like(ds,dtype=float)
    if len(ds)>1: s[1:]=ds[1:]/ds[:-1]
    return s

def _t(trans,i): return trans[i] if isinstance(trans,(list,tuple)) else trans

def apply_duration_dynamics_patch():
    import hipporeplayimm.encoding as enc
    import hipporeplayimm.kd_reference as kd
    import hipporeplayimm.state_space as ss
    if getattr(ss,'_duration_dynamics_patch_applied',False): return
    _patch_builders(enc,kd); _patch_state_space(ss); _patch_kd(kd)
    ss._duration_dynamics_patch_applied=True

def _patch_builders(enc,kd):
    if not getattr(enc.build_emissions,'_duration_wrapped',False):
        orig=enc.build_emissions
        def f(*a,**k): return attach_duration_metadata(orig(*a,**k))
        f._duration_wrapped=True; enc.build_emissions=f
    if not getattr(kd.build_kd_emissions,'_duration_wrapped',False):
        orig=kd.build_kd_emissions
        def f(*a,**k): return attach_duration_metadata(orig(*a,**k))
        f._duration_wrapped=True; kd.build_kd_emissions=f

def _patch_state_space(ss):
    def fb(loglik,trans):
        T,N=loglik.shape; scaled,offs=ss._scaled_emissions(loglik); filt=np.zeros((T,N)); sc=np.zeros(T)
        a=scaled[0]/N; sc[0]=float(a.sum())
        if sc[0]<=0: raise ValueError('first emission row has no finite likelihood mass')
        a/=sc[0]; filt[0]=a; lp=float(np.log(sc[0])+offs[0])
        for i in range(1,T):
            a=np.asarray(_t(trans,i-1)@a,float)*scaled[i]; sc[i]=float(a.sum())
            if sc[i]<=0: raise ValueError(f'emission row {i} has no finite predicted mass')
            a/=sc[i]; filt[i]=a; lp+=float(np.log(sc[i])+offs[i])
        sm=np.zeros_like(filt); b=np.ones(N); sm[-1]=filt[-1]
        for i in range(T-1,0,-1):
            b=np.asarray(_t(trans,i-1).T@(scaled[i]*b),float)/sc[i]
            g=filt[i-1]*b; tot=float(g.sum()); sm[i-1]=g/tot if tot>0 else filt[i-1]
        return lp,ss._as_log_probs(sm)
    def apply_fimm(loglik,centers,*,stationary_sigma_cm,diffusion_transitions,max_step_sigma,mode_stickiness):
        modes=('stationary','diffusion','fragmented'); M=len(modes); T,N=loglik.shape
        trans={'stationary':ss._gaussian_transition_matrix(centers,stationary_sigma_cm,max_step_sigma),'diffusion':diffusion_transitions,'fragmented':None}
        mt=ss._mode_transition_matrix(M,mode_stickiness); scaled,offs=ss._scaled_emissions(loglik)
        filt=np.zeros((T,M,N)); sc=np.zeros(T); a=np.tile(scaled[0]/(N*M),(M,1)); sc[0]=float(a.sum())
        if sc[0]<=0: raise ValueError('first emission row has no finite likelihood mass')
        a/=sc[0]; filt[0]=a; lp=float(np.log(sc[0])+offs[0])
        for ti in range(1,T):
            pred=np.zeros_like(a)
            for d,dm in enumerate(modes):
                dst=np.zeros(N)
                for s in range(M):
                    tr=trans[dm]
                    if tr is None: val=np.full(N,float(a[s].sum())/N)
                    else: val=np.asarray(_t(tr,ti-1)@a[s],float)
                    dst+=mt[s,d]*val
                pred[d]=dst
            a=pred*scaled[ti][None,:]; sc[ti]=float(a.sum())
            if sc[ti]<=0: raise ValueError(f'emission row {ti} has no finite predicted mass')
            a/=sc[ti]; filt[ti]=a; lp+=float(np.log(sc[ti])+offs[ti])
        sm=np.zeros_like(filt); b=np.ones((M,N)); sm[-1]=filt[-1]
        for ti in range(T-1,0,-1):
            bp=np.zeros_like(b)
            for s in range(M):
                for d,dm in enumerate(modes):
                    tr=trans[dm]; v=scaled[ti]*b[d]
                    val=np.full(N,float(v.sum())/N) if tr is None else np.asarray(_t(tr,ti-1).T@v,float)
                    bp[s]+=mt[s,d]*val
            b=bp/sc[ti]; g=filt[ti-1]*b; tot=float(g.sum()); sm[ti-1]=g/tot if tot>0 else filt[ti-1]
        return lp,ss._as_log_probs(sm.sum(axis=1)),sm.sum(axis=2)
    def adv_mom(pair,pp,p,c,ce,centers,*,sigma_cm,velocity_decay,time_scale):
        cpp=centers[pp]; cp=centers[p]; cc=centers[c]; out=np.full((len(p),len(c)),ss.LOG_ZERO)
        for j in range(len(p)):
            pred=cp[j][None,:]+float(velocity_decay)*float(time_scale)*(cp[j][None,:]-cpp)
            lk=ss._full_grid_normalized_pairwise_gaussian_log_prob(pred,cc,centers,float(sigma_cm))
            out[j]=logsumexp(pair[:,j][:,None]+lk,axis=0)+ce
        return out
    def back_mom(nb,pp,p,c,ce,centers,*,sigma_cm,velocity_decay,time_scale):
        cpp=centers[pp]; cp=centers[p]; cc=centers[c]; out=np.full((len(pp),len(p)),ss.LOG_ZERO)
        for j in range(len(p)):
            pred=cp[j][None,:]+float(velocity_decay)*float(time_scale)*(cp[j][None,:]-cpp)
            lk=ss._full_grid_normalized_pairwise_gaussian_log_prob(pred,cc,centers,float(sigma_cm))
            out[:,j]=logsumexp(lk+ce[None,:]+nb[j][None,:],axis=1)
        return out
    def score_mom(em,centers,cands,*,sigmas_cm,initial_sigma_cm,velocity_decays,time_scales):
        if em.n_time==1:
            lp,tr=ss._score_fragmented(em); return lp,tr,[0.0]
        masses=ss._candidate_log_masses(em.log_likelihood,cands)
        pair=ss._init_pair_log_alpha(em.log_likelihood,cands[0],cands[1],centers,sigma_cm=float(initial_sigma_cm)); al=[pair]
        for ti in range(2,em.n_time):
            k=ti-1; pair=adv_mom(pair,cands[ti-2],cands[ti-1],cands[ti],em.log_likelihood[ti,cands[ti]],centers,sigma_cm=sigmas_cm[k],velocity_decay=velocity_decays[k],time_scale=time_scales[k]); al.append(pair)
        lp=float(logsumexp(al[-1])); be=[np.zeros_like(al[-1]) for _ in al]
        for pi in range(len(al)-2,-1,-1):
            k=pi+1; ct=pi+2; be[pi]=back_mom(be[pi+1],cands[pi],cands[pi+1],cands[ct],em.log_likelihood[ct,cands[ct]],centers,sigma_cm=sigmas_cm[k],velocity_decay=velocity_decays[k],time_scale=time_scales[k])
        traj=np.full((em.n_time,em.n_bins),ss.LOG_ZERO)
        for pi,(a,b) in enumerate(zip(al,be,strict=True)):
            post=a+b-lp
            if pi==0: traj[0,cands[0]]=logsumexp(post,axis=1)
            traj[pi+1,cands[pi+1]]=logsumexp(post,axis=0)
        for i in range(em.n_time): traj[i]-=logsumexp(traj[i])
        return lp,traj,masses
    def score(self,em,centers,candidate_indices=None):
        if em.n_time==0: raise ValueError('emissions must contain at least one time bin')
        if em.n_bins!=centers.shape[0]: raise ValueError('emissions.n_bins must match bin_centers rows')
        assert self.config is not None
        ds=transition_durations_s(em); attach_duration_metadata(em); extra={}
        if self.mode=='stationary': lp,tr=ss._score_stationary(em); ts=0.0
        elif self.mode in {'fragmented','jump'}: lp,tr=ss._score_fragmented(em); ts=float('inf')
        elif self.mode=='diffusion':
            sig=_pss(self.config.diffusion_sigma_cm_sqrt_s,ds,float(em.dt)); ts=_rep(self.config.diffusion_sigma_cm_sqrt_s,ds,float(em.dt))
            mats=[ss._gaussian_transition_matrix(centers,float(s),self.config.max_step_sigma) for s in sig]
            lp,tr=fb(em.log_likelihood,mats)
        elif self.mode=='first-order-imm':
            sig=_pss(self.config.diffusion_sigma_cm_sqrt_s,ds,float(em.dt)); ts=_rep(self.config.diffusion_sigma_cm_sqrt_s,ds,float(em.dt))
            mats=[ss._gaussian_transition_matrix(centers,float(s),self.config.max_step_sigma) for s in sig]
            lp,tr,mp=apply_fimm(em.log_likelihood,centers,stationary_sigma_cm=self.config.stationary_sigma_cm,diffusion_transitions=mats,max_step_sigma=self.config.max_step_sigma,mode_stickiness=self.config.imm_mode_stickiness)
            names=('stationary','diffusion','fragmented'); extra={f'state_space_mode_{n}_terminal_probability':float(mp[-1,i]) for i,n in enumerate(names)}; extra.update({'state_space_imm_modes':','.join(names),'state_space_imm_evidence_support':'exact_full_grid'})
        elif self.mode in {'momentum','imm'}:
            # Use duration-aware momentum for momentum mode; delegate four-mode IMM to existing scorer with representative dt.
            if self.mode=='imm':
                old=em.dt; em.dt=float(old.base if hasattr(old,'base') else old)
                res=ss.StateSpaceReplayModel.__duration_original_score__(self,em,centers,candidate_indices)
                em.dt=old
                res.diagnostics['state_space_transition_durations']=','.join(f'{d:.12g}' for d in ds)
                return res
            sig=_pss(self.config.momentum_sigma_cm_sqrt_s,ds,float(em.dt)); ts=_rep(self.config.momentum_sigma_cm_sqrt_s,ds,float(em.dt))
            ini=_ps(self.config.momentum_initial_sigma_cm_sqrt_s,ds[0] if len(ds) else float(em.dt)); dec=_decays(self.config.momentum_velocity_decay,ds,float(em.dt)); sc=_scales(ds)
            c=self.candidate_indices(em) if candidate_indices is None else candidate_indices; ss._validate_candidate_indices(c,em.n_time,em.n_bins)
            lp,tr,masses=score_mom(em,centers,c,sigmas_cm=sig,initial_sigma_cm=ini,velocity_decays=dec,time_scales=sc)
            extra={'mean_candidate_log_mass':float(np.mean(masses)),'state_space_momentum_candidate_top_k':int(self.config.momentum_candidate_top_k),'state_space_momentum_candidate_support':'derived' if candidate_indices is None else 'provided','state_space_momentum_trajectory_posterior':'smoothed_pair_marginal','state_space_momentum_evidence_support':'truncated_full_grid'}
        else: raise ValueError(f'Unsupported state-space mode: {self.mode}')
        term=tr[-1]; diag={'state_space_mode':str(self.mode),'state_space_time_bin_s':float(em.dt),'state_space_transition_durations':','.join(f'{d:.12g}' for d in ds),'state_space_trajectory_posterior':1,'state_space_trajectory_time_bins':int(em.n_time),'state_space_stationary_sigma_cm':float(self.config.stationary_sigma_cm),'state_space_diffusion_sigma_cm_sqrt_s':float(self.config.diffusion_sigma_cm_sqrt_s),'state_space_max_step_sigma':float(self.config.max_step_sigma),'state_space_imm_mode_stickiness':float(self.config.imm_mode_stickiness),'state_space_momentum_sigma_cm_sqrt_s':float(self.config.momentum_sigma_cm_sqrt_s),'state_space_momentum_initial_sigma_cm_sqrt_s':float(self.config.momentum_initial_sigma_cm_sqrt_s),'state_space_momentum_velocity_decay':float(self.config.momentum_velocity_decay),'state_space_transition_sigma_cm':float(ts),'mean_trajectory_posterior_entropy':ss._mean_entropy(tr),**extra}
        diag.update(ss._posterior_diagnostics(term,centers)); return ss.EventScore(str(self.name),float(lp),em.n_time,em.n_spikes,diagnostics=diag,terminal_log_posterior=term,trajectory_log_posterior=tr)
    ss.StateSpaceReplayModel.__duration_original_score__=ss.StateSpaceReplayModel.score
    ss.StateSpaceReplayModel.score=score

def _patch_kd(kd):
    od=kd.diffusion_transition_1d; om=kd.momentum_transition_1d; oa=kd.adjusted_momentum_parameters
    def diff1(n,sd,bin_cm,dt):
        ds=_dur_from_dt(dt)
        return [od(n,sd,bin_cm,float(d)) for d in ds] if ds is not None else od(n,sd,bin_cm,dt)
    def first_var(loge,nx,ny,trans):
        e,o=kd._scaled_emission(loge,0); a=e.reshape(nx,ny)/loge.shape[1]; c=float(a.sum()); lp=np.log(c)+o; a/=c
        for i in range(1,loge.shape[0]):
            e,o=kd._scaled_emission(loge,i); tr=_t(trans,i-1); a=(tr@a@tr.T)*e.reshape(nx,ny); c=float(a.sum())
            if c<=0: return float('-inf')
            lp+=np.log(c)+o; a/=c
        return float(lp)
    def kde_from(loge,nx,ny,trans): return first_var(loge,nx,ny,trans) if isinstance(trans,(list,tuple)) else kd._first_order_separable_log_evidence(loge,nx,ny,trans)
    def kde(loge,nx,ny,sd,bin_cm,dt): return kde_from(loge,nx,ny,diff1(nx,sd,bin_cm,dt))
    def adj(th,sig,dt):
        ds=_dur_from_dt(dt)
        return oa(th,sig,float(np.median(ds))) if ds is not None and len(ds) else oa(th,sig,dt)
    def mom1(n,sd,decay,bin_cm,dt):
        ds=_dur_from_dt(dt)
        return [om(n,sd,decay,bin_cm,float(d)) for d in ds[1:]] if ds is not None else om(n,sd,decay,bin_cm,dt)
    def second_var(loge,n,init,trans):
        if loge.shape[0]==1: return kd.kd_random_log_evidence(loge)
        e0,o0=kd._scaled_emission(loge,0); a0=e0.reshape(n,n)/loge.shape[1]; c=float(a0.sum()); lp=np.log(c)+o0; a0/=c
        e1,o1=kd._scaled_emission(loge,1); eg=e1.reshape(n,n); a=np.einsum('ip,jq,pq,ij->ijpq',init,init,a0,eg,optimize=True); c=float(a.sum())
        if c<=0: return float('-inf')
        lp+=np.log(c)+o1; a/=c
        for i in range(2,loge.shape[0]):
            tr=_t(trans,i-2); e,o=kd._scaled_emission(loge,i); eg=e.reshape(n,n); ys=np.einsum('jbq,abpq->abpj',tr,a,optimize=True); pr=np.einsum('iap,abpj->ijab',tr,ys,optimize=True); a=pr*eg[:,:,None,None]; c=float(a.sum())
            if c<=0: return float('-inf')
            lp+=np.log(c)+o; a/=c
        return float(lp)
    def kdm_from(loge,n,init,trans): return second_var(loge,n,init,trans) if isinstance(trans,(list,tuple)) else kd._second_order_separable_log_evidence(loge,n,init,trans)
    def kdm(loge,nx,ny,sd,decay,init_sd,bin_cm,dt):
        if nx!=ny: raise ValueError('KD momentum scorer currently requires a square grid')
        if loge.shape[0]==1: return kd.kd_random_log_evidence(loge)
        ds=_dur_from_dt(dt); adj_decay,adj_sd=(adj(decay,sd,dt) if decay>1 else (decay,sd)); first=float(ds[0]) if ds is not None and len(ds) else float(dt)
        init=od(nx,init_sd*first,bin_cm,dt=1.0); trans=([om(nx,adj_sd,adj_decay,bin_cm,float(d)) for d in ds[1:]] if ds is not None else om(nx,adj_sd,adj_decay,bin_cm,dt))
        return kdm_from(loge,nx,init,trans)
    kd.diffusion_transition_1d=diff1; kd.kd_diffusion_log_evidence_from_transition=kde_from; kd.kd_diffusion_log_evidence=kde
    kd.adjusted_momentum_parameters=adj; kd.momentum_transition_1d=mom1; kd.kd_momentum_log_evidence_from_transitions=kdm_from; kd.kd_momentum_log_evidence=kdm
