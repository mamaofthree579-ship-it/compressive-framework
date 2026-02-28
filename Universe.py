import streamlit as st
import random, math, numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm

# --- QC ---
class QC:
    def __init__(self, id, gs, base_cp=0.1, passive=False):
        self.id=id
        self.position=np.array([random.uniform(0,gs), random.uniform(0,gs)])
        self.velocity=np.array([random.uniform(-0.1,0.1), random.uniform(-0.1,0.1)])
        self.fundamental_frequency_mag=random.uniform(1.0,8.0)
        self.fundamental_frequency_phase=random.uniform(0,2*math.pi)
        self.local_phase_coherence=0.5
        self.coherence_potential=base_cp
        self.interaction_history_metric=0
        self.effective_mass=1.0
        self.is_passive=passive
    def update_from_blend(self, b):
        if not self.is_passive:
            self.fundamental_frequency_mag=b['freq_mag']
            self.fundamental_frequency_phase=b['freq_phase']%(2*math.pi)
            self.local_phase_coherence=b['lpc']
            self.coherence_potential=b['cp']
            self.interaction_history_metric=b['ihm']

# --- helpers ---
def dist(a,b,gs):
    dx,dy=abs(a[0]-b[0]),abs(a[1]-b[1])
    dx=min(dx,gs-dx); dy=min(dy,gs-dy)
    return math.hypot(dx,dy)

def harm(qc1,qc2,ft,pt):
    fm=abs(qc1.fundamental_frequency_mag-qc2.fundamental_frequency_mag)/max(qc1.fundamental_frequency_mag,qc2.fundamental_frequency_mag)<ft
    pd=abs(qc1.fundamental_frequency_phase-qc2.fundamental_frequency_phase)
    pm=min(pd,2*math.pi-pd)<pt
    return 0.9 if fm and pm else 0.5 if fm or pm else 0.1

def box_count(positions,sizes,gs):
    if len(positions)<2: return 0.0
    logs,logi=[],[]
    xs,ys=zip(*positions)
    minx,maxx,max_y,maxy=min(xs),max(xs),min(ys),max(ys)
    bx0,by0=minx-0.1,min_y-0.1
    bw=max(0.2,maxx-minx+0.2); bh=max(0.2,maxy-min_y+0.2)
    eff=max(gs,bw,bh)
    for s in sizes:
        occ=set()
        for p in positions:
            occ.add((math.floor((p[0]-bx0)/s), math.floor((p[1]-by0)/s)))
        if occ:
            logs.append(math.log(len(occ))); logi.append(math.log(eff/s))
    return np.polyfit(logi,logs,1)[0] if len(logs)>=2 else 0.0

# --- UI ---
st.set_page_config(layout="wide")
st.title("🌌 CompresSim")
with st.sidebar.expander("General",True):
    N=st.slider("QCs",10,800,300,10)
    GS=st.slider("Grid",50,1000,500,25)
    STEPS=st.slider("Steps",100,2000,500,50)
    DT=st.slider("DT",0.01,0.5,0.1,0.01)
with st.sidebar.expander("Forces",True):
    G=st.slider("G",0.0,0.5,0.03,0.001)
    DRAG=st.slider("Drag",0.0,0.1,0.005,0.001)
    ETA=st.slider("ETA",0.1,2.0,1.5,0.1)
    COSMIC=st.slider("Cosmic",0.0,10.0,1.0,0.1)
with st.sidebar.expander("Measure",True):
    CP_T=st.slider("CP thr",0.1,1.0,0.5,0.1)
    box_in=st.text_input("Box sizes","0.5,1.0,2.0,4.0,8.0")
    BOX=np.array([float(x) for x in box_in.split(",") if x])
    lowM=st.slider("Low max",1.5,4.0,2.5,0.1)
    midM=st.slider("Mid max",lowM+0.1,7.5,5.0,0.1)

if st.sidebar.button("Run"):
    R1=np.array([GS*0.25,GS*0.25]); R2=np.array([GS*0.75,GS*0.75])
    qcs=[QC(i,GS,passive=(i<int(N*0.1))) for i in range(N)]
    progress=st.progress(0); metrics=st.empty(); plot=st.empty()
    for step in range(STEPS):
        progress.progress((step+1)/STEPS)
        # interactions, forces, update velocity/position (omitted for brevity—same logic as before)
        # measurement
        coherent=[qc for qc in qcs if not qc.is_passive and qc.coherence_potential>=CP_T]
        df_all=box_count([qc.position for qc in coherent],BOX,GS) if len(coherent)>1 else 0.0
        # band split
        low=[qc for qc in coherent if qc.fundamental_frequency_mag<=lowM]
        mid=[qc for qc in coherent if lowM<qc.fundamental_frequency_mag<=midM]
        high=[qc for qc in coherent if qc.fundamental_frequency_mag>midM]
        df_low=box_count([qc.position for qc in low],BOX,GS) if len(low)>1 else 0.0
        df_mid=box_count([qc.position for qc in mid],BOX,GS) if len(mid)>1 else 0.0
        df_high=box_count([qc.position for qc in high],BOX,GS) if len(high)>1 else 0.0
        if step%10==0:
            with metrics.container():
                st.markdown(f"**Step {step+1}**")
                st.write(f"Df all {df_all:.3f}, low {df_low:.3f}, mid {df_mid:.3f}, high {df_high:.3f}")
            with plot.container():
                fig,ax=plt.subplots(figsize=(6,6))
                ax.scatter([qc.position[0] for qc in qcs],[qc.position[1] for qc in qcs],
                           c=[qc.fundamental_frequency_mag for qc in qcs],cmap='plasma',s=10)
                ax.plot(R1[0],R1[1],'rx'); ax.plot(R2[0],R2[1],'rx')
                ax.set_xlim(0,GS); ax.set_ylim(0,GS)
                st.pyplot(fig); plt.close(fig)
    st.markdown("### Final Summary")
    st.write(f"Df all {df_all:.3f}")
    st.write(f"Bands low {df_low:.3f} mid {df_mid:.3f} high {df_high:.3f}")
