"""
Behavioral Fingerprint Insider Threat Detection
Streamlit App — Train · Verify · Visualize
"""

import streamlit as st
import os, io, random, pickle, warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Behavioral Fingerprint",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }

.stApp { background: #0d0f14; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2130; }

.metric-card {
    background: #151820;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.metric-card .label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.metric-card .value { font-size: 26px; font-weight: 500; font-family: 'IBM Plex Mono', monospace; color: #38bdf8; }
.metric-card .sub { font-size: 11px; color: #475569; margin-top: 2px; }

.status-bar {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 1rem;
}
.result-accept {
    background: #0a1f13;
    border: 1px solid #166534;
    border-left: 3px solid #22c55e;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    font-family: 'IBM Plex Mono', monospace;
}
.result-reject {
    background: #1f0a0a;
    border: 1px solid #7f1d1d;
    border-left: 3px solid #ef4444;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    font-family: 'IBM Plex Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 30
STRIDE      = 10
EMBED_DIM   = 64
HIDDEN_DIM  = 64
BATCH_SIZE  = 128
EPOCHS      = 30
MAX_USERS   = 70
MIN_WINDOWS = 4
AUX_WEIGHT  = 0.1
FEATURE_COLS = ["event_code", "hour_norm", "dow_norm", "log_td_norm", "is_weekend", "is_after_hours"]

COLUMN_ALIASES = {
    "user":     ["user","User","username","Username","user_id"],
    "date":     ["date","Date","timestamp","Timestamp","time","Time","datetime","DateTime"],
    "pc":       ["pc","PC","host","Host","machine","Machine","computer","Computer"],
    "activity": ["activity","Activity","action","Action","event_type","type","Type"],
}

# ── Model classes ─────────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        w = torch.softmax(self.attn(h).squeeze(-1), dim=-1).unsqueeze(-1)
        return (h * w).sum(dim=1), w.squeeze(-1)

class BehavioralFingerprintEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, embed_dim=64, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim//2, n_layers, batch_first=True, bidirectional=True, dropout=dropout if n_layers>1 else 0.0)
        self.attention = AttentionPooling(hidden_dim)
        self.embed_head = nn.Sequential(nn.Linear(hidden_dim, embed_dim*2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim*2, embed_dim))
    def forward(self, x):
        proj, _ = self.bilstm(self.input_proj(x))
        ctx, aw = self.attention(proj)
        emb = F.normalize(self.embed_head(ctx), p=2, dim=-1)
        return emb, aw

class AuxClassifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)
    def forward(self, x): return self.fc(x)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, a, p, n):
        return F.relu((a-p).pow(2).sum(1) - (a-n).pow(2).sum(1) + self.margin).mean()

class TripletBehaviorDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = torch.tensor(seqs, dtype=torch.float32)
        self.labels = labels
        self.lbl_idx = {}
        for i, l in enumerate(labels): self.lbl_idx.setdefault(l, []).append(i)
        self.unique = list(self.lbl_idx.keys())
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        a, al = self.seqs[idx], self.labels[idx]
        pc = [i for i in self.lbl_idx[al] if i != idx] or self.lbl_idx[al]
        nl = random.choice([l for l in self.unique if l != al])
        return a, self.seqs[random.choice(pc)], self.seqs[random.choice(self.lbl_idx[nl])], torch.tensor(al, dtype=torch.long)

# ── Helper functions ──────────────────────────────────────────────────────────
def normalise_columns(df, name):
    rm = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in df.columns and canon not in df.columns:
                rm[a] = canon; break
    df = df.rename(columns=rm)
    missing = [c for c in ("user","date","activity") if c not in df.columns]
    if missing:
        raise ValueError(f"'{name}.csv' missing columns after normalisation: {missing}. Found: {list(df.columns)}")
    if "pc" not in df.columns: df["pc"] = "unknown_pc"
    return df

def safe_parse_timestamps(s):
    p = pd.to_datetime(s, infer_datetime_format=True, errors="coerce")
    mask = p.isna()
    if mask.any():
        try: p[mask] = pd.to_datetime(s[mask], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        except: pass
    return p

def load_modality(name, uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory=False)
    df = normalise_columns(df, name)
    df["date"] = safe_parse_timestamps(df["date"])
    before = len(df)
    df = df.dropna(subset=["date","user"])
    dropped = before - len(df)
    df["modality"] = name
    df["activity"] = df["activity"].astype(str).str.strip().str.lower()
    df["pc"] = df["pc"].astype(str).str.strip().fillna("unknown_pc")
    return df, dropped

def engineer_features(merged_df):
    top_users = merged_df["user"].value_counts().head(MAX_USERS).index.tolist()
    df = merged_df[merged_df["user"].isin(top_users)].copy()
    df["event_label"] = df["modality"] + "*" + df["activity"] + "*" + df["pc"].astype(str)
    event_enc = LabelEncoder(); df["event_code"] = event_enc.fit_transform(df["event_label"])
    user_enc  = LabelEncoder(); df["user_id"]    = user_enc.fit_transform(df["user"])
    pc_enc    = LabelEncoder(); df["pc_code"]    = pc_enc.fit_transform(df["pc"].fillna("unknown_pc"))
    df = df.sort_values(["user","date"]).reset_index(drop=True)
    df["hour"]         = df["date"].dt.hour
    df["dow"]          = df["date"].dt.dayofweek
    df["is_weekend"]   = (df["dow"] >= 5).astype(int)
    df["is_after_hours"] = ((df["hour"] < 8) | (df["hour"] > 18)).astype(int)
    df["time_delta"]   = df.groupby("user")["date"].diff().dt.total_seconds().div(60)
    df["time_delta"]   = df["time_delta"].fillna(0).clip(0, 1440)
    df["log_time_delta"] = np.log1p(df["time_delta"])
    td_scaler = MinMaxScaler()
    df["log_td_norm"]  = td_scaler.fit_transform(df[["log_time_delta"]]).flatten()
    df["hour_norm"]    = df["hour"] / 23.0
    df["dow_norm"]     = df["dow"] / 6.0
    scaler = StandardScaler()
    df[FEATURE_COLS]   = scaler.fit_transform(df[FEATURE_COLS])
    return df, user_enc, event_enc, pc_enc, td_scaler

def build_sequences(df):
    seqs, labels = [], []
    for uid, g in df.groupby("user_id"):
        vals = g[FEATURE_COLS].values.astype(np.float32)
        for s in range(0, len(vals) - WINDOW_SIZE + 1, STRIDE):
            seqs.append(vals[s:s+WINDOW_SIZE]); labels.append(uid)
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int64)

def extract_embeddings(loader, model):
    model.eval(); all_e, all_l = [], []
    with torch.no_grad():
        for a,_,_,l in loader:
            e,_ = model(a.to(DEVICE)); all_e.append(e.cpu()); all_l.append(l)
    return torch.cat(all_e), torch.cat(all_l)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 BehavioralFingerprint")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Overview", "🧠 Train Model", "🔎 Verify Session", "📊 Visualize Results"])
    st.markdown("---")
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown(f"**Window:** `{WINDOW_SIZE}` events")
    st.markdown(f"**Embed dim:** `{EMBED_DIM}`")

    if "model" in st.session_state:
        st.success("Model loaded ✓")
    if "metrics" in st.session_state:
        m = st.session_state["metrics"]
        st.markdown("---")
        st.markdown("**Last run metrics**")
        st.markdown(f"Accuracy: `{m['accuracy']*100:.1f}%`")
        st.markdown(f"EER: `{m['eer']*100:.1f}%`")
        st.markdown(f"AUC: `{m['auc']:.3f}`")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Behavioral Fingerprint")
    st.markdown("#### Insider Threat Detection via Identity-Aware Embedding")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="label">Architecture</div>
            <div class="value">BiLSTM</div>
            <div class="sub">+ self-attention + triplet loss</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="label">Dataset</div>
            <div class="value">CERT r4.2</div>
            <div class="sub">5 activity modalities</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="label">Threshold</div>
            <div class="value">EER-derived</div>
            <div class="sub">cosine similarity @ equal error rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### How it works")
    st.markdown("""
1. **Load data** — Upload all 5 CERT CSV files (logon, file, email, http, device)  
2. **Feature engineering** — Each event becomes a token: `modality * activity * pc`, enriched with temporal features (hour, day-of-week, time-delta)  
3. **Sliding windows** — User activity is split into overlapping windows of 30 events  
4. **Triplet training** — BiLSTM + attention encoder learns to push same-user embeddings together and different-user embeddings apart  
5. **Gallery** — A mean prototype embedding is stored per user  
6. **Verification** — A new session embedding is compared to the claimed user's prototype via cosine similarity; sessions below the EER threshold are flagged as intruders
    """)

    st.markdown("### Required CSV files")
    for f in ["logon.csv", "file.csv", "email.csv", "http.csv", "device.csv"]:
        st.code(f)
    st.info("Download the CERT r4.2 dataset from: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Train Model":
    st.title("Train Model")
    st.markdown("Upload all 5 CERT CSV files, configure hyperparameters, then train.")
    st.markdown("---")

    with st.sidebar:
        st.markdown("### Hyperparameters")
        max_users   = st.slider("Max users",    10, 100, MAX_USERS)
        window_size = st.slider("Window size",  10,  60, WINDOW_SIZE)
        stride      = st.slider("Stride",        5,  20, STRIDE)
        embed_dim   = st.select_slider("Embed dim", [32, 64, 128], EMBED_DIM)
        hidden_dim  = st.select_slider("Hidden dim",[32, 64, 128], HIDDEN_DIM)
        epochs      = st.slider("Epochs",         5,  60, EPOCHS)
        lr          = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3], 5e-4)

    # File upload
    uploaded = st.file_uploader(
        "Upload CERT CSV files (logon, file, email, http, device)",
        type="csv", accept_multiple_files=True,
        help="Select all 5 files at once"
    )

    # Load saved model
    st.markdown("---")
    st.markdown("**Or load a previously saved model:**")
    col_m, col_e = st.columns(2)
    with col_m: model_file = st.file_uploader("Model (.pth)", type="pth", key="model_up")
    with col_e: enc_file   = st.file_uploader("Encoders (.pkl)", type="pkl", key="enc_up")

    if model_file and enc_file:
        try:
            enc = pickle.load(enc_file)
            ckpt = torch.load(model_file, map_location=DEVICE)
            hp = ckpt["hyperparams"]
            m = BehavioralFingerprintEncoder(hp["input_dim"], hp["hidden_dim"], embed_dim=hp["embed_dim"]).to(DEVICE)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            ac = AuxClassifier(hp["embed_dim"], hp["n_classes"]).to(DEVICE)
            ac.load_state_dict(ckpt["aux_clf_state_dict"])
            st.session_state["model"]   = m
            st.session_state["aux_clf"] = ac
            st.session_state["gallery"] = ckpt["gallery_matrix"].to(DEVICE)
            st.session_state["enc"]     = enc
            st.session_state["threshold"] = enc["threshold"]
            st.success(f"Model loaded — {hp['n_classes']} users, embed_dim={hp['embed_dim']}, threshold={enc['threshold']:.4f}")
        except Exception as ex:
            st.error(f"Failed to load model: {ex}")

    # Train
    st.markdown("---")
    if st.button("🚀 Train Model", type="primary", disabled=not uploaded):
        file_map = {f.name.lower().replace(".csv",""): f for f in uploaded}
        required = ["logon","file","email","http","device"]
        missing  = [r for r in required if r not in file_map]
        if missing:
            st.error(f"Missing files: {missing}")
            st.stop()

        progress = st.progress(0)
        status   = st.empty()

        # Load data
        status.markdown('<div class="status-bar">⬜ Loading modalities...</div>', unsafe_allow_html=True)
        frames = []
        for mod in required:
            df_mod, dropped = load_modality(mod, file_map[mod])
            frames.append(df_mod[["user","date","activity","modality","pc"]])
            st.write(f"  `{mod}`: {len(df_mod):,} rows ({dropped} dropped)")
        merged = pd.concat(frames, ignore_index=True).sort_values(["user","date"]).reset_index(drop=True)
        st.write(f"**Total events:** {len(merged):,} | **Unique users:** {merged['user'].nunique()}")
        progress.progress(10)

        # Features
        status.markdown('<div class="status-bar">⬜ Engineering features...</div>', unsafe_allow_html=True)
        df_feat, user_enc, event_enc, pc_enc, td_scaler = engineer_features(merged)
        N_EVENTS = len(event_enc.classes_)
        N_USERS  = len(user_enc.classes_)
        st.write(f"**Event vocab size:** {N_EVENTS} | **Users:** {N_USERS}")
        progress.progress(20)

        # Sequences
        status.markdown('<div class="status-bar">⬜ Building sequences...</div>', unsafe_allow_html=True)
        all_seqs, all_labels = build_sequences(df_feat)
        counts = Counter(all_labels)
        valid  = {u for u, c in counts.items() if c >= MIN_WINDOWS}
        mask   = np.array([l in valid for l in all_labels])
        all_seqs, all_labels = all_seqs[mask], all_labels[mask]
        label_remap = {old: new for new, old in enumerate(sorted(valid))}
        all_labels  = np.array([label_remap[l] for l in all_labels], dtype=np.int64)
        N_CLASSES   = len(label_remap)
        st.write(f"**Sequences:** {len(all_seqs):,} | **Classes:** {N_CLASSES}")
        progress.progress(30)

        # Split
        idx = np.arange(len(all_seqs))
        ti, tmp = train_test_split(idx, test_size=0.30, stratify=all_labels, random_state=SEED)
        vi, tei = train_test_split(tmp, test_size=0.50, stratify=all_labels[tmp], random_state=SEED)
        X_tr,y_tr = all_seqs[ti], all_labels[ti]
        X_va,y_va = all_seqs[vi], all_labels[vi]
        X_te,y_te = all_seqs[tei],all_labels[tei]

        tr_dl = DataLoader(TripletBehaviorDataset(X_tr,y_tr), BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
        va_dl = DataLoader(TripletBehaviorDataset(X_va,y_va), BATCH_SIZE, shuffle=False, num_workers=0)
        te_dl = DataLoader(TripletBehaviorDataset(X_te,y_te), BATCH_SIZE, shuffle=False, num_workers=0)

        # Model
        N_FEAT = len(FEATURE_COLS)
        model  = BehavioralFingerprintEncoder(N_FEAT, hidden_dim, embed_dim=embed_dim).to(DEVICE)
        ac     = AuxClassifier(embed_dim, N_CLASSES).to(DEVICE)
        tri_fn = TripletLoss(1.0)
        ce_fn  = nn.CrossEntropyLoss()
        opt    = torch.optim.Adam(list(model.parameters())+list(ac.parameters()), lr=lr, weight_decay=5e-4)
        sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        # Training
        status.markdown('<div class="status-bar">⬜ Training...</div>', unsafe_allow_html=True)
        tr_losses, va_losses = [], []
        best_val, patience, counter = float("inf"), 10, 0
        epoch_bar = st.progress(0)
        epoch_text = st.empty()

        for ep in range(1, epochs+1):
            model.train(); ac.train(); ep_loss = 0
            for a,p,n,l in tr_dl:
                a,p,n,l = a.to(DEVICE),p.to(DEVICE),n.to(DEVICE),l.to(DEVICE)
                ea,_=model(a); ep2,_=model(p); en,_=model(n)
                loss = tri_fn(ea,ep2,en) + AUX_WEIGHT*ce_fn(ac(ea),l)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); ep_loss += loss.item()
            sched.step()
            avg_tr = ep_loss/len(tr_dl); tr_losses.append(avg_tr)

            model.eval(); ac.eval(); vl=0
            with torch.no_grad():
                for a,p,n,l in va_dl:
                    a,p,n,l=a.to(DEVICE),p.to(DEVICE),n.to(DEVICE),l.to(DEVICE)
                    ea,_=model(a); ep2,_=model(p); en,_=model(n)
                    vl += (tri_fn(ea,ep2,en)+AUX_WEIGHT*ce_fn(ac(ea),l)).item()
            avg_va = vl/len(va_dl); va_losses.append(avg_va)

            if avg_va < best_val: best_val=avg_va; counter=0
            else:
                counter+=1
                if counter >= patience:
                    epoch_text.write(f"Early stopping at epoch {ep}")
                    break

            epoch_bar.progress(ep/epochs)
            epoch_text.write(f"Epoch {ep}/{epochs} — Train: `{avg_tr:.4f}` | Val: `{avg_va:.4f}`")

        progress.progress(70)

        # Gallery
        status.markdown('<div class="status-bar">⬜ Building gallery...</div>', unsafe_allow_html=True)
        tr_embs, tr_lbls = extract_embeddings(tr_dl, model)
        te_embs, te_lbls = extract_embeddings(te_dl, model)
        gallery = {}
        for uid in range(N_CLASSES):
            mask = (tr_lbls == uid)
            if mask.sum() > 0: gallery[uid] = tr_embs[mask].mean(dim=0)
        gallery_mat = torch.stack([gallery[u] for u in range(N_CLASSES)])
        progress.progress(80)

        # Evaluation
        status.markdown('<div class="status-bar">⬜ Evaluating...</div>', unsafe_allow_html=True)
        raw = []
        model.eval()
        with torch.no_grad():
            for a,_,_,l in te_dl:
                embs = model(a.to(DEVICE))[0].cpu()
                for emb, lbl in zip(embs, l):
                    lbl = lbl.item()
                    sim_l = F.cosine_similarity(emb.unsqueeze(0), gallery_mat[lbl].unsqueeze(0)).item()
                    raw.append((True,  sim_l))
                    wid = random.choice([u for u in range(N_CLASSES) if u != lbl])
                    sim_i = F.cosine_similarity(emb.unsqueeze(0), gallery_mat[wid].unsqueeze(0)).item()
                    raw.append((False, sim_i))

        y_true = np.array([1 if r[0] else 0 for r in raw])
        scores = np.array([r[1] for r in raw])
        legit_sims    = scores[y_true==1]
        intruder_sims = scores[y_true==0]

        # EER
        thr_sweep = np.linspace(0,1,200)
        far_a, frr_a = [], []
        for t in thr_sweep:
            yp = (scores>=t).astype(int)
            fp_=(( yp==1)&(y_true==0)).sum(); fn_=((yp==0)&(y_true==1)).sum()
            tp_=((yp==1)&(y_true==1)).sum();  tn_=((yp==0)&(y_true==0)).sum()
            far_a.append(fp_/(fp_+tn_+1e-9)); frr_a.append(fn_/(fn_+tp_+1e-9))
        far_a, frr_a = np.array(far_a), np.array(frr_a)
        eer_idx = np.argmin(np.abs(far_a-frr_a))
        eer_thr = float(thr_sweep[eer_idx]); eer_val = float((far_a[eer_idx]+frr_a[eer_idx])/2)
        THRESHOLD = eer_thr

        y_pred = (scores>=THRESHOLD).astype(int)
        tp=int(((y_pred==1)&(y_true==1)).sum()); tn=int(((y_pred==0)&(y_true==0)).sum())
        fp=int(((y_pred==1)&(y_true==0)).sum()); fn=int(((y_pred==0)&(y_true==1)).sum())
        accuracy = (tp+tn)/len(y_true)
        FAR = fp/(fp+tn+1e-9); FRR = fn/(fn+tp+1e-9)
        fpr_r, tpr_r, _ = roc_curve(y_true, scores); roc_auc = auc(fpr_r, tpr_r)

        # Attribution
        attr_c=0; attr_t=0
        with torch.no_grad():
            for a,_,_,l in te_dl:
                embs=model(a.to(DEVICE))[0].cpu()
                for emb, lbl in zip(embs, l):
                    sims=F.cosine_similarity(emb.unsqueeze(0), gallery_mat)
                    if sims.argmax().item()==lbl.item(): attr_c+=1
                    attr_t+=1
        attr_acc=attr_c/attr_t

        progress.progress(90)

        # Store session state
        enc_bundle = {"user_enc":user_enc,"event_enc":event_enc,"pc_enc":pc_enc,"td_scaler":td_scaler,"label_remap":label_remap,"threshold":THRESHOLD}
        st.session_state.update({
            "model": model, "aux_clf": ac, "gallery": gallery_mat, "enc": enc_bundle,
            "threshold": THRESHOLD, "N_CLASSES": N_CLASSES, "N_FEAT": N_FEAT,
            "embed_dim": embed_dim, "hidden_dim": hidden_dim,
            "tr_losses": tr_losses, "va_losses": va_losses,
            "y_true": y_true, "scores": scores, "legit_sims": legit_sims,
            "intruder_sims": intruder_sims, "fpr": fpr_r, "tpr": tpr_r,
            "thr_sweep": thr_sweep, "far_arr": far_a, "frr_arr": frr_a,
            "te_embs": te_embs.numpy(), "te_lbls": te_lbls.numpy(),
            "tr_embs": tr_embs.numpy(), "tr_lbls": tr_lbls.numpy(),
            "metrics": {"accuracy":accuracy,"FAR":FAR,"FRR":FRR,"eer":eer_val,"auc":roc_auc,"attr_acc":attr_acc,"threshold":THRESHOLD}
        })

        progress.progress(100)
        status.markdown('<div class="status-bar">✅ Training complete!</div>', unsafe_allow_html=True)

        # Results
        st.markdown("### Results")
        c1,c2,c3,c4 = st.columns(4)
        for col, label, val in [(c1,"Accuracy",f"{accuracy*100:.2f}%"),(c2,"EER",f"{eer_val*100:.2f}%"),(c3,"ROC AUC",f"{roc_auc:.4f}"),(c4,"Attribution",f"{attr_acc*100:.2f}%")]:
            col.metric(label, val)

        # Save
        st.markdown("### Save model")
        buf_m = io.BytesIO(); torch.save({"model_state_dict":model.state_dict(),"aux_clf_state_dict":ac.state_dict(),"gallery_matrix":gallery_mat,"hyperparams":{"input_dim":N_FEAT,"hidden_dim":hidden_dim,"embed_dim":embed_dim,"window_size":window_size,"n_classes":N_CLASSES,"threshold":THRESHOLD}}, buf_m); buf_m.seek(0)
        buf_e = io.BytesIO(); pickle.dump(enc_bundle, buf_e); buf_e.seek(0)
        cs,ce = st.columns(2)
        cs.download_button("⬇ Download model (.pth)", buf_m, "behavioral_fingerprint_model.pth")
        ce.download_button("⬇ Download encoders (.pkl)", buf_e, "behavioral_fingerprint_encoders.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: VERIFY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Verify Session":
    st.title("Verify Session")
    st.markdown("Paste or upload a session log CSV to check if the activity matches the claimed user.")
    st.markdown("---")

    if "model" not in st.session_state:
        st.warning("No model loaded. Train a model or load a saved checkpoint in the **Train Model** page first.")
        st.stop()

    model      = st.session_state["model"]
    gallery    = st.session_state["gallery"]
    enc        = st.session_state["enc"]
    threshold  = st.session_state["threshold"]
    N_CLASSES  = st.session_state["N_CLASSES"]

    st.markdown(f"**Active threshold (EER):** `{threshold:.4f}`  |  **Users in model:** `{N_CLASSES}`")

    session_file = st.file_uploader("Upload session CSV (same schema as CERT modalities)", type="csv")

    claimed_user = st.text_input("Claimed user ID (as it appears in training data)", placeholder="e.g. ACM2278")

    if session_file and claimed_user and st.button("🔎 Verify", type="primary"):
        try:
            df_s = pd.read_csv(session_file, low_memory=False)
            df_s = normalise_columns(df_s, "session")
            df_s["date"]     = safe_parse_timestamps(df_s["date"])
            df_s             = df_s.dropna(subset=["date","user"])
            df_s["activity"] = df_s["activity"].astype(str).str.strip().str.lower()
            df_s["pc"]       = df_s["pc"].astype(str).str.strip().fillna("unknown_pc")
            df_s["modality"] = "session"

            # Encode with saved encoders
            user_enc  = enc["user_enc"]
            event_enc = enc["event_enc"]
            pc_enc    = enc["pc_enc"]
            td_scaler = enc["td_scaler"]

            if claimed_user not in user_enc.classes_:
                st.error(f"User '{claimed_user}' was not seen during training. Available users: {list(user_enc.classes_[:10])}...")
                st.stop()

            claimed_id = int(user_enc.transform([claimed_user])[0])

            df_s["event_label"] = df_s["modality"] + "*" + df_s["activity"] + "*" + df_s["pc"].astype(str)
            df_s["event_code"]  = df_s["event_label"].apply(lambda x: event_enc.transform([x])[0] if x in event_enc.classes_ else 0)
            df_s = df_s.sort_values("date").reset_index(drop=True)
            df_s["hour"]          = df_s["date"].dt.hour
            df_s["dow"]           = df_s["date"].dt.dayofweek
            df_s["is_weekend"]    = (df_s["dow"] >= 5).astype(int)
            df_s["is_after_hours"]= ((df_s["hour"] < 8) | (df_s["hour"] > 18)).astype(int)
            df_s["time_delta"]    = df_s["date"].diff().dt.total_seconds().div(60).fillna(0).clip(0,1440)
            df_s["log_time_delta"]= np.log1p(df_s["time_delta"])
            df_s["log_td_norm"]   = td_scaler.transform(df_s[["log_time_delta"]]).flatten()
            df_s["hour_norm"]     = df_s["hour"] / 23.0
            df_s["dow_norm"]      = df_s["dow"] / 6.0

            vals = df_s[FEATURE_COLS].values.astype(np.float32)
            if len(vals) < WINDOW_SIZE:
                st.error(f"Session has only {len(vals)} events — need at least {WINDOW_SIZE}.")
                st.stop()

            windows = []
            for s in range(0, len(vals)-WINDOW_SIZE+1, STRIDE): windows.append(vals[s:s+WINDOW_SIZE])
            X = torch.tensor(np.array(windows), dtype=torch.float32).to(DEVICE)

            model.eval()
            with torch.no_grad():
                embs,_ = model(X)
            session_emb = embs.mean(dim=0)

            sims_all = F.cosine_similarity(session_emb.unsqueeze(0), gallery)
            claimed_sim = sims_all[claimed_id].item()
            best_uid    = sims_all.argmax().item()
            best_sim    = sims_all.max().item()
            accepted    = claimed_sim >= threshold

            if accepted:
                st.markdown(f"""<div class="result-accept">
                    <strong style="color:#22c55e;font-size:18px">✅ ACCEPTED — Identity Verified</strong><br><br>
                    Claimed user: <code>{claimed_user}</code> (ID {claimed_id})<br>
                    Cosine similarity: <code>{claimed_sim:.4f}</code> ≥ threshold <code>{threshold:.4f}</code><br>
                    Windows analysed: <code>{len(windows)}</code>
                </div>""", unsafe_allow_html=True)
            else:
                best_name = user_enc.inverse_transform([best_uid])[0]
                st.markdown(f"""<div class="result-reject">
                    <strong style="color:#ef4444;font-size:18px">🚨 REJECTED — Intruder Detected</strong><br><br>
                    Claimed user: <code>{claimed_user}</code> (ID {claimed_id})<br>
                    Similarity to claimed: <code>{claimed_sim:.4f}</code> &lt; threshold <code>{threshold:.4f}</code><br>
                    Best match: <code>{best_name}</code> (similarity <code>{best_sim:.4f}</code>)<br>
                    Windows analysed: <code>{len(windows)}</code>
                </div>""", unsafe_allow_html=True)

            # Similarity bar chart
            st.markdown("#### Similarity to all gallery users")
            top_n = 15
            top_idx  = sims_all.cpu().numpy().argsort()[-top_n:][::-1]
            top_sims = sims_all.cpu().numpy()[top_idx]
            top_names= [user_enc.inverse_transform([i])[0] for i in top_idx]
            fig, ax = plt.subplots(figsize=(10,3)); fig.patch.set_facecolor("#0d0f14"); ax.set_facecolor("#151820")
            colors = ["#22c55e" if i==claimed_id else "#ef4444" if i==best_uid and not accepted else "#38bdf8" for i in top_idx]
            ax.barh(top_names[::-1], top_sims[::-1], color=colors[::-1])
            ax.axvline(threshold, color="#facc15", linestyle="--", linewidth=1, label=f"Threshold={threshold:.3f}")
            ax.set_xlabel("Cosine similarity", color="#94a3b8"); ax.tick_params(colors="#94a3b8"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            for s in ["bottom","left"]: ax.spines[s].set_color("#1e2130")
            ax.legend(facecolor="#151820", labelcolor="#94a3b8")
            st.pyplot(fig); plt.close()

        except Exception as ex:
            st.error(f"Error during verification: {ex}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualize Results":
    st.title("Visualize Results")
    st.markdown("Charts generated from the last training run.")
    st.markdown("---")

    if "metrics" not in st.session_state:
        st.warning("No results yet. Run training first on the **Train Model** page.")
        st.stop()

    m = st.session_state["metrics"]
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col,(label,val) in zip([c1,c2,c3,c4,c5,c6],[
        ("Accuracy",f"{m['accuracy']*100:.2f}%"),("FAR",f"{m['FAR']*100:.2f}%"),
        ("FRR",f"{m['FRR']*100:.2f}%"),("EER",f"{m['eer']*100:.2f}%"),
        ("AUC",f"{m['auc']:.4f}"),("Attribution",f"{m['attr_acc']*100:.2f}%")]):
        col.metric(label, val)

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Loss curve", "ROC curve", "Confusion matrix", "Similarity dist.", "t-SNE"])

    BG = "#0d0f14"; CARD = "#151820"; BORDER = "#1e2130"; TXT = "#94a3b8"

    def styled_fig(w=8, h=4):
        fig, ax = plt.subplots(figsize=(w,h)); fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        ax.tick_params(colors=TXT); ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        for s in ["bottom","left"]: ax.spines[s].set_color(BORDER)
        return fig, ax

    with tab1:
        fig, ax = styled_fig()
        ax.plot(st.session_state["tr_losses"], label="Train", color="#38bdf8", lw=2)
        ax.plot(st.session_state["va_losses"], label="Val",   color="#f472b6", lw=2, linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(facecolor=CARD, labelcolor=TXT)
        ax.set_title("Training & Validation Loss", color=TXT, pad=12)
        st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = styled_fig(6,5)
        ax.plot(st.session_state["fpr"], st.session_state["tpr"], color="#38bdf8", lw=2, label=f"AUC = {m['auc']:.3f}")
        ax.plot([0,1],[0,1], color="#475569", lw=1, linestyle="--")
        ax.scatter([m["FAR"]],[1-m["FRR"]], color="#ef4444", zorder=5, label=f"EER op. point θ={m['threshold']:.3f}")
        ax.set_xlabel("FAR"); ax.set_ylabel("TAR (1-FRR)"); ax.legend(facecolor=CARD, labelcolor=TXT)
        ax.set_title("ROC Curve", color=TXT, pad=12)
        st.pyplot(fig); plt.close()

    with tab3:
        y_true = st.session_state["y_true"]; scores = st.session_state["scores"]
        y_pred = (scores >= m["threshold"]).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[1,0])
        fig, ax = styled_fig(6,5)
        im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Legitimate","Intruder"], color=TXT)
        ax.set_yticklabels(["Legitimate","Intruder"], color=TXT)
        ax.set_xlabel("Predicted", color=TXT); ax.set_ylabel("Actual", color=TXT)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
                ax.text(j, i+0.3, f"({cm[i,j]/cm.sum()*100:.1f}%)", ha="center", va="center", color="#94a3b8", fontsize=9)
        ax.set_title(f"Confusion Matrix  θ={m['threshold']:.4f}", color=TXT, pad=12)
        st.pyplot(fig); plt.close()

    with tab4:
        fig, ax = styled_fig(8,4)
        ax.hist(st.session_state["legit_sims"],    bins=40, alpha=0.7, color="#22c55e", label="Legitimate")
        ax.hist(st.session_state["intruder_sims"], bins=40, alpha=0.7, color="#ef4444", label="Intruder")
        ax.axvline(m["threshold"], color="#facc15", linestyle="--", lw=1.5, label=f"EER θ={m['threshold']:.4f}")
        ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Count"); ax.legend(facecolor=CARD, labelcolor=TXT)
        ax.set_title("Similarity Score Distribution", color=TXT, pad=12)
        st.pyplot(fig); plt.close()

    with tab5:
        st.info("t-SNE may take 30–60 seconds for large datasets.")
        MAX_VIS_USERS = st.slider("Max users to display", 5, 20, 12)
        MAX_VIS_PTS   = st.slider("Max points", 200, 1000, 600)
        if st.button("Run t-SNE"):
            tr_embs = st.session_state["tr_embs"]; tr_lbls = st.session_state["tr_lbls"]
            freq_users = np.array([uid for uid,_ in Counter(tr_lbls).most_common(MAX_VIS_USERS)])
            mask = np.isin(tr_lbls, freq_users)
            vis_e, vis_l = tr_embs[mask], tr_lbls[mask]
            if len(vis_e) > MAX_VIS_PTS:
                idx = np.random.choice(len(vis_e), MAX_VIS_PTS, replace=False)
                vis_e, vis_l = vis_e[idx], vis_l[idx]
            with st.spinner(f"Running t-SNE on {len(vis_e)} points…"):
                tsne = TSNE(n_components=2, perplexity=min(30, len(vis_e)//4), random_state=SEED, n_iter=1000)
                e2d  = tsne.fit_transform(vis_e)
            remap = {old:new for new,old in enumerate(freq_users)}
            vis_l2 = np.array([remap[l] for l in vis_l])
            K = len(freq_users); cmap = plt.cm.get_cmap("tab20", K)
            fig, ax = styled_fig(10,7)
            for k in range(K):
                mk = vis_l2==k
                ax.scatter(e2d[mk,0], e2d[mk,1], color=cmap(k), label=f"User {freq_users[k]}", alpha=0.7, s=18, edgecolors="none")
            ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
            ax.set_title("t-SNE of Behavioral Fingerprint Embeddings", color=TXT, pad=12)
            ax.legend(loc="upper right", fontsize=7, ncol=2, facecolor=CARD, labelcolor=TXT, markerscale=1.5)
            st.pyplot(fig); plt.close()

        # FAR/FRR sweep
        st.markdown("#### FAR vs FRR threshold sweep")
        fig, ax = styled_fig(8,4)
        ax.plot(st.session_state["thr_sweep"], st.session_state["far_arr"], color="#ef4444", label="FAR")
        ax.plot(st.session_state["thr_sweep"], st.session_state["frr_arr"], color="#38bdf8", label="FRR")
        ax.axvline(m["threshold"], color="#facc15", linestyle="--", lw=1.5, label=f"EER θ={m['threshold']:.4f}")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Error rate"); ax.legend(facecolor=CARD, labelcolor=TXT)
        ax.set_title("FAR vs FRR Sweep", color=TXT, pad=12)
        st.pyplot(fig); plt.close()
