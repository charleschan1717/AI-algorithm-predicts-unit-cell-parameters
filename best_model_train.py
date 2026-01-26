import os, math, random
import numpy as np
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NPZ_FILE   = 'train_processed.npz'
MODEL_SAVE_PATH = 'unicrystal_best_by_score.pt'

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
VAL_RATIO = 0.10

# 2θ / 物理常量
WAVELENGTH = 1.5406      # Cu Kα1 (Å)
TTH_MIN, TTH_MAX = 5.0, 80.0
DELTA_TTH = 0.10         # 贪心匹配阈值（度）
INSTR_RADIUS_MM = 240.0  # 仪器半径（样品偏移修正）

# 偏移预测范围与损失权重
MAX_ZERO_SHIFT = 0.5     # 预测零漂范围 ±0.5°
MAX_SAMPLE_DISP = 0.2    # 预测样偏范围 ±0.2 mm
SHIFT_LOSS_W = 0.075
DISP_LOSS_W = 0.075

# 任务权重
CELL_LOSS_W = 0.55
TTH_LOSS_W  = 0.30
HKL_LOSS_W  = 0.05       # 可设为0关闭 HKL 监督
ANG_CE_W    = 0.05       # 角度规划分类损失权重（可设为0关闭）

# HKL 词表最小频数
MIN_HKL_FREQ = 2

# 其它
GRAD_CLIP_NORM = 2.0
USE_AMP = True

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def apply_zero_shift(theta_deg: np.ndarray, intensity: np.ndarray, max_shift=MAX_ZERO_SHIFT):
    if random.random() < 0.7:
        shift = random.uniform(-max_shift, max_shift)  # deg
        x = theta_deg; y = intensity
        shifted = np.interp(x, x - shift, y, left=y[0], right=y[-1])
        return shifted, shift
    return intensity, 0.0

def apply_sample_displacement(theta_deg: np.ndarray, intensity: np.ndarray, max_disp=MAX_SAMPLE_DISP):
    # Δ(2θ) ≈ - (2*s/R) * cosθ  (弧度)
    if random.random() < 0.7:
        s = random.uniform(-max_disp, max_disp)  # mm
        theta_rad = np.deg2rad(theta_deg / 2.0)
        d2th_rad = - (2.0 * s / INSTR_RADIUS_MM) * np.cos(theta_rad)
        d2th_deg = np.rad2deg(d2th_rad)
        x = theta_deg; y = intensity
        warped = np.interp(x, x - d2th_deg, y, left=y[0], right=y[-1])
        return warped, s
    return intensity, 0.0

def apply_peak_broadening(intensity: np.ndarray, p=0.5):
    if random.random() > p:
        return intensity
    w = random.randint(3, 13)
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(intensity, kernel, mode='same')

def add_impurity_peaks(intensity: np.ndarray, p=0.3):
    if random.random() > p:
        return intensity
    x = np.arange(len(intensity))
    out = intensity.copy()
    k = random.randint(1, 3)
    for _ in range(k):
        pos = random.randint(int(0.1*len(x)), int(0.9*len(x)))
        amp = random.uniform(0.05, 0.3) * (intensity.max() + 1e-8)
        width = random.randint(10, 50)
        out += amp * np.exp(-((x - pos)**2) / (2.0 * width**2))
    return out

class XRDDataset(Dataset):
    def __init__(self, X, Y_cell, Y_hkl, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_cell = torch.tensor(Y_cell, dtype=torch.float32)
        self.Y_hkl = torch.tensor(Y_hkl, dtype=torch.float32)
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone().numpy()
        theta_axis = np.linspace(TTH_MIN, TTH_MAX, x.shape[-1])

        zshift = 0.0
        sdisp  = 0.0
        if self.augment:
            x, z = apply_zero_shift(theta_axis, x, max_shift=MAX_ZERO_SHIFT); zshift += z
            x, s = apply_sample_displacement(theta_axis, x, max_disp=MAX_SAMPLE_DISP); sdisp += s
            x = apply_peak_broadening(x); x = add_impurity_peaks(x)
        x[x < 0] = 0
        mx = x.max()
        if mx > 0: x = x / mx
        x = torch.tensor(x, dtype=torch.float32)
        y_cell = self.Y_cell[idx]
        y_hkl  = self.Y_hkl[idx]
        return x.unsqueeze(0), y_cell, y_hkl, torch.tensor([zshift], dtype=torch.float32), torch.tensor([sdisp], dtype=torch.float32)

def build_hkl_vocab(hkl_lists, min_freq: int = MIN_HKL_FREQ):
    from collections import defaultdict
    freq = defaultdict(int)
    for peaks in hkl_lists:
        for p in peaks:
            try:
                hkl = tuple(p[3])
                freq[hkl] += 1
            except Exception:
                continue
    vocab = [h for h, c in freq.items() if c >= min_freq]
    vocab.sort()
    hkl_to_idx = {h:i for i,h in enumerate(vocab)}
    freq_vec = np.array([freq[h] for h in vocab], dtype=np.float32)
    return hkl_to_idx, freq_vec

def encode_multi_hot(peaks, hkl_to_idx, vocab_size: int):
    v = np.zeros(vocab_size, dtype=np.float32)
    for p in peaks:
        try:
            idx = hkl_to_idx.get(tuple(p[3]), None)
            if idx is not None:
                v[idx] = 1.0
        except Exception:
            pass
    return v

# ==================== 模型框架：CrystalNeXtT ====================

class ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp1 = nn.Conv1d(channels, hidden, 1, bias=False)
        self.mlp2 = nn.Conv1d(hidden, channels, 1, bias=False)
        # 恒等友好初始化：让初始 mask≈0.5，避免无残差时整体衰减
        nn.init.zeros_(self.mlp1.weight)
        nn.init.zeros_(self.mlp2.weight)

    def forward(self, x):  # (B,C,L)
        avg = F.adaptive_avg_pool1d(x, 1)
        mx  = F.adaptive_max_pool1d(x, 1)
        h   = F.relu(self.mlp1(avg)) + F.relu(self.mlp1(mx))
        m   = torch.sigmoid(self.mlp2(h))   # (B,C,1) in (0,1)
        return m


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size: int = 11):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=pad, bias=False)
        nn.init.zeros_(self.conv.weight)    # 初始≈0.5
    def forward(self, x):  # (B,C,L)
        avg = torch.mean(x, dim=1, keepdim=True)   # (B,1,L)
        mx  = torch.amax(x, dim=1, keepdim=True)   # (B,1,L)
        m   = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B,1,L)
        return m


class CBAM1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_ks: int = 11, gate_scale: float = 0.2):
        super().__init__()
        self.ca = ChannelAttention1D(channels, reduction)
        self.sa = SpatialAttention1D(spatial_ks)
        self.gate_scale = gate_scale

    def _gate(self, x, mask):
        return x * (1.0 + self.gate_scale * (2.0*mask - 1.0))

    def forward(self, x):  # (B,C,L)
        mc = self.ca(x)    # (B,C,1)
        x  = self._gate(x, mc)
        ms = self.sa(x)    # (B,1,L)
        x  = self._gate(x, ms)
        return x

class ECA1D(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1, gate_scale: float = 0.2):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else (k + 1)                 # odd kernel size
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        nn.init.zeros_(self.conv.weight)            # 让 sigmoid(0)=0.5，初始不缩放
        self.gate_scale = gate_scale

    def forward(self, x):                           # x: (B,C,L)
        y = self.pool(x)                            # (B,C,1)
        y = y.permute(0, 2, 1)                      # (B,1,C)
        y = self.conv(y)                            # (B,1,C)
        y = y.permute(0, 2, 1)                      # (B,C,1)
        m = torch.sigmoid(y)                        # (B,C,1) ∈ (0,1)
        return x * (1.0 + self.gate_scale * (2.0*m - 1.0))


class LayerNorm1d(nn.Module):
    """Channels-first LayerNorm for 1D (B,C,L)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):
        # x: (B,C,L) -> (B,L,C) -> LN -> (B,C,L)
        x = x.permute(0,2,1)
        x = self.ln(x)
        return x.permute(0,2,1)

class MultiDilatedDWConv1d(nn.Module):
    def __init__(self, channels, kernel_size=7, dilations=(1,3,5)):
        super().__init__()
        pad = lambda k, d: d * (k//2)
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, padding=pad(kernel_size,d), dilation=d, groups=channels)
            for d in dilations
        ])
        self.proj = nn.Conv1d(channels, channels, 1)
    def forward(self, x):
        y = 0
        for conv in self.branches:
            y = y + conv(x)
        y = self.proj(y)
        return y

class ConvNeXtBlock1D(nn.Module):
    """
    DW/MD-DW -> LN -> ECA -> PW(×4)->GELU->PW -> gamma -> drop/residual
    """
    def __init__(self, channels, kernel_size=7, drop_path=0.0, use_md_dw=True):
        super().__init__()
        self.dw = (MultiDilatedDWConv1d(channels, kernel_size) if use_md_dw
                   else nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels))
        self.ln   = LayerNorm1d(channels)
        #用 ECA,备选CBAM、loord
        self.eca  = ECA1D(channels, gamma=2, b=1, gate_scale=0.2)

        self.pw1  = nn.Conv1d(channels, 4*channels, 1)
        self.act  = nn.GELU()
        self.pw2  = nn.Conv1d(4*channels, channels, 1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        y = self.dw(x)
        y = self.ln(y)
        y = self.eca(y)                          # 注意力在 MLP 之前
        y = self.pw2(self.act(self.pw1(y)))
        y = self.gamma * y
        
        try:
            return x + self.drop_path(y)
        except:
            return self.drop_path(y)

class DropPath(nn.Module):
    """Stochastic Depth."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,)*(x.ndim-1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep) * random_tensor

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, seq_len: int):
        return self.pe[:, :seq_len, :]

class CrystalNeXtT(nn.Module):
    """
    ConvNeXt1D batch + Transformer encoder
    pred_cell(6), hkl_logits/None, ang_logits(3x3), ang_reg(3), zero(1), disp(1)
    """
    def __init__(self, num_hkl_classes, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        C = d_model

        # Stem：更强的下采样与通道扩展（1->C/4->C/2->C）
        self.stem = nn.Sequential(
            nn.Conv1d(1,     C//4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(C//4), nn.GELU(),
            nn.Conv1d(C//4, C//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(C//2), nn.GELU(),
            nn.Conv1d(C//2, C,    kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(C), nn.GELU(),
        )

        # ConvNeXt stages（总 8 个 Block，含多膨胀 DWConv）
        dp = [0.0, 0.02, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06]
        blocks = []
        for i in range(8):
            blocks.append(ConvNeXtBlock1D(C, kernel_size=7, drop_path=dp[i], use_md_dw=True))
        self.cnx = nn.Sequential(*blocks)

        # Transformer 编码器（轻量 4 层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=C, nhead=nhead, dim_feedforward=C*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, C))
        self.pos_encoder = SinusoidalPE(d_model=C, max_len=4096)

        # Heads
        self.head_ln = nn.LayerNorm(C)
        self.cell_head = nn.Linear(C, 6)
        self.hkl_head  = nn.Linear(C, num_hkl_classes) if num_hkl_classes>0 else None
        self.ang_gate_head = nn.Linear(C, 9)
        self.zero_head = nn.Linear(C, 1)
        self.disp_head = nn.Linear(C, 1)

        self.register_buffer('hkl_bias', None)

    def set_hkl_prior_bias(self, bias: torch.Tensor):
        self.hkl_bias = bias.view(1, -1).float()

    def forward(self, x):
        # x: (B,1,L)
        x = self.stem(x)          # (B,C,L')
        x = self.cnx(x)           # (B,C,L')
        x = x.permute(0,2,1)      # (B,L',C)

        B, L, C = x.shape
        cls_tokens = self.cls_token.expand(B, 1, C)
        x = torch.cat((cls_tokens, x), dim=1)  # (B,L'+1,C)
        x = x + self.pos_encoder(x.size(1))
        feats = self.transformer_encoder(x)
        cls_feat = self.head_ln(feats[:, 0])   # (B,C)

        # cell + 角度规划
        raw_cell = self.cell_head(cls_feat)           # (B,6)
        len_pos  = F.softplus(raw_cell[:, :3]) + 1e-3
        ang_reg  = torch.sigmoid(raw_cell[:, 3:]) * 180.0

        ang_logits = self.ang_gate_head(cls_feat).view(B, 3, 3)
        ang_prob   = torch.softmax(ang_logits, dim=-1)
        anchor90   = torch.full_like(ang_reg, 90.0)
        anchor120  = torch.full_like(ang_reg, 120.0)
        ang_final  = ang_prob[..., 0]*anchor90 + ang_prob[..., 1]*anchor120 + ang_prob[..., 2]*ang_reg

        pred_cell = torch.cat([len_pos, ang_final], dim=1)

        # HKL logits（可选）
        if self.hkl_head is not None:
            logits_hkl = self.hkl_head(cls_feat)
            if self.hkl_bias is not None:
                logits_hkl = logits_hkl + self.hkl_bias
        else:
            logits_hkl = None

        # 偏移
        zero_pred = torch.tanh(self.zero_head(cls_feat)) * MAX_ZERO_SHIFT
        disp_pred = torch.tanh(self.disp_head(cls_feat)) * MAX_SAMPLE_DISP

        return pred_cell, logits_hkl, ang_logits, ang_reg, zero_pred, disp_pred

# ==================== 物理/几何 与 打分实现 ====================
def reciprocal_metric(a,b,c, alpha,beta,gamma):
    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    ca, cb, cg = math.cos(ar), math.cos(br), math.cos(gr)
    G = np.array([
        [a*a, a*b*cg, a*c*cb],
        [a*b*cg, b*b,   b*c*ca],
        [a*c*cb, b*c*ca, c*c]
    ], dtype=np.float64)
    Gstar = np.linalg.inv(G)
    return Gstar

def enumerate_hkls(hmax=8):
    hkls = []
    for h in range(-hmax, hmax+1):
        for k in range(-hmax, hmax+1):
            for l in range(-hmax, hmax+1):
                if h==k==l==0: continue
                hkls.append((h,k,l))
    return np.array(hkls, dtype=np.int32)

HKLS_TABLE = enumerate_hkls(8)

def tth_from_cell_numpy(cell6, wavelength=WAVELENGTH, tth_max=TTH_MAX):
    a,b,c, alpha,beta,gamma = cell6
    Gstar = reciprocal_metric(a,b,c, alpha,beta,gamma)
    H = HKLS_TABLE.astype(np.float64)
    inv_d2 = np.sum(H @ Gstar * H, axis=1)  # (M,)
    valid = inv_d2 > 0
    inv_d = np.sqrt(inv_d2[valid])
    d = 1.0 / inv_d
    s = np.clip(wavelength / (2.0*d), 0.0, 1.0)
    tth = 2.0 * np.degrees(np.arcsin(s))
    tth = np.sort(tth[(tth >= TTH_MIN) & (tth <= tth_max)])
    return tth

def apply_offsets_to_tth(tth_pred: np.ndarray, zero_shift_deg: float, sample_disp_mm: float):
    tt = tth_pred + zero_shift_deg
    theta_rad = np.radians(tt / 2.0)
    d2th_rad = - (2.0 * sample_disp_mm / INSTR_RADIUS_MM) * np.cos(theta_rad)
    tt = tt + np.degrees(d2th_rad)
    return tt

def greedy_match_accuracy_numpy(tth_pred: np.ndarray, tth_gt: np.ndarray, delta=DELTA_TTH) -> float:
    i = j = matched = 0
    tth_pred = np.sort(tth_pred)
    tth_gt   = np.sort(tth_gt)
    while i < len(tth_gt) and j < len(tth_pred):
        d = tth_pred[j] - tth_gt[i]
        if abs(d) <= delta:
            matched += 1; i += 1; j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return matched / max(1, len(tth_gt))

def chamfer_huber_tth_loss(tth_pred_t: torch.Tensor, tth_gt_t: torch.Tensor, delta=DELTA_TTH):
    if tth_pred_t.numel()==0 or tth_gt_t.numel()==0:
        return torch.tensor(0.0, device=tth_pred_t.device)
    P = tth_pred_t[:,None]
    G = tth_gt_t[None,:]
    dist = torch.abs(P - G)
    d_g = dist.min(dim=0).values
    d_p = dist.min(dim=1).values
    def huber(x, d=delta):
        quad = 0.5 * (x**2) / d
        lin  = x - 0.5*d
        return torch.where(x <= d, quad, lin)
    return huber(d_g).mean() + huber(d_p).mean()

def angle_gate_targets(y_cell: torch.Tensor, tol90=2.0, tol120=2.0) -> torch.Tensor:
    angles = y_cell[:, 3:6]  # (B,3)
    tgt = torch.full((angles.size(0), 3), 2, dtype=torch.long, device=angles.device)  # default free=2
    for i in range(3):
        ai = angles[:, i]
        tgt[:, i] = torch.where(torch.abs(ai - 90.0) <= tol90, torch.zeros_like(ai, dtype=torch.long), tgt[:, i])
        tgt[:, i] = torch.where(torch.abs(ai - 120.0) <= tol120, torch.ones_like(ai, dtype=torch.long), tgt[:, i])
    return tgt  # (B,3)

# ==================== 训练/验证 ====================
def run_epoch(model, loader, optimizer=None, scheduler=None, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_cell = total_tth = total_hkl = total_zero = total_disp = total_ang = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP) if is_train else None

    pbar = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
    for (x, y_cell, y_hkl, y_zero, y_disp) in pbar:
        x      = x.to(DEVICE)
        y_cell = y_cell.to(DEVICE)
        y_hkl  = y_hkl.to(DEVICE)
        y_zero = y_zero.to(DEVICE)
        y_disp = y_disp.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(x)
            pred_cell, hkl_logits, ang_logits, ang_reg, pred_zero, pred_disp = outputs

            # L_cell
            l_cell = F.mse_loss(pred_cell, y_cell)

            # L_tth
            l_tth_batch = []
            for i in range(x.size(0)):
                c_pred = pred_cell[i].detach().cpu().numpy()
                c_true = y_cell[i].detach().cpu().numpy()
                tth_pred = tth_from_cell_numpy(c_pred)
                tth_pred = apply_offsets_to_tth(tth_pred,
                                                float(pred_zero[i].detach().cpu().numpy()),
                                                float(pred_disp[i].detach().cpu().numpy()))
                tth_true = tth_from_cell_numpy(c_true)
                tth_pred_t = torch.tensor(tth_pred, dtype=torch.float32, device=DEVICE)
                tth_true_t = torch.tensor(tth_true, dtype=torch.float32, device=DEVICE)
                l_tth_batch.append(chamfer_huber_tth_loss(tth_pred_t, tth_true_t, delta=DELTA_TTH))
            l_tth = torch.stack(l_tth_batch).mean() if l_tth_batch else torch.tensor(0.0, device=DEVICE)

            # 偏移 L1
            l_zero = F.l1_loss(pred_zero, y_zero)
            l_disp = F.l1_loss(pred_disp, y_disp)

            # HKL BCE（可选）
            if hkl_logits is not None and y_hkl.numel()>0 and HKL_LOSS_W>0:
                l_hkl = F.binary_cross_entropy_with_logits(hkl_logits, y_hkl)
            else:
                l_hkl = torch.tensor(0.0, device=DEVICE)

            # 角度规划 CE（可选）
            if ANG_CE_W > 0:
                tgt = angle_gate_targets(y_cell)           # (B,3) in {0,1,2}
                l_ang = 0.0
                for i in range(3):
                    l_ang += F.cross_entropy(ang_logits[:, i, :], tgt[:, i])
                l_ang = l_ang / 3.0
            else:
                l_ang = torch.tensor(0.0, device=DEVICE)

            loss = CELL_LOSS_W*l_cell + TTH_LOSS_W*l_tth + SHIFT_LOSS_W*l_zero + DISP_LOSS_W*l_disp \
                   + HKL_LOSS_W*l_hkl + ANG_CE_W*l_ang

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if GRAD_CLIP_NORM is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.detach())
        total_cell += float(l_cell.detach())
        total_tth  += float(l_tth.detach())
        total_zero += float(l_zero.detach())
        total_disp += float(l_disp.detach())
        total_hkl  += float(l_hkl.detach())
        total_ang  += float(l_ang.detach())
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1e-9):.4f}")

    n = len(loader)
    return (total_loss/n, total_cell/n, total_tth/n, total_zero/n, total_disp/n, total_hkl/n, total_ang/n)

@torch.no_grad()
def evaluate_score(model, loader):
    """RMSE+ AvgAcc"""
    sse_sum = 0.0
    n_samples = 0
    acc_sum = 0.0

    for (x, y_cell, y_hkl, y_zero, y_disp) in tqdm(loader, desc="Scoring", leave=False):
        x = x.to(DEVICE)
        outputs = model(x)
        pred_cell, _, _, _, pred_zero, pred_disp = outputs

        for i in range(x.size(0)):
            c_pred = pred_cell[i].cpu().numpy().astype(np.float64)
            c_true = y_cell[i].numpy().astype(np.float64)
            # RMSE: 角度按弧度
            cp = c_pred.copy(); ct = c_true.copy()
            cp[3:6] = np.radians(cp[3:6]); ct[3:6] = np.radians(ct[3:6])
            sse = float(np.sum((cp - ct)**2))
            sse_sum += sse; n_samples += 1

            # 2θ 匹配
            tth_p = tth_from_cell_numpy(c_pred)
            tth_p = apply_offsets_to_tth(tth_p,
                                         float(pred_zero[i].cpu().numpy()),
                                         float(pred_disp[i].cpu().numpy()))
            tth_g = tth_from_cell_numpy(c_true)
            acc = greedy_match_accuracy_numpy(tth_p, tth_g, delta=DELTA_TTH)
            acc_sum += acc

    rmse = math.sqrt(sse_sum / max(1, n_samples))
    avg_acc = acc_sum / max(1, n_samples)
    score = 0.9*avg_acc - 0.1*rmse
    return rmse, avg_acc, score

# ==================== main ====================
def main():
    seed_everything(SEED)
    print(f"Device: {DEVICE}")

    # 读取数据并划分
    d = np.load(NPZ_FILE, allow_pickle=True)
    X_all, y_cell_all = d['X'], d['y_cell']
    # y_hkl 可选
    y_hkl_all = d['y_hkl'] if 'y_hkl' in d.files else np.zeros((len(X_all), 0), dtype=np.float32)

    n = len(X_all); assert len(y_cell_all)==n
    rng = np.random.default_rng(SEED); idx = np.arange(n); rng.shuffle(idx)
    split = int(n * (1.0 - VAL_RATIO))
    idx_tr, idx_val = idx[:split], idx[split:]
    X_tr, y_cell_tr_raw, y_hkl_tr_raw   = X_all[idx_tr],  y_cell_all[idx_tr],  y_hkl_all[idx_tr]
    X_val, y_cell_val_raw, y_hkl_val_raw = X_all[idx_val], y_cell_all[idx_val], y_hkl_all[idx_val]

    # HKL 词表基于训练划分
    if y_hkl_tr_raw.ndim == 1 or (len(y_hkl_tr_raw)>0 and isinstance(y_hkl_tr_raw[0], list)):
        hkl_to_idx, _ = build_hkl_vocab(y_hkl_tr_raw, MIN_HKL_FREQ)
        vocab_size = len(hkl_to_idx)
        y_hkl_tr  = np.array([encode_multi_hot(p, hkl_to_idx, vocab_size) for p in y_hkl_tr_raw], dtype=np.float32) if vocab_size>0 else np.zeros((len(X_tr),0),dtype=np.float32)
        y_hkl_val = np.array([encode_multi_hot(p, hkl_to_idx, vocab_size) for p in y_hkl_val_raw], dtype=np.float32) if vocab_size>0 else np.zeros((len(X_val),0),dtype=np.float32)
    else:
        vocab_size = y_hkl_tr_raw.shape[1] if y_hkl_tr_raw.size>0 else 0
        y_hkl_tr, y_hkl_val = y_hkl_tr_raw, y_hkl_val_raw

    train_ds = XRDDataset(X_tr,  y_cell_tr_raw,  y_hkl_tr,  augment=True)
    val_ds   = XRDDataset(X_val, y_cell_val_raw, y_hkl_val, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    model = CrystalNeXtT(num_hkl_classes=vocab_size).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    total_steps  = steps_per_epoch * EPOCHS
    sched_warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    sched_cos  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)
    scheduler  = torch.optim.lr_scheduler.SequentialLR(optimizer, [sched_warm, sched_cos], milestones=[warmup_steps])

    best_score = -1e9

    for epoch in range(1, EPOCHS+1):
        tr = run_epoch(model, train_loader, optimizer=optimizer, scheduler=scheduler, is_train=True)
        val = run_epoch(model, val_loader, optimizer=None, scheduler=None, is_train=False)

        rmse, avg_acc, score = evaluate_score(model, val_loader)
        print(f"\nEpoch {epoch}/{EPOCHS} "
              f"| ValLoss {val[0]:.4f} | Cell {val[1]:.4f} | tth {val[2]:.4f} | zero {val[3]:.4f} | disp {val[4]:.4f} | hkl {val[5]:.4f} | ang {val[6]:.4f} "
              f"| RMSE {rmse:.4f} | AvgAcc {avg_acc:.4f} | Score {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'WAVELENGTH': WAVELENGTH,
                    'DELTA_TTH': DELTA_TTH,
                    'INSTR_RADIUS_MM': INSTR_RADIUS_MM,
                    'ARCH': 'CrystalNeXtT'
                }
            }, MODEL_SAVE_PATH)
            print(f"New best-by-score saved to {MODEL_SAVE_PATH}")

    print(f"\n训练结束。Best Score = {best_score:.4f}，权重已保存到 {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
