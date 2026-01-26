import argparse
import json
import math
import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 全局配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 1.5406      # Å (Cu Kα1)
TTH_MIN, TTH_MAX = 5.0, 80.0

# TTA配置
TTA_SHIFTS = [-0.06, -0.03, 0.0, +0.03, +0.06]

# 角度吸附配置
SNAP_ANGLES = True
SNAP_PROB_TH = 0.60
SNAP_DIST_90 = 2.0
SNAP_DIST_120 = 2.5

# 预测范围
MAX_ZERO_SHIFT = 0.5
MAX_SAMPLE_DISP = 0.2

#Model
class ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp1 = nn.Conv1d(channels, hidden, 1, bias=False)
        self.mlp2 = nn.Conv1d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.mlp1.weight)
        nn.init.zeros_(self.mlp2.weight)

    def forward(self, x):
        avg = F.adaptive_avg_pool1d(x, 1)
        mx  = F.adaptive_max_pool1d(x, 1)
        h   = F.relu(self.mlp1(avg)) + F.relu(self.mlp1(mx))
        m   = torch.sigmoid(self.mlp2(h))
        return m

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size: int = 11):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=pad, bias=False)
        nn.init.zeros_(self.conv.weight)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.amax(x, dim=1, keepdim=True)
        m   = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return m

class ECA1D(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1, gate_scale: float = 0.2):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else (k + 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        nn.init.zeros_(self.conv.weight)
        self.gate_scale = gate_scale

    def forward(self, x):
        y = self.conv(self.pool(x).permute(0, 2, 1)).permute(0, 2, 1)
        m = torch.sigmoid(y)
        return x * (1.0 + self.gate_scale * (2.0 * m - 1.0))

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):
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
        y = sum(conv(x) for conv in self.branches)
        return self.proj(y)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,)*(x.ndim-1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep) * random_tensor

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7, drop_path=0.0, use_md_dw=True):
        super().__init__()
        self.dw = (MultiDilatedDWConv1d(channels, kernel_size) if use_md_dw
                   else nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels))
        self.ln   = LayerNorm1d(channels)
        self.eca  = ECA1D(channels, gamma=2, b=1, gate_scale=0.2)
        self.pw1  = nn.Conv1d(channels, 4*channels, 1)
        self.act  = nn.GELU()
        self.pw2  = nn.Conv1d(4*channels, channels, 1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        y = self.dw(x)
        y = self.ln(y)
        y = self.eca(y)
        y = self.pw2(self.act(self.pw1(y)))
        return x + self.drop_path(self.gamma * y)

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
    def __init__(self, num_hkl_classes, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        C = d_model
        self.stem = nn.Sequential(
            nn.Conv1d(1,     C//4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(C//4), nn.GELU(),
            nn.Conv1d(C//4, C//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(C//2), nn.GELU(),
            nn.Conv1d(C//2, C,    kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(C), nn.GELU(),
        )
        dp = [0.0, 0.02, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06]
        blocks = [ConvNeXtBlock1D(C, kernel_size=7, drop_path=dp[i], use_md_dw=True) for i in range(8)]
        self.cnx = nn.Sequential(*blocks)

        encoder_layer = nn.TransformerEncoderLayer(d_model=C, nhead=nhead, dim_feedforward=C*4,
                                                 dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, C))
        self.pos_encoder = SinusoidalPE(d_model=C, max_len=4096)

        self.head_ln = nn.LayerNorm(C)
        self.cell_head = nn.Linear(C, 6)
        self.hkl_head  = nn.Linear(C, num_hkl_classes) if num_hkl_classes>0 else None
        self.ang_gate_head = nn.Linear(C, 9)
        self.zero_head = nn.Linear(C, 1)
        self.disp_head = nn.Linear(C, 1)
        self.register_buffer('hkl_bias', None)

    def forward(self, x):
        x = self.stem(x)
        x = self.cnx(x)
        x = x.permute(0,2,1)
        B, L, C = x.shape
        cls_tokens = self.cls_token.expand(B, 1, C)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_encoder(x.size(1))
        feats = self.transformer_encoder(x)
        cls_feat = self.head_ln(feats[:, 0])

        raw_cell = self.cell_head(cls_feat)
        len_pos  = F.softplus(raw_cell[:, :3]) + 1e-3
        ang_reg  = torch.sigmoid(raw_cell[:, 3:]) * 180.0

        ang_logits = self.ang_gate_head(cls_feat).view(B, 3, 3)
        ang_prob   = torch.softmax(ang_logits, dim=-1)
        
        # 角度混合: 0->90°, 1->120°, 2->Regression
        anchor90   = torch.full_like(ang_reg, 90.0)
        anchor120  = torch.full_like(ang_reg, 120.0)
        ang_final  = ang_prob[..., 0]*anchor90 + ang_prob[..., 1]*anchor120 + ang_prob[..., 2]*ang_reg
        pred_cell = torch.cat([len_pos, ang_final], dim=1)

        logits_hkl = self.hkl_head(cls_feat) + self.hkl_bias if (self.hkl_head and self.hkl_bias) else None
        zero_pred = torch.tanh(self.zero_head(cls_feat)) * MAX_ZERO_SHIFT
        disp_pred = torch.tanh(self.disp_head(cls_feat)) * MAX_SAMPLE_DISP

        return pred_cell, logits_hkl, ang_logits, ang_reg, zero_pred, disp_pred

# ==================== 物理工具与推理逻辑 ====================

def _shift_linear(x: torch.Tensor, shift_deg: float):
    """频谱线性插值平移"""
    B, C, L = x.shape
    t  = torch.linspace(TTH_MIN, TTH_MAX, L, device=x.device).view(1,1,L)
    t2 = torch.clamp(t + shift_deg, TTH_MIN, TTH_MAX)
    idx = (t2 - TTH_MIN) / (TTH_MAX - TTH_MIN) * (L-1)
    i0  = idx.floor().long().clamp(0, L-1)
    i1  = (i0 + 1).clamp(0, L-1)
    w   = (idx - i0.float())
    return (1-w)*x.gather(2, i0) + w*x.gather(2, i1)

def _reciprocal_Gstar(a,b,c, alpha,beta,gamma):
    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    ca, cb, cg = math.cos(ar), math.cos(br), math.cos(gr)
    G = np.array([
        [a*a, a*b*cg, a*c*cb], [a*b*cg, b*b, b*c*ca], [a*c*cb, b*c*ca, c*c]
    ], dtype=np.float64)
    return np.linalg.inv(G)

def enumerate_tth_dynamic(cell6: np.ndarray, wavelength: float):
    """动态生成 HKL 衍射峰用于结果展示"""
    a,b,c, alpha,beta,gamma = [float(x) for x in cell6.tolist()]
    Gstar = _reciprocal_Gstar(a,b,c, alpha,beta,gamma)
    
    dmin = wavelength / (2.0 * math.sin(math.radians(TTH_MAX / 2.0)))
    T = 1.0 / (dmin**2)
    lam_min = max(float(np.min(np.linalg.eigvalsh(Gstar))), 1e-12)
    H = int(math.ceil(math.sqrt(T / lam_min))) + 1

    peaks = []
    for h in range(-H, H+1):
        for k in range(-H, H+1):
            for l in range(-H, H+1):
                if h==0 and k==0 and l==0: continue
                vec = np.array([h,k,l], dtype=np.float64)
                inv_d2 = float(vec @ Gstar @ vec)
                if inv_d2 <= 0 or inv_d2 > T: continue
                d = 1.0 / math.sqrt(inv_d2)
                s = wavelength / (2.0*d)
                if abs(s) > 1.0: continue
                tth = 2.0 * math.degrees(math.asin(s))
                if TTH_MIN <= tth <= TTH_MAX:
                    peaks.append((tth, d, h,k,l))
    
    peaks.sort(key=lambda x: x[0])
    merged = []
    for p in peaks:
        if not merged or abs(p[0] - merged[-1][0]) >= 1e-3:
            merged.append(p)
    return merged

def snap_angles_inplace(cell6: np.ndarray, gate_logits_avg: torch.Tensor):
    """根据分类头预测结果进行角度吸附 (90/120度)"""
    if gate_logits_avg is None: return
    gate_prob = torch.softmax(gate_logits_avg, dim=-1).cpu().numpy()
    
    for j in range(3): # alpha, beta, gamma
        ang = float(cell6[3+j])
        p90, p120 = float(gate_prob[j,0]), float(gate_prob[j,1])
        
        if (p90 >= SNAP_PROB_TH) or (abs(ang - 90.0) <= SNAP_DIST_90):
            cell6[3+j] = 90.0
        elif (p120 >= SNAP_PROB_TH) or (abs(ang - 120.0) <= SNAP_DIST_120):
            cell6[3+j] = 120.0

def load_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model = CrystalNeXtT(num_hkl_classes=0).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def write_json_output(out_dir: str, sid: str, cell: np.ndarray, zero: float, disp: float):
    peaks = enumerate_tth_dynamic(cell, WAVELENGTH)
    peaks_fmt = [[round(p[0],5), round(p[1],6), 1.0, [int(p[2]), int(p[3]), int(p[4])]] for p in peaks]

    obj = {
        "crystal_info": {
            "a": float(cell[0]), "b": float(cell[1]), "c": float(cell[2]),
            "alpha": float(cell[3]), "beta": float(cell[4]), "gamma": float(cell[5]),
        },
        "zero_shift": float(zero),
        "sample_displacement": float(disp),
        "diffraction_info": { "peaks": peaks_fmt }
    }
    with open(os.path.join(out_dir, f"{sid}.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def inference_loop(args):
    # 1. 准备目录
    os.makedirs(args.output, exist_ok=True)

    # 2. 加载数据
    print(f"[加载] 读取数据: {args.input}")
    npz = np.load(args.input, allow_pickle=True)
    X, ids = npz["X"], npz["ids"]
    N = X.shape[0]

    # 3. 加载模型
    print(f"[加载] 模型权重: {args.model_path}")
    model = load_model(args.model_path, DEVICE)

    # 4. 推理
    print(f"[推理] 开始处理 {N} 个样本 (TTA={len(TTA_SHIFTS)}x, Device={DEVICE})...")
    
    for i in tqdm(range(N), unit="img"):
        # TTA 循环
        x_in = torch.from_numpy(X[i].astype(np.float32)).view(1,1,-1).to(DEVICE)
        
        pc_sum, z_sum, d_sum, gate_sum = None, 0.0, 0.0, None
        
        with torch.no_grad():
            for s in TTA_SHIFTS:
                x_aug = _shift_linear(x_in, s)
                pc, _, ang_logits, _, z, d = model(x_aug)
                
                pc_sum   = pc if pc_sum is None else (pc_sum + pc)
                z_sum   += z
                d_sum   += d
                gate_sum = ang_logits if gate_sum is None else (gate_sum + ang_logits)

        # 取平均
        k = len(TTA_SHIFTS)
        pred_cell = (pc_sum / k)[0].cpu().numpy()
        z_avg     = (z_sum / k).item()
        d_avg     = (d_sum / k).item()
        gate_avg  = (gate_sum / k)[0]

        # 角度后处理
        if SNAP_ANGLES:
            snap_angles_inplace(pred_cell, gate_avg)

        # 保存结果
        write_json_output(args.output, str(ids[i]), pred_cell, z_avg, d_avg)

    # 5. 打包
    if args.zip:
        zip_name = os.path.join(os.path.dirname(args.output), "submission")
        print(f"[打包] 正在压缩结果至 {zip_name}.zip ...")
        shutil.make_archive(zip_name, 'zip', root_dir=args.output, base_dir=".")
        print(f"[完成] 提交文件已生成: {zip_name}.zip")
    else:
        print(f"[完成] 所有结果已保存至: {args.output}")

def main():
    parser = argparse.ArgumentParser(description="CrystalNeXtT 预测推理脚本")
    
    parser.add_argument('--input', type=str, required=True, help='预处理后的测试集 .npz 路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重 .pt 路径')
    parser.add_argument('--output', type=str, default='output_json', help='结果保存目录')
    parser.add_argument('--zip', action='store_true', help='完成后自动打包成 .zip 用于提交')
    
    args = parser.parse_args()
    
    inference_loop(args)

if __name__ == "__main__":
    main()
