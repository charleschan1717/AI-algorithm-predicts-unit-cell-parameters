import os
import json
import math
import argparse
import numpy as np
from typing import List, Tuple, Optional

# --- 物理与评分常数 ---
DEFAULT_CFG = {
    "wavelength": 1.5406,     # Cu Kα1 (Å)
    "tol_deg": 0.10,          # 峰位匹配容差 (度)
    "tth_min": 5.0,
    "tth_max": 80.0,
    "hmax": 10,               # hkl 枚举上限
    "radius_mm": 240.0,       # 衍射仪半径
    "rmse_angle_unit": "rad", # RMSE计算时角度单位: 'rad' 或 'deg'
    "acc_denom": "gt"         # 准确率分母: 'gt'(默认), 'pred', 'max'
}

def load_json(fp: str) -> dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_cell(obj: dict) -> List[float]:
    """从字典中解析晶胞参数 [a, b, c, alpha, beta, gamma]"""
    keys = ["crystal_info", "cell", "cell_params", "lattice"]
    target = None
    
    # 查找顶层字段
    for k in keys:
        if k in obj:
            target = obj[k]
            break
            
    # 查找 metadata 字段
    if target is None and "metadata" in obj:
        for k in keys:
            if k in obj["metadata"]:
                target = obj["metadata"][k]
                break
                
    if target:
        if isinstance(target, dict):
            return [float(target[k]) for k in ["a","b","c","alpha","beta","gamma"]]
        return [float(x) for x in target]
        
    raise ValueError(f"Missing cell info in {obj.keys()}")

def extract_offsets(obj: dict) -> Tuple[float, float]:
    """提取 (zero_shift, sample_displacement)"""
    z_keys = ["zero_shift", "zeroShift", "zshift"]
    s_keys = ["sample_displacement", "sampleDisp", "sdisp"]
    
    def _find(d, keys):
        for k in keys:
            if k in d: return float(d[k])
        return 0.0

    z = _find(obj, z_keys)
    s = _find(obj, s_keys)
    
    # Check metadata if zero
    if z == 0.0 and s == 0.0 and "metadata" in obj:
        z = _find(obj.get("metadata", {}), z_keys)
        s = _find(obj.get("metadata", {}), s_keys)
        
    return z, s

def extract_gt_peaks(obj: dict) -> Optional[List[float]]:
    """提取真实峰位 (2θ)"""
    # 优先尝试 standard format
    if "diffraction_info" in obj and "peaks" in obj["diffraction_info"]:
        peaks = obj["diffraction_info"]["peaks"]
    else:
        # Fallback keys
        peaks = obj.get("peaks", obj.get("gt_peaks"))

    if not peaks: return None

    out = []
    for p in peaks:
        # Format: [2theta, I, ...] or {"two_theta": val, ...}
        val = None
        if isinstance(p, (list, tuple)) and len(p) > 0:
            val = p[0]
        elif isinstance(p, dict):
            val = p.get("two_theta", p.get("2theta"))
            
        if val is not None:
            out.append(float(val))
            
    return sorted(out)

# --- 物理计算核心 ---

def tth_from_cell(cell: List[float], cfg: dict) -> np.ndarray:
    """根据 Bragg 方程计算理论 2θ"""
    a, b, c, alpha, beta, gamma = cell
    ar, br, gr = np.radians([alpha, beta, gamma])
    ca, cb, cg = np.cos([ar, br, gr])
    
    # Metric Tensor
    G = np.array([
        [a*a,    a*b*cg, a*c*cb],
        [a*b*cg, b*b,    b*c*ca],
        [a*c*cb, b*c*ca, c*c]
    ])
    try:
        Gstar = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return np.array([])

    # Enumerate HKL
    hmax = cfg['hmax']
    r = range(-hmax, hmax + 1)
    h, k, l = np.meshgrid(r, r, r)
    h, k, l = h.flatten(), k.flatten(), l.flatten()
    # Filter (0,0,0)
    valid = (h!=0) | (k!=0) | (l!=0)
    H = np.vstack([h[valid], k[valid], l[valid]]).T
    
    # d-spacing calculation
    inv_d2 = np.sum((H @ Gstar) * H, axis=1)
    valid_d = inv_d2 > 0
    d = 1.0 / np.sqrt(inv_d2[valid_d])
    
    # 2Theta
    wl = cfg['wavelength']
    s = wl / (2.0 * d)
    valid_s = (s >= 0) & (s <= 1.0)
    tth = 2.0 * np.degrees(np.arcsin(s[valid_s]))
    
    # Range filter
    mask = (tth >= cfg['tth_min']) & (tth <= cfg['tth_max'])
    return np.sort(tth[mask])

def apply_offsets(tth: np.ndarray, z: float, s: float, r: float) -> np.ndarray:
    """应用零漂和样偏修正"""
    if len(tth) == 0: return tth
    # Zero shift
    tt = tth + z
    # Sample displacement: Δ2θ ≈ -(2s/R)cosθ
    theta_rad = np.radians(tt / 2.0)
    d2th = -np.degrees((2.0 * s / r) * np.cos(theta_rad))
    return tt + d2th

# --- 评分逻辑 ---

def compute_rmse(pred: List[float], gt: List[float], angle_unit: str) -> float:
    p = np.array(pred)
    g = np.array(gt)
    # 角度处理
    if angle_unit == 'rad':
        p[3:6] = np.radians(p[3:6])
        g[3:6] = np.radians(g[3:6])
    return float(np.sum((p - g) ** 2))

def compute_accuracy(pred_tth: np.ndarray, gt_tth: np.ndarray, tol: float, denom_mode: str) -> float:
    # 贪心匹配
    matched = 0
    i, j = 0, 0
    n_g, n_p = len(gt_tth), len(pred_tth)
    
    while i < n_g and j < n_p:
        diff = pred_tth[j] - gt_tth[i]
        if abs(diff) <= tol:
            matched += 1
            i += 1
            j += 1
        elif diff < 0:
            j += 1
        else:
            i += 1
            
    # 分母计算
    if denom_mode == 'pred': denom = n_p
    elif denom_mode == 'max': denom = max(n_g, n_p)
    elif denom_mode == 'mean': denom = (n_g + n_p) / 2.0
    else: denom = n_g  # default 'gt'
    
    return matched / max(1, denom)

def evaluate(ans_dir: str, gt_dir: str, args):
    files = [f for f in os.listdir(ans_dir) if f.endswith(".json")]
    files.sort()
    
    print(f"Found {len(files)} files in {ans_dir}")
    
    sse_list = []
    acc_list = []
    valid_count = 0
    
    cfg = vars(args) # convert args to dict for functions
    
    for fn in files:
        ans_fp = os.path.join(ans_dir, fn)
        gt_fp = os.path.join(gt_dir, fn)
        
        if not os.path.exists(gt_fp):
            # Try finding without extension match or different naming if needed
            # For strict mode, skip
            continue
            
        try:
            ans_data = load_json(ans_fp)
            gt_data = load_json(gt_fp)
            
            # 1. RMSE (Cell Parameters)
            c_pred = extract_cell(ans_data)
            c_gt = extract_cell(gt_data)
            sse = compute_rmse(c_pred, c_gt, args.rmse_angle_unit)
            
            # 2. Accuracy (Peak Matching)
            # Try getting explicit peaks first, else calculate from cell
            gt_peaks = extract_gt_peaks(gt_data)
            if gt_peaks is None or len(gt_peaks) == 0:
                gt_peaks = tth_from_cell(c_gt, cfg)
            
            z, s = extract_offsets(ans_data)
            pred_peaks = tth_from_cell(c_pred, cfg)
            pred_peaks = apply_offsets(pred_peaks, z, s, args.radius_mm)
            
            acc = compute_accuracy(pred_peaks, np.array(gt_peaks), args.tol_deg, args.acc_denom)
            
            sse_list.append(sse)
            acc_list.append(acc)
            valid_count += 1
            
        except Exception as e:
            print(f"[Warning] Failed to score {fn}: {e}")

    if valid_count == 0:
        print("[Error] No valid samples evaluated.")
        return

    # Aggregate
    rmse = math.sqrt(sum(sse_list) / valid_count)
    avg_acc = sum(acc_list) / valid_count
    final_score = 0.9 * avg_acc - 0.1 * rmse + 2.0
    
    print("-" * 60)
    print(f"Evaluation Report")
    print("-" * 60)
    print(f"Samples Evaluated : {valid_count}")
    print(f"RMSE (Angle={args.rmse_angle_unit}): {rmse:.6f}")
    print(f"Avg Accuracy      : {avg_acc:.6f}")
    print(f"Final Score       : {final_score:.6f}")
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Official Score Evaluation Script")
    
    # Paths
    parser.add_argument("--ans", type=str, required=True, help="Path to prediction directory (JSONs)")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth directory (JSONs)")
    
    # Physics & Matching Params
    parser.add_argument("--tol_deg", type=float, default=DEFAULT_CFG['tol_deg'], help="Matching tolerance (deg)")
    parser.add_argument("--wavelength", type=float, default=DEFAULT_CFG['wavelength'], help="Wavelength (Å)")
    parser.add_argument("--tth_min", type=float, default=DEFAULT_CFG['tth_min'])
    parser.add_argument("--tth_max", type=float, default=DEFAULT_CFG['tth_max'])
    parser.add_argument("--radius_mm", type=float, default=DEFAULT_CFG['radius_mm'])
    parser.add_argument("--hmax", type=int, default=DEFAULT_CFG['hmax'])
    
    # Metric Config
    parser.add_argument("--rmse_angle_unit", type=str, default=DEFAULT_CFG['rmse_angle_unit'], choices=['rad', 'deg'])
    parser.add_argument("--acc_denom", type=str, default=DEFAULT_CFG['acc_denom'], choices=['gt', 'pred', 'max', 'mean'])
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ans) or not os.path.exists(args.gt):
        print("Error: Input directories do not exist.")
        return
        
    evaluate(args.ans, args.gt, args)

if __name__ == "__main__":
    main()
