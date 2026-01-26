import argparse
import glob
import json
import numpy as np
import multiprocessing as mp
from pathlib import Path
from joblib import Parallel, delayed
from scipy import interpolate, signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

THETA_MIN, THETA_MAX = 5.0, 80.0
FIXED_LEN = 10_000
ASLS_LAMBDA = 1e6
ASLS_P = 0.01
SG_WIN, SG_DEG = 17, 3
N_JOBS = max(mp.cpu_count() - 2, 1)

def baseline_asls_sparse(y, lam=ASLS_LAMBDA, p=ASLS_P, niter=10):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), format='csr')
    D_T_D = D.transpose().dot(D)
    w = np.ones(L)
    W = diags(w, 0, shape=(L, L), format='csr')
    
    z = np.zeros_like(y)
    for _ in range(niter):
        Z = W + lam * D_T_D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
        W.setdiag(w)
    return z

def process_signal(theta, intensity):
    mask = (theta >= THETA_MIN) & (theta <= THETA_MAX)
    t, i = theta[mask], intensity[mask]

    if len(t) < SG_WIN: return None

    baseline = baseline_asls_sparse(i)
    i = np.maximum(i - baseline, 0)
    
    i_smooth = signal.savgol_filter(i, SG_WIN, SG_DEG)
    denom = i_smooth.ptp()
    i_norm = (i_smooth - i_smooth.min()) / (denom if denom > 1e-12 else 1.0)

    f = interpolate.interp1d(t, i_norm, kind='linear', bounds_error=False, fill_value=0.0)
    grid = np.linspace(THETA_MIN, THETA_MAX, FIXED_LEN, dtype=np.float32)
    return f(grid).astype(np.float32)

def worker(path_str, data_dir, is_test):
    try:
        path = Path(path_str)
        raw = np.loadtxt(path, dtype=np.float32)
        x_feat = process_signal(raw[:, 0], raw[:, 1])
        
        if x_feat is None: return None

        y_cell, y_hkl = None, None
        if not is_test:
            with open(data_dir / f"{path.stem}.json", 'r', encoding='utf-8') as f:
                d = json.load(f)
            c = d["crystal_info"]
            y_cell = np.array([c["a"], c["b"], c["c"], c["alpha"], c["beta"], c["gamma"]], dtype=np.float32)
            y_hkl = d["diffraction_info"]["peaks"]

        return (x_feat, y_cell, y_hkl, path.stem)
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    args = parser.parse_args()

    in_dir = Path(args.input)
    files = glob.glob(str(in_dir / "*.xy")) + glob.glob(str(in_dir / "*.XY"))
    
    print(f"处理中: {len(files)} 样本 | 模式: {args.mode}")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(worker)(f, in_dir, args.mode == 'test') for f in tqdm(files, unit="task", ascii=True)
    )

    # 过滤失败样本
    valid_res = [r for r in results if r is not None]
    
    X = np.stack([r[0] for r in valid_res], dtype=np.float32)
    ids = np.array([r[3] for r in valid_res], dtype=object)
    
    save_dict = {"X": X, "ids": ids}
    if args.mode == 'train':
        save_dict["y_cell"] = np.stack([r[1] for r in valid_res], dtype=np.float32)
        save_dict["y_hkl"] = np.array([r[2] for r in valid_res], dtype=object)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **save_dict)
    print(f"完成. 有效样本: {len(valid_res)}/{len(files)} -> {args.output}")

if __name__ == "__main__":
    main()