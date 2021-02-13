from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

dims = [2, 3]
sizes = [500, 1000, 2000, 5000]
threads = [1, 2, 3, 4, 5, 6, 7, 8]
cuda_flg = ["naive", "sm"]
cuda_thds = [32, 64, 128, 256, 512, 1024]
fnames = [f"{dim}d_{size}.csv" for dim in dims for size in sizes]
threads_dirs = [f"{t}_thread" for t in threads]

def build_logbook(path: str) -> dict:
    logbook = {}
    for d in dims:
        try:
            logbook[d]
        except:
            logbook[d] = {}
        for s in sizes:
            fname = os.path.join(path, f"{d}d_{s}.csv")
            df = pd.read_csv(fname, header=None, names=['t'])
            logbook[d][s] = (df['t'].mean(), df['t'].var())

    return logbook

def build_logbooks_cuda(path: str) -> (dict, dict):
    naive_lb = {}
    sm_lb = {}
    for t in cuda_thds:
        naive_pth = os.path.join(path, str(t), "naive")
        sm_path = os.path.join(path, str(t), "sm")
        naive_lb[t] = build_logbook(naive_pth)
        sm_lb[t] = build_logbook(sm_path)
    return (naive_lb, sm_lb)

def build_static_logbook(path: str) -> dict:
    logbook = {}
    for t in threads:
        subir = os.path.join(path, f"{t}_thread")
        logbook[t] = build_logbook(subir)
    return logbook

def build_speedup(lb_a: dict, lb_b: dict) -> dict:
    assert(lb_a.keys() == lb_b.keys())
    assert(list(lb_a.keys()) == dims)
    speedup = {}
    for d in dims:
        slb_a = lb_a[d]
        slb_b = lb_b[d]
        assert(slb_a.keys() == slb_b.keys())
        assert(list(slb_a.keys()) == sizes)
        speedup[d] = {}
        for s in sizes:
            avg_time_a, var_a = slb_a[s]
            avg_time_b, var_b = slb_b[s]
            spdup = avg_time_a / avg_time_b
            variance = (spdup ** 2) * ((var_a) / (avg_time_a**2) + (var_b) / (avg_time_b**2))
            speedup[d][s] = (spdup, variance)
    return speedup
    
def build_speedup_static(seq_lb: dict, par_lb: dict, cuda: bool = False) -> dict:
    assert(list(seq_lb.keys()) == dims)
    if cuda:
        assert(list(par_lb.keys()) == cuda_thds)
        thds = cuda_thds
    else:
        assert(list(par_lb.keys()) == threads)
        thds = threads
    speedups = {}
    for t in thds:
        spar_lb = par_lb[t]
        assert(list(spar_lb.keys()) == dims)
        speedups[t] = {}
        for d in dims:
            sspar_lb = spar_lb[d]
            sseq_lb = seq_lb[d]
            assert(list(sspar_lb.keys()) == list(sseq_lb) == sizes)
            speedups[t][d] = {}
            for s in sizes:
                seq_time, seq_var = sseq_lb[s]
                par_time, par_var = sspar_lb[s]
                spdup = seq_time / par_time
                variance = (spdup ** 2) * ((seq_var) / (seq_time**2) + (par_var)/ (par_time**2))
                speedups[t][d][s] = (spdup, variance)     
    return speedups

def plot_logbook(lb: dict, title: str = "", save: bool = False) -> None:
    assert(list(lb.keys()) == dims)
    plt.clf()
    plt.figure()
    for d in dims:
        slb = lb[d]
        x = list(slb.keys())
        assert(x == sizes)
        y = [s for (s,v) in slb.values()]
        err = [np.sqrt(v) for (s,v) in slb.values()]
        plt.errorbar(x, y, yerr=err, marker='.', label=f"{d}D")
    plt.title(title)
    plt.grid(linestyle='dotted', alpha=0.5)
    plt.xlabel("Size")
    plt.ylabel("Time (ms)")
    plt.legend()
    if save:
        plt.savefig(f"{title}.png")
    else:
        plt.show()
    return

def plot_two_logbooks(lb_a: dict, lb_b: dict, title: str = "", 
                      label_a : str = "", label_b : str = "", save: bool = False) -> None:
    assert(list(lb_a.keys()) == list(lb_b.keys()) == dims)
    plt.clf()
    plt.figure()
    for d in dims:
        slb_a = lb_a[d]
        slb_b = lb_b[d]
        x = list(slb_a.keys())
        assert(x == list(slb_b.keys()) == sizes)
        y_a = [s for (s,v) in slb_a.values()]
        y_b = [s for (s,v) in slb_b.values()]
        err_a = [np.sqrt(v) for (s,v) in slb_a.values()]
        err_b = [np.sqrt(v) for (s,v) in slb_b.values()]
        plt.errorbar(x, y_a, yerr=err_a, marker='.', label=f"{label_a} - {d}D")
        plt.errorbar(x, y_b, yerr=err_b, marker='s', label=f"{label_b} - {d}D", linestyle='dashed')
    plt.title(title)
    plt.grid(linestyle='dotted', alpha=0.5)
    plt.xlabel("Size")
    plt.ylabel("Time (ms)")
    plt.legend()
    if save:
        plt.savefig(f"{title}.png")
    else:
        plt.show()
    return

def plot_speedup(su_lb: dict, title: str, filt: int = 2, save: bool = False, cuda: bool = False) -> None:
    if cuda:
        assert(list(su_lb.keys()) == cuda_thds)
        thds = cuda_thds
    else:
        assert(list(su_lb.keys()) == threads)
        thds = threads
    plot_lb = {s : {} for s in sizes}
    plt.clf()
    plt.figure()
    for t in thds:
        ssu_lb = su_lb[t]
        assert(list(ssu_lb.keys()) == dims)
        assert(filt in dims)
        fltrd = ssu_lb[filt]
        assert(list(fltrd.keys()) == sizes)
        for s in sizes:
            plot_lb[s][t] = fltrd[s]
    for s in sizes:
        y = [plot_lb[s][t][0] for t in thds]
        err = [np.sqrt(plot_lb[s][t][1]) for t in thds]
        plt.errorbar(x=thds, y=y, yerr=err, label=str(s), marker='.')
    plt.title(title)
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.grid(linestyle='dotted', alpha=0.5)
    plt.legend()
    if save:
        plt.savefig(f"{title}.png")
    else:
        plt.show()
    return
    
if __name__ == "__main__":

    seq_lb = build_logbook("sequential")
    dyn_lb = build_logbook("openmp/dynamic")
    sta_lb = build_static_logbook("openmp/static")
    sng_thd_lb = sta_lb[1]
    
    save = True

    plot_logbook(seq_lb, title="Sequential", save=save)
    plot_logbook(dyn_lb, title="Dynamic Scheduling", save=save)

    against_seq = build_speedup_static(seq_lb, sta_lb)
    filt = 2
    plot_speedup(against_seq, title=f"{filt}D - Against sequential", filt=filt, save=save)
    filt = 3
    plot_speedup(against_seq, title=f"{filt}D - Against sequential", filt=filt, save=save)

    against_dyn = build_speedup_static(dyn_lb, sta_lb)
    filt = 2
    plot_speedup(against_dyn, title=f"{filt}D - Against Dynamic", filt=filt, save=save)
    filt = 3
    plot_speedup(against_dyn, title=f"{filt}D - Against Dynamic", filt=filt, save=save)

    against_st = build_speedup_static(sng_thd_lb, sta_lb)
    filt = 2
    plot_speedup(against_st, title=f"{filt}D - Against T=1", filt=filt, save=save)
    filt = 3
    plot_speedup(against_st, title=f"{filt}D - Against T=1", filt=filt, save=save)


    naive_lb, sm_lb = build_logbooks_cuda("cuda")

    plot_logbook(naive_lb[64], title="Naive CUDA (64 threads)", save=save)
    plot_logbook(sm_lb[64], title="Tiling CUDA (64 threads)", save=save)

    naive_against_seq = build_speedup_static(seq_lb, naive_lb, cuda=True)
    filt = 2
    plot_speedup(naive_against_seq, title=f"{filt}D - Naive CUDA vs Sequential", filt=filt, save=save, cuda=True)
    filt = 3
    plot_speedup(naive_against_seq, title=f"{filt}D - Naive CUDA vs Sequential", filt=filt, save=save, cuda=True)

    sm_against_seq = build_speedup_static(seq_lb, sm_lb, cuda=True)
    filt = 2
    plot_speedup(sm_against_seq, title=f"{filt}D - Tiling CUDA vs Sequential", filt=filt, save=save, cuda=True)
    filt = 3
    plot_speedup(sm_against_seq, title=f"{filt}D - Tiling CUDA vs Sequential", filt=filt, save=save, cuda=True)

    naive_against_dyn = build_speedup_static(dyn_lb, naive_lb, cuda=True)
    filt = 2
    plot_speedup(naive_against_dyn, title=f"{filt}D - Naive CUDA vs Dynamic", filt=filt, save=save, cuda=True)
    filt = 3
    plot_speedup(naive_against_dyn, title=f"{filt}D - Naive CUDA vs Dynamic", filt=filt, save=save, cuda=True)

    sm_against_dyn = build_speedup_static(dyn_lb, sm_lb, cuda=True)
    filt = 2
    plot_speedup(sm_against_dyn, title=f"{filt}D - Tiling CUDA vs Dynamic", filt=filt, save=save, cuda=True)
    filt = 3
    plot_speedup(sm_against_dyn, title=f"{filt}D - Tiling CUDA vs Dynamic", filt=filt, save=save, cuda=True)

    plot_two_logbooks(seq_lb, dyn_lb, title="Sequential vs Dynamic Scheduling", label_a="Sequential", label_b="Dynamic", save=save)
    plot_two_logbooks(naive_lb[64], sm_lb[64], title="CUDA: Naive vs Tiling (64 threads)", label_a="Naive", label_b="Tiling", save=save)