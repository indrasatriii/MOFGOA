import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV as HV_pymoo
from FGOA import FGOA  # Pastikan kamu punya modul FGOA.py

# ======= METRIK EVALUASI =======
def compute_igd(archive, pareto_front):
    if len(archive) == 0:
        return np.inf
    solutions = np.array([entry['fit'] for entry in archive])
    combined = np.vstack([pareto_front, solutions])
    scaled = MinMaxScaler().fit_transform(combined)
    pf_scaled = scaled[:len(pareto_front)]
    sol_scaled = scaled[len(pareto_front):]
    return IGD(pf_scaled).do(sol_scaled)

def compute_hv(archive, ref_point):
    if len(archive) == 0:
        return 0.0
    solutions = np.array([entry['fit'] for entry in archive])

    # Normalisasi
    scaler = MinMaxScaler()
    scaled_solutions = scaler.fit_transform(solutions)

    # Normalisasi ref_point juga, berdasarkan scaler
    ref_point_scaled = scaler.transform([ref_point])[0]

    return HV_pymoo(ref_point=ref_point_scaled).do(scaled_solutions)


def compute_statistics(values):
    return np.min(values), np.max(values), np.mean(values), np.median(values), np.std(values)

# ======= JALANKAN FGOA =======
def run_fgoa(problem, pareto_front, ref_point, dim, size, minx, maxx,
             max_evals, incentive_threshold, archive_size, ngrid):
    fgoa = FGOA(
        problem,
        dim=dim,
        size=size,
        minx=minx,
        maxx=maxx,
        max_evals=max_evals,
        incentive_threshold=incentive_threshold,
        archive_size=archive_size,
        ngrid=ngrid,
        max_iter=max_evals // size
    )
    archive_items = fgoa.optimize(pareto_front)
    positions = np.array([item['pos'] for item in archive_items])
    objectives = np.array([item['fit'] for item in archive_items])

    #update ref_point1 for HV
    ref_point1 = np.max(objectives, axis=0) + 0.05
    
    igd = compute_igd(archive_items, pareto_front)
    hv = compute_hv(archive_items, ref_point)
    
    return igd, hv, archive_items, ref_point


# ======= JALANKAN NSGA3 =======
def run_nsga3(problem, ref_point, pop_size, max_evals, seed):
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=15)
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    res = minimize(problem,
                   algorithm,
                   seed=seed,
                   termination=('n_eval', max_evals),
                   verbose=False)
    igd = IGD(problem.pareto_front())(res.F)
    hv = HV_pymoo(ref_point=ref_point)(res.F)
    return igd, hv, res.F

# ======= JALANKAN NSGA-II =======
def run_nsga2(problem, ref_point, pop_size, max_evals, seed):
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem,
                   algorithm,
                   seed=seed,
                   termination=('n_eval', max_evals),
                   verbose=False)
    igd = IGD(problem.pareto_front())(res.F)
    hv = HV_pymoo(ref_point=ref_point)(res.F)
    return igd, hv, res.F


def run_moead(problem, ref_point, pop_size, max_evals, seed):
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=15)
    algorithm = MOEAD(ref_dirs=ref_dirs)
    res = minimize(problem,
                   algorithm,
                   seed=seed,
                   termination=('n_eval', max_evals),
                   verbose=False)
    igd = IGD(problem.pareto_front())(res.F)
    hv = HV_pymoo(ref_point=ref_point)(res.F)
    return igd, hv, res.F


def run_spea2(problem, ref_point, pop_size, max_evals, seed):
    algorithm = SPEA2(pop_size=pop_size)
    res = minimize(problem,
                   algorithm,
                   seed=seed,
                   termination=('n_eval', max_evals),
                   verbose=False)
    igd = IGD(problem.pareto_front())(res.F)
    hv = HV_pymoo(ref_point=ref_point)(res.F)
    return igd, hv, res.F


def run_rnsga2(problem, ref_point, pop_size, max_evals, seed):
    # Gunakan np.array untuk ref_points
    algorithm = RNSGA2(ref_points=np.array([ref_point]))
    res = minimize(problem,
                   algorithm,
                   seed=seed,
                   termination=('n_eval', max_evals),
                   verbose=False)
    igd = IGD(problem.pareto_front())(res.F)
    hv = HV_pymoo(ref_point=ref_point)(res.F)
    return igd, hv, res.F


def run_agemoea(problem, ref_point, pop_size, max_evals, seed):
    algorithm = AGEMOEA()
    res = minimize(problem,
                   algorithm,
                   seed=seed,
                   termination=('n_eval', max_evals),
                   verbose=False)
    igd = IGD(problem.pareto_front())(res.F)
    hv = HV_pymoo(ref_point=ref_point)(res.F)
    return igd, hv, res.F

# ======= SETUP EKSPERIMEN =======
a= "dtlz7"
problem_name = a
problem = get_problem(problem_name)
pareto_front = problem.pareto_front(use_cache=False)
ref_point = ([1.5,1.5,1.5])
#ref_point1 = np.max(pareto_front, axis=0) + 0.05
#ref_point1 = ref_point
num_runs = 2
pop_size =300
archive_size =300
max_evals = 150000

fgoa_igd, fgoa_hv, fgoa_archives = [], [], []
nsga2_igd, nsga2_hv, nsga2_fronts = [], [], []
nsga3_igd, nsga3_hv, nsga3_fronts = [], [], []
moead_igd, moead_hv, moead_fronts = [], [], []
spea2_igd, spea2_hv, spea2_fronts = [], [], []
rnsga2_igd, rnsga2_hv, rnsga2_fronts = [], [], []
agemoea_igd, agemoea_hv, agemoea_fronts = [], [], []

# ======= EKSEKUSI MULTIPLE RUN =======
for run in range(num_runs):
    print(f"\n🌀🪿 Run {run+1}/{num_runs}")

    # FGOA
    igd_fgoa, hv_fgoa, archive,ref_point = run_fgoa(
        problem, pareto_front, ref_point,
        dim=problem.n_var,
        size=pop_size,
        minx=0.0,
        maxx=1.0,
        max_evals=max_evals,
        incentive_threshold=1.5,
        archive_size=archive_size,
        ngrid=200
    )
    fgoa_igd.append(igd_fgoa)
    fgoa_hv.append(hv_fgoa)
    fgoa_archives.append(archive)
    print(f"FGOA → IGD: {igd_fgoa:.6f}, HV: {hv_fgoa:.6f}")

    # NSGA2
    igd_nsga2, hv_nsga2, front2 = run_nsga2(problem, ref_point, pop_size, max_evals, seed=run)
    nsga2_igd.append(igd_nsga2)
    nsga2_hv.append(hv_nsga2)
    nsga2_fronts.append(front2)
    print(f"NSGA2 → IGD: {igd_nsga2:.6f}, HV: {hv_nsga2:.6f}")

    # NSGA3
    igd_nsga3, hv_nsga3, front3 = run_nsga3(problem, ref_point, pop_size, max_evals, seed=run)
    nsga3_igd.append(igd_nsga3)
    nsga3_hv.append(hv_nsga3)
    nsga3_fronts.append(front3)
    print(f"NSGA3 → IGD: {igd_nsga3:.6f}, HV: {hv_nsga3:.6f}")

    # MOEA/D
    igd_moead, hv_moead, front4 = run_moead(problem, ref_point, pop_size, max_evals, seed=run)
    moead_igd.append(igd_moead)
    moead_hv.append(hv_moead)
    moead_fronts.append(front4)
    print(f"MOEA/D → IGD: {igd_moead:.6f}, HV: {hv_moead:.6f}")

    # SPEA2
    igd_spea2, hv_spea2, front5 = run_spea2(problem, ref_point, pop_size, max_evals, seed=run)
    spea2_igd.append(igd_spea2)
    spea2_hv.append(hv_spea2)
    spea2_fronts.append(front5)
    print(f"SPEA2 → IGD: {igd_spea2:.6f}, HV: {hv_spea2:.6f}")

    # RNSGA2
    igd_rnsga2, hv_rnsga2, front6 = run_rnsga2(problem, ref_point, pop_size, max_evals, seed=run)
    rnsga2_igd.append(igd_rnsga2)
    rnsga2_hv.append(hv_rnsga2)
    rnsga2_fronts.append(front6)
    print(f"RNSGA2 → IGD: {igd_rnsga2:.6f}, HV: {hv_rnsga2:.6f}")

    # AGE-MOEA
    igd_agemoea, hv_agemoea, front7 = run_agemoea(problem, ref_point, pop_size, max_evals, seed=run)
    agemoea_igd.append(igd_agemoea)
    agemoea_hv.append(hv_agemoea)
    agemoea_fronts.append(front7)
    print(f"AGE-MOEA → IGD: {igd_agemoea:.6f}, HV: {hv_agemoea:.6f}")

all_results = [
    ("FGOA", np.mean(fgoa_igd), np.std(fgoa_igd), np.mean(fgoa_hv), np.std(fgoa_hv)),
    ("NSGA2", np.mean(nsga2_igd), np.std(nsga2_igd), np.mean(nsga2_hv), np.std(nsga2_hv)),
    ("NSGA3", np.mean(nsga3_igd), np.std(nsga3_igd), np.mean(nsga3_hv), np.std(nsga3_hv)),
    ("MOEA/D", np.mean(moead_igd), np.std(moead_igd), np.mean(moead_hv), np.std(moead_hv)),
    ("SPEA2", np.mean(spea2_igd), np.std(spea2_igd), np.mean(spea2_hv), np.std(spea2_hv)),
    ("RNSGA2", np.mean(rnsga2_igd), np.std(rnsga2_igd), np.mean(rnsga2_hv), np.std(rnsga2_hv)),
    ("AGE-MOEA", np.mean(agemoea_igd), np.std(agemoea_igd), np.mean(agemoea_hv), np.std(agemoea_hv))
]


# Ranking by IGD (semakin kecil semakin bagus)
print("\n🏆 Ranking by IGD (lower is better):")
for name, igd_mean, igd_std, _, _ in sorted(all_results, key=lambda x: x[1]):
    print(f"{name}: IGD Mean = {igd_mean:.6f}, Std = {igd_std:.6f}")

# Ranking by HV (semakin besar semakin bagus)
print("\n🏆 Ranking by HV (higher is better):")
for name, _, _, hv_mean, hv_std in sorted(all_results, key=lambda x: -x[3]):
    print(f"{name}: HV Mean = {hv_mean:.6f}, Std = {hv_std:.6f}")

print(a)
from mpl_toolkits.mplot3d import Axes3D  # Jika belum diimpor sebelumnya

# Contoh data (pastikan semua front berupa array dengan 3 kolom)
fit_fgoa = np.array([entry['fit'] for entry in fgoa_archives[-1]])
fit_nsga3 = nsga3_fronts[-1]
fit_agemoea = agemoea_fronts[-1]
fit_nsga2 = nsga2_fronts[-1]
if len(moead_fronts) > 0:
    fit_MOEA = moead_fronts[-1]  # ambil front terakhir
else:
    print("⚠️ Tidak ada solusi MOEA/D tersedia.")
    fit_MOEA = np.empty((0, problem.n_obj))  # array kosong agar tidak error
fit_rnsga2 = rnsga2_fronts[-1]

# Fungsi untuk plotting 3D
def plot_3d_pf(true_pf, alg_fit, alg_name, title_suffix):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # True Pareto Front
    ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], 
               label="True PF", color='r', marker='x', s=50)

    # Algoritma
    ax.scatter(alg_fit[:, 0], alg_fit[:, 1], alg_fit[:, 2], 
               label=alg_name, color='b' if alg_name == "FGOA" else 'g' if alg_name == "NSGA3" else 'm' if alg_name == "AGE-MOEA" else 'c' if alg_name == "NSGA2" else 'orange' if alg_name == "MOEA/D" else 'purple', 
               marker='o', alpha=0.6)

    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_zlabel("Objective 3")
    ax.set_title(f"{title_suffix} on {a}")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Panggil fungsi untuk setiap algoritma
plot_3d_pf(pareto_front, fit_fgoa, "FGOA", "FGOA vs True PF")
plot_3d_pf(pareto_front, fit_nsga3, "NSGA3", "NSGA3 vs True PF")
plot_3d_pf(pareto_front, fit_agemoea, "AGE-MOEA", "AGE-MOEA vs True PF")
plot_3d_pf(pareto_front, fit_nsga2, "NSGA2", "NSGA2 vs True PF")
plot_3d_pf(pareto_front, fit_MOEA, "MOEA/D", "MOEA/D vs True PF")
plot_3d_pf(pareto_front, fit_rnsga2, "RNSGA2", "RNSGA2 vs True PF")

# =============================================================================
# def plot_2d_pf(true_pf, alg_fit, alg_name, title_suffix):
#     plt.figure(figsize=(8, 6))
# 
#     # True Pareto Front
#     plt.scatter(true_pf[:, 0], true_pf[:, 1], 
#                 label="True PF", color='r', marker='x', s=50)
# 
#     # Algoritma
#     color_map = {
#         "FGOA": 'b',
#         "NSGA3": 'g',
#         "AGE-MOEA": 'm',
#         "NSGA2": 'c',
#         "MOEA/D": 'orange',
#         "RNSGA2": 'purple'
#     }
#     plt.scatter(alg_fit[:, 0], alg_fit[:, 1], 
#                 label=alg_name, 
#                 color=color_map.get(alg_name, 'k'), 
#                 marker='o', alpha=0.6)
# 
#     plt.xlabel("Objective 1")
#     plt.ylabel("Objective 2")
#     plt.title(f"{title_suffix} on {a}")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
# 
# # Panggil fungsi plot 2D untuk setiap algoritma
# plot_2d_pf(pareto_front, fit_fgoa, "FGOA", "FGOA vs True PF")
# plot_2d_pf(pareto_front, fit_nsga3, "NSGA3", "NSGA3 vs True PF")
# plot_2d_pf(pareto_front, fit_agemoea, "AGE-MOEA", "AGE-MOEA vs True PF")
# plot_2d_pf(pareto_front, fit_nsga2, "NSGA2", "NSGA2 vs True PF")
# plot_2d_pf(pareto_front, fit_MOEA, "MOEA/D", "MOEA/D vs True PF")
# plot_2d_pf(pareto_front, fit_rnsga2, "RNSGA2", "RNSGA2 vs True PF")
# =============================================================================


from scipy.stats import wilcoxon
import itertools

# Data IGD dan HV per algoritma
igd_dict = {
    "FGOA": fgoa_igd,
    "NSGA2": nsga2_igd,
    "NSGA3": nsga3_igd,
    "MOEA/D": moead_igd,
    "SPEA2": spea2_igd,
    "RNSGA2": rnsga2_igd,
    "AGE-MOEA": agemoea_igd
}

hv_dict = {
    "FGOA": fgoa_hv,
    "NSGA2": nsga2_hv,
    "NSGA3": nsga3_hv,
    "MOEA/D": moead_hv,
    "SPEA2": spea2_hv,
    "RNSGA2": rnsga2_hv,
    "AGE-MOEA": agemoea_hv
}

def perform_wilcoxon_tests(metric_dict, metric_name):
    print(f"\n Wilcoxon Signed-Rank Test for {metric_name}:")
    algo_names = list(metric_dict.keys())
    for algo1, algo2 in itertools.combinations(algo_names, 2):
        data1 = metric_dict[algo1]
        data2 = metric_dict[algo2]
        try:
            stat, p = wilcoxon(data1, data2)
            result = "✅ Signifikan" if p < 0.05 else "not_Sig"
            print(f"{algo1} vs {algo2} → p = {p:.4f} → {result}")
        except ValueError as e:
            print(f"{algo1} vs {algo2} → Uji gagal: {e}")

# Jalankan uji untuk IGD dan HV
perform_wilcoxon_tests(igd_dict, "IGD")
perform_wilcoxon_tests(hv_dict, "HV")
