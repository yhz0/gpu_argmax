import numpy as np
import h5py
import os
from smps_reader import SMPSReader
PROB_NAME = "cep"
FILE_DIR = os.path.join("smps_data", PROB_NAME)

CORE_FILENAME = f"{PROB_NAME}.mps"
TIME_FILENAME = f"{PROB_NAME}.tim"
STO_FILENAME = f"{PROB_NAME}.sto"

core_filepath = os.path.join(FILE_DIR, CORE_FILENAME)
time_filepath = os.path.join(FILE_DIR, TIME_FILENAME)
sto_filepath = os.path.join(FILE_DIR, STO_FILENAME)

reader = SMPSReader(core_filepath, time_filepath, sto_filepath)
reader.load_and_extract()

# obtain from reader  r_bar, C
r_bar = reader.r_bar
C = reader.C

# load C:\Users\yhz0\worksync\gpu_argmax\cep_100scen_results.h5
with h5py.File("cep_100scen_results.h5", "r") as f:
    num_scenarios = f['/metadata'].attrs['num_scenarios']
    pi_s = f['/solution/dual/pi_s'][:]
    stochastic_rhs_parts = f['/scenarios/stochastic_rhs_parts'][:]
    cbasis_y_all = f['/basis/cbasis_y_all'][:]
    vbasis_y_all = f['/basis/vbasis_y_all'][:]
    x_sol = f['solution/primal/x'][:]

# form r[s] = r_bar for s = 1...num_scenarios
# for each r[s][i] = stochastic_rhs_parts[s][i] for s = 1...num_scenarios  and i = 1...num_stochastic_elements
r = np.tile(r_bar, (num_scenarios, 1))
r[:, reader.stochastic_rows_relative_indices] = stochastic_rhs_parts

# for s = 1... num_scenarios, calculate pi[s] dot (r[s] - Cx)
Cx_product = C.dot(x_sol)
difference_term = r - Cx_product
correct_scenario_objective = np.sum(pi_s * difference_term, axis=1).mean()

del r, Cx_product, difference_term

# calculate the short delta r
short_delta_r = reader.get_short_delta_r(stochastic_rhs_parts)

from argmax_operation import ArgmaxOperation
op = ArgmaxOperation.from_smps_reader(reader, 10000, 10000, device='cpu')

for s in range(num_scenarios):
    op.add_pi(pi_s[s, :], np.array([]), vbasis_y_all[s, :], cbasis_y_all[s, :])

op.add_scenarios(short_delta_r.transpose())

# display op.num_pi, op.num_scenarios
print("op.num_pi", op.num_pi)
print("op.num_scenarios", op.num_scenarios)

alpha, beta, best_k_index = op.calculate_cut(x_sol)
estimated = alpha + beta @ x_sol

print("estimated", estimated)
print("correct_scenario_objective", correct_scenario_objective)
