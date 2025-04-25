from smps_reader import SMPSReader
from argmax_operation import ArgmaxOperation
import os

file_dir = os.path.join("smps_data", "ssn")

core_filename = "ssn.mps"
time_filename = "ssn.tim"
sto_filename = "ssn.sto"
core_filepath = os.path.join(file_dir, core_filename)
time_filepath = os.path.join(file_dir, time_filename)
sto_filepath = os.path.join(file_dir, sto_filename)

reader = SMPSReader(core_filepath, time_filepath, sto_filepath)
reader.load_and_extract()

MAX_PI = 10000
MAX_OMEGA = 10000
scenario_batch_size = 1000
op = ArgmaxOperation.from_smps_reader(reader, MAX_PI, MAX_OMEGA, scenario_batch_size)

print(op.r_sparse_indices_cpu)