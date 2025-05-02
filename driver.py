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

from argmax_operation import ArgmaxOperation

op = ArgmaxOperation.from_smps_reader(reader, 1000, 1000, device='cpu')

