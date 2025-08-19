# Run instance for ssn, cep, storm, 20, baa99-20
$THREAD = 32
$SCENARIOS = 1000
python run.py build_saa --core-file smps_data/ssn/ssn.mps --time-file smps_data/ssn/ssn.tim --sto-file smps_data/ssn/ssn.sto --num-scenarios $SCENARIOS --num-threads $THREAD
python run.py build_saa --core-file smps_data/cep/cep.mps --time-file smps_data/cep/cep.tim --sto-file smps_data/cep/cep.sto --num-scenarios $SCENARIOS --num-threads $THREAD
python run.py build_saa --core-file smps_data/storm/storm.mps --time-file smps_data/storm/storm.tim --sto-file smps_data/storm/storm.sto --num-scenarios $SCENARIOS --num-threads $THREAD
python run.py build_saa --core-file smps_data/20/20.mps --time-file smps_data/20/20.tim --sto-file smps_data/20/20.sto --num-scenarios $SCENARIOS --num-threads $THREAD
python run.py build_saa --core-file smps_data/baa99-20/baa99-20.mps --time-file smps_data/baa99-20/baa99-20.tim --sto-file smps_data/baa99-20/baa99-20.sto --num-scenarios $SCENARIOS --num-threads $THREAD
