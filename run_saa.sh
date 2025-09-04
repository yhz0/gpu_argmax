# Run instance for ssn, cep, storm, 20, baa99-20
THREAD=384
python run.py build_saa --core-file smps_data/ssn/ssn.mps --time-file smps_data/ssn/ssn.tim --sto-file smps_data/ssn/ssn.sto --num-scenarios 1000 --num-threads $THREAD
