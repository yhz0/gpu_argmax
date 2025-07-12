import argparse
import json
import logging
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="A unified command-line interface for the project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Benchmark subcommand
    parser_benchmark = subparsers.add_parser("benchmark", help="Run the benchmark script.")
    parser_benchmark.add_argument("--n", type=int, required=True, help="Number of scenarios.")
    parser_benchmark.add_argument("--m", type=int, required=True, help="Number of dual solutions.")
    parser_benchmark.add_argument("--device", type=str, required=True, choices=['cpu', 'cuda'], help="Device to run on.")

    # Build SAA subcommand
    parser_build_saa = subparsers.add_parser("build_saa", help="Run the SAA builder script.")
    parser_build_saa.add_argument("--core-file", type=str, required=True, help="Path to the .cor or .mps file.")
    parser_build_saa.add_argument("--time-file", type=str, required=True, help="Path to the .tim file.")
    parser_build_saa.add_argument("--sto-file", type=str, required=True, help="Path to the .sto file.")
    parser_build_saa.add_argument("--num-scenarios", type=int, required=True, help="Number of scenarios for the SAA problem.")
    parser_build_saa.add_argument("--random-seed", type=int, help="Random seed for scenario sampling.")
    parser_build_saa.add_argument("--num-threads", type=int, help="Number of threads for Gurobi.")

    # Benders subcommand
    parser_benders = subparsers.add_parser("benders", help="Run the Benders solver script.")
    benders_group = parser_benders.add_mutually_exclusive_group()
    benders_group.add_argument("--config", type=str, help="Path to a JSON configuration file. Defaults to benders_config.json")
    benders_group.add_argument("--manual-config", nargs=4, metavar=('CORE_FILE', 'TIME_FILE', 'STO_FILE', 'H5_BASIS_FILE'), help="Manual configuration.")

    # Incumbent Benders subcommand
    parser_incumbent_benders = subparsers.add_parser("incumbent_benders", help="Run the Benders solver with incumbent strategy.")
    incumbent_benders_group = parser_incumbent_benders.add_mutually_exclusive_group()
    incumbent_benders_group.add_argument("--config", type=str, help="Path to a JSON configuration file. Defaults to benders_config.json")
    incumbent_benders_group.add_argument("--manual-config", nargs=4, metavar=('CORE_FILE', 'TIME_FILE', 'STO_FILE', 'H5_BASIS_FILE'), help="Manual configuration.")

    args = parser.parse_args()

    if args.command == "benchmark":
        from scripts.benchmark_argmax import run_benchmark
        run_benchmark(args.n, args.m, args.device)
    elif args.command == "build_saa":
        from scripts.build_saa import SAABuilder
        builder = SAABuilder(
            core_filepath=args.core_file,
            time_filepath=args.time_file,
            sto_filepath=args.sto_file,
            num_scenarios=args.num_scenarios,
            random_seed=args.random_seed,
            num_threads=args.num_threads
        )
        builder.run_pipeline(solve_model=True, save_hdf5=True, hdf5_filepath=f"{os.path.basename(args.core_file).split('.')[0]}_{args.num_scenarios}scen_results.h5")
    elif args.command == "benders":
        from scripts.run_benders_solver import BendersSolver
        config = None
        if args.config:
            config_path = args.config
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif args.manual_config:
            config = {
                'smps_core_file': args.manual_config[0],
                'smps_time_file': args.manual_config[1],
                'smps_sto_file': args.manual_config[2],
                'input_h5_basis_file': args.manual_config[3],
                'MAX_PI': 200000,
                'MAX_OMEGA': 100000,
                'SCENARIO_BATCH_SIZE': 1000,
                'NUM_SAMPLES_FOR_POOL': 100000,
                'ETA_LOWER_BOUND': 0.0,
                'initial_rho': 1.0,
                'rho_increase_factor': 2.0,
                'rho_decrease_factor': 0.5,
                'min_rho': 1e-6,
                'gamma': 0.5,
                'tolerance': 1e-4,
                'max_iterations': 20,
                'num_duals_to_add_per_iteration': 10000,
                'argmax_tol_cutoff': 1e-3,
                'num_workers': 4,
                'instance_name': "benders_run"
            }
        else:
            # Default to benders_config.json
            config_path = "benders_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        solver = BendersSolver(config=config)
        solver.run()
    elif args.command == "incumbent_benders":
        from scripts.run_incumbent_benders_solver import BendersSolver
        config = None
        if args.config:
            config_path = args.config
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif args.manual_config:
            config = {
                'smps_core_file': args.manual_config[0],
                'smps_time_file': args.manual_config[1],
                'smps_sto_file': args.manual_config[2],
                'input_h5_basis_file': args.manual_config[3],
                'MAX_PI': 200000,
                'MAX_OMEGA': 100000,
                'SCENARIO_BATCH_SIZE': 1000,
                'NUM_SAMPLES_FOR_POOL': 100000,
                'ETA_LOWER_BOUND': 0.0,
                'initial_rho': 1.0,
                'rho_increase_factor': 2.0,
                'rho_decrease_factor': 0.5,
                'min_rho': 1e-6,
                'gamma': 0.5,
                'tolerance': 1e-4,
                'max_iterations': 20,
                'num_duals_to_add_per_iteration': 10000,
                'argmax_tol_cutoff': 1e-3,
                'num_workers': 4,
                'instance_name': "incumbent_benders_run"
            }
        else:
            # Default to benders_config.json
            config_path = "benders_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        solver = BendersSolver(config=config)
        solver.run()

if __name__ == "__main__":
    main()
