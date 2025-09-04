import argparse
import json
import logging
import os
import sys

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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


    # Incumbent Benders subcommand
    parser_incumbent_benders = subparsers.add_parser("incumbent_benders", help="Run the Benders solver with incumbent strategy.")
    parser_incumbent_benders.add_argument("--config", type=str, help="Path to a JSON configuration file. Defaults to benders_config.json")

    # SMPS reader subcommand
    parser_smps_reader = subparsers.add_parser("smps_reader", help="Run the SMPS reader script.")
    parser_smps_reader.add_argument("--core-file", type=str, required=True, help="Path to the .cor or .mps file.")
    parser_smps_reader.add_argument("--time-file", type=str, required=True, help="Path to the .tim file.")
    parser_smps_reader.add_argument("--sto-file", type=str, required=True, help="Path to the .sto file.")

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
    elif args.command == "incumbent_benders":
        from scripts.run_incumbent_benders_solver import BendersSolver
        if args.config:
            config_path = args.config
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
    elif args.command == "smps_reader":
        from src.smps_reader import SMPSReader
        reader = SMPSReader(
            core_file=args.core_file,
            time_file=args.time_file,
            sto_file=args.sto_file
        )
        reader.load_and_extract()
        reader.print_summary()


if __name__ == "__main__":
    # # DEBUG: Direct run for testing
    # import logging
    # import os
    # from scripts.run_incumbent_benders_solver import BendersSolver
    
    # config_path = "configs/ssn_small_config.json"
    # with open(config_path, 'r') as f:
    #     config = json.load(f)
    
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S'
    # )
    
    # solver = BendersSolver(config=config)
    main()