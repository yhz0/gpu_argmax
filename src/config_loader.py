import yaml
import os
from pathlib import Path
from typing import Dict, Any, Union


class ConfigLoader:
    """
    Configuration loader that supports YAML files with organized structure
    and flattens them for compatibility with existing code.
    """
    
    @staticmethod
    def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a YAML configuration file and flatten the nested structure.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Flattened configuration dictionary compatible with existing code
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if not config_path.suffix.lower() in ['.yaml', '.yml']:
            raise ValueError(f"Expected YAML file, got: {config_path.suffix}")
        
        with open(config_path, 'r') as f:
            nested_config = yaml.safe_load(f)
        
        # Flatten the nested configuration
        flat_config = ConfigLoader._flatten_config(nested_config)
        
        # Convert numeric values to proper types
        flat_config = ConfigLoader._convert_numeric_values(flat_config)
        
        # Validate required parameters
        ConfigLoader._validate_config(flat_config)
        
        return flat_config
    
    @staticmethod
    def _flatten_config(nested_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten the organized YAML structure to flat keys expected by BendersSolver.
        
        Maps:
        - metadata.instance_name -> instance_name
        - input_files.* -> direct keys (smps_core_file, etc.)
        - argmax_pool.* -> direct keys (MAX_PI, etc.)
        - algorithm_parameters.* -> direct keys
        - regularization_control.* -> direct keys
        - validation.* -> direct keys
        - logging.* -> direct keys
        """
        flat = {}
        
        # Metadata section
        if 'metadata' in nested_config:
            for key, value in nested_config['metadata'].items():
                flat[key] = value
        
        # Input files section
        if 'input_files' in nested_config:
            for key, value in nested_config['input_files'].items():
                flat[key] = value
        
        # Argmax pool section
        if 'argmax_pool' in nested_config:
            for key, value in nested_config['argmax_pool'].items():
                flat[key] = value
        
        # Algorithm parameters section
        if 'algorithm_parameters' in nested_config:
            for key, value in nested_config['algorithm_parameters'].items():
                flat[key] = value
        
        # Regularization control section
        if 'regularization_control' in nested_config:
            for key, value in nested_config['regularization_control'].items():
                flat[key] = value
        
        # Validation section
        if 'validation' in nested_config:
            for key, value in nested_config['validation'].items():
                flat[key] = value
        
        # Logging section
        if 'logging' in nested_config:
            for key, value in nested_config['logging'].items():
                flat[key] = value
        
        return flat
    
    @staticmethod
    def _convert_numeric_values(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert string representations of numeric values to proper numeric types.
        
        Args:
            config: Flattened configuration dictionary
            
        Returns:
            Configuration dictionary with numeric values properly typed
        """
        # Define which parameters should be numeric and their expected types
        numeric_params = {
            # Integer parameters
            'MAX_PI': int,
            'MAX_OMEGA': int,
            'SCENARIO_BATCH_SIZE': int,
            'NUM_CANDIDATES': int,
            'NUM_SAMPLES_FOR_POOL': int,
            'max_iterations': int,
            'num_duals_to_add_per_iteration': int,
            'num_workers': int,
            'num_lp_scenarios_per_iteration': int,
            'validation_N1': int,
            'validation_N2': int,
            'validation_seed': int,
            
            # Float parameters
            'ETA_LOWER_BOUND': float,
            'tolerance': float,
            'argmax_tol_cutoff': float,
            'primal_feas_tol': float,
            'initial_rho': float,
            'gamma': float,
            'rho_decrease_factor': float,
            'rho_increase_factor': float,
            'min_rho': float,
            'max_rho': float,
            'validation_confidence_level': float,
        }
        
        # Convert values to proper types
        converted_config = config.copy()
        for param, expected_type in numeric_params.items():
            if param in converted_config:
                try:
                    if isinstance(converted_config[param], str):
                        # Handle scientific notation and other string representations
                        converted_config[param] = expected_type(float(converted_config[param]))
                    elif not isinstance(converted_config[param], expected_type):
                        # Convert to expected type if it's not already
                        converted_config[param] = expected_type(converted_config[param])
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Could not convert parameter '{param}' with value '{converted_config[param]}' to {expected_type.__name__}: {e}")
        
        return converted_config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """
        Validate that required configuration parameters are present.
        
        Args:
            config: Flattened configuration dictionary
            
        Raises:
            ValueError: If required parameters are missing
        """
        required_params = {
            # Input files
            'smps_core_file',
            'smps_time_file', 
            'smps_sto_file',
            'input_h5_basis_file',
            
            # Argmax pool
            'MAX_PI',
            'MAX_OMEGA',
            'SCENARIO_BATCH_SIZE',
            
            # Algorithm parameters
            'ETA_LOWER_BOUND',
            'NUM_SAMPLES_FOR_POOL',
            'tolerance',
            'max_iterations',
            
            # Regularization control
            'initial_rho',
            'gamma',
            'rho_decrease_factor',
            'rho_increase_factor',
            'min_rho',
        }
        
        missing_params = required_params - set(config.keys())
        
        if missing_params:
            raise ValueError(f"Missing required configuration parameters: {sorted(missing_params)}")
        
        # Validate file paths exist
        file_params = ['smps_core_file', 'smps_time_file', 'smps_sto_file', 'input_h5_basis_file']
        for param in file_params:
            if param in config:
                file_path = Path(config[param])
                if not file_path.exists():
                    raise FileNotFoundError(f"File specified in '{param}' does not exist: {file_path}")