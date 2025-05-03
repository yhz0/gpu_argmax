import inspect
import sys
import os
from pathlib import Path
import importlib

MODULE_DIRS = [
    '.',
]

# List of tuples: (module_name, class_name)
CLASSES_TO_DOCUMENT = [
    ('smps_reader', 'SMPSReader'),
    ('argmax_operation', 'ArgmaxOperation'),
    ('second_stage_worker', 'SecondStageWorker'),
    ('master', 'AbstractMasterProblem'),
    ('benders', 'BendersMasterProblem'),
]

OUTPUT_FILE = "combined_docstrings.txt"

# --- Helper Functions ---
def import_class(module_name, class_name):
    """Dynamically imports a class from a module."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError:
        print(f"ERROR: Could not import module '{module_name}'. Skipping.")
        return None
    except AttributeError:
        print(f"ERROR: Could not find class '{class_name}' in module '{module_name}'. Skipping.")
        return None
    except Exception as e:
         print(f"ERROR: Unexpected error importing {module_name}.{class_name}: {e}")
         return None

def format_docstrings(classes_to_doc, output_filename):
    """
    Extracts and formats docstrings for specified classes and their methods,
    writing the output to a file.
    """
    output_lines = []
    print("Starting docstring extraction...")

    for module_name, class_name in classes_to_doc:
        print(f" Processing {module_name}.{class_name}...")
        cls = import_class(module_name, class_name)
        if cls is None:
            output_lines.append(f"\n{'='*80}\nERROR: Could not load class {module_name}.{class_name}\n{'='*80}\n\n")
            continue # Skip to the next class if import failed

        # --- Class Documentation ---
        output_lines.append("=" * 80)
        output_lines.append(f"Class: {module_name}.{class_name}")
        output_lines.append("=" * 80)

        class_doc = inspect.getdoc(cls) # getdoc handles indentation nicely
        output_lines.append("\n--- Class Docstring ---\n")
        output_lines.append(class_doc if class_doc else "(No Class Docstring Found)")
        output_lines.append("\n")

        # --- Method Documentation ---
        output_lines.append("\n--- Methods ---\n")
        found_methods = False
        # Get members that are functions, methods, classmethods, staticmethods
        # Attempt to sort by source line number for better order
        try:
            members = sorted(
                inspect.getmembers(cls, inspect.isroutine),
                key=lambda item: inspect.getsourcelines(item[1])[1]
            )
        except (TypeError, OSError, IOError): # Handle errors getting source lines (e.g., for builtins)
            members = sorted(inspect.getmembers(cls, inspect.isroutine))

        for name, member in members:
            # Filter out private/special methods unless it's __init__ or __del__
            # You might want to adjust this filter based on your needs
            if name.startswith('__') and name not in ['__init__', '__del__']:
                continue
            # Exclude inherited methods if desired (optional)
            # if member.__qualname__.split('.')[0] != cls.__name__:
            #    continue

            method_doc = inspect.getdoc(member)
            if method_doc:
                found_methods = True
                try:
                    # Get signature for better context
                    sig = str(inspect.signature(member))
                except (ValueError, TypeError):
                    sig = "(...)" # Fallback if signature cannot be determined

                output_lines.append(f"\nMethod: {name}{sig}")
                output_lines.append("-" * (len(f"Method: {name}{sig}"))) # Underline
                output_lines.append(method_doc)
                output_lines.append("\n")
            # Optionally list methods missing docstrings:
            # elif not name.startswith('_'): # Don't list private without docs
            #    output_lines.append(f"\nMethod: {name}{sig}")
            #    output_lines.append("(No Docstring Found)\n")


        if not found_methods:
            output_lines.append("(No documented methods found for this class)\n")

        output_lines.append("\n\n") # Add space between classes

    # --- Write to File ---
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\nSuccessfully wrote combined docstrings to '{output_filename}'")
    except IOError as e:
        print(f"\nERROR: Failed to write docstrings to file '{output_filename}': {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Add specified directories to sys.path for imports
    for dir_path in MODULE_DIRS:
        full_path = str(Path(dir_path).resolve())
        if full_path not in sys.path:
            sys.path.insert(0, full_path)
            print(f"Added to sys.path: {full_path}")

    format_docstrings(CLASSES_TO_DOCUMENT, OUTPUT_FILE)