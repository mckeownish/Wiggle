import subprocess
import os

def run_crysol(pdb_file, dat_file, output_dir='crysol_output', smax=0.2):

# Create output directory if it doesn't exist

    os.makedirs(output_dir, exist_ok=True)

    # Run crysol command with subprocess
    result = subprocess.run([
        'crysol', 
        pdb_file, 
        dat_file, 
        '--smax', str(smax)
    ], 
    cwd=output_dir,
    capture_output=True, 
    text=True
    )

    print(result.stderr)

