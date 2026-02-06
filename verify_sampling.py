import subprocess
import os
import sys

def verify_sampling_with_datadir():
    print("--- Verifying Sampling with --data_dir ---")
    
    # We need a dummy experiment or just point to data we have
    # Since we can't easily train a model here, we will just check if sample.py accepts the argument 
    # and processes it correctly (by checking the help or running it and expecting a specific error/success message)
    
    # Test 1: Check help message for --data_dir
    print("Checking help message...")
    try:
        output = subprocess.check_output([sys.executable, "src/sample.py", "--help"], stderr=subprocess.STDOUT).decode()
        if "--data_dir" in output:
            print("SUCCESS: --data_dir argument exists in sample.py")
        else:
            print("FAILURE: --data_dir argument missing in sample.py help")
            return
    except subprocess.CalledProcessError as e:
        print(f"Error checking help: {e.output.decode()}")
        return

    # Test 2: Point to the SPY data and ensure it loads (it will fail on checkpoint load, but we can check if it gets past path resolution)
    # We will use a non-existent experiment to trigger an early exit after config loading
    print("\nChecking path resolution pass-through...")
    cmd = [
        sys.executable, "src/sample.py", 
        "--experiment_name", "dummy_test_non_existent",
        "--dataset", "stocks",
        "--data_dir", "data/stocks/SPY_stock_data.csv",
        "--num_samples", "10"
    ]
    
    try:
        # We expect this to fail because dummy_test_non_existent doesn't exist, 
        # but the output should show it identified the dataset as 'stocks' and the data_dir correctly.
        # We need to see if it printed "Loading stock data from data/stocks/SPY_stock_data.csv..."
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, _ = process.communicate()
        output = output.decode()
        
        print("Sampling Output Snippet:")
        print(output[:500])
        
        if "Loading stock data from data/stocks/SPY_stock_data.csv..." in output:
            print("\nSUCCESS: data_dir correctly passed through to loader.")
        else:
            print("\nFAILURE: data_dir did not reach the loader correctly.")
            
    except Exception as e:
        print(f"Error running sampling test: {e}")

if __name__ == "__main__":
    verify_sampling_with_datadir()
