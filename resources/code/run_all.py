"""Run every numbered file in order and save what each one prints in the results folder."""
import os
import subprocess
import sys

os.makedirs("results", exist_ok=True)

for file_name in sorted(os.listdir(".")):
    if file_name[:2].isdigit() and file_name.endswith(".py"):
        print("Running", file_name, "...")
        output_file = open(os.path.join("results", file_name.replace(".py", ".txt")), "w")
        subprocess.run([sys.executable, file_name], stdout=output_file, stderr=subprocess.STDOUT)
        output_file.close()

print("Done. Outputs are in the results folder.")
