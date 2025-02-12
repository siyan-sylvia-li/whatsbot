import subprocess
import re

def schedule_job(command, interval):
    # Schedule a command at specific intervals
    result = subprocess.run(f"echo \"{command}\" | at now + {interval}", shell=True, capture_output=True, text=True)
    std_err = result.stderr
    std_err = std_err.split("\n")[-2]
    job_num = re.search(r"[0-9]+", std_err).group(0)
    return job_num

def cancel_job(job_num):
    subprocess.run(f"atrm {job_num}", shell=True, check=True)

# Define the command
# command = "echo 'Hello from Python hiya' > ~/output.txt"

# # Schedule the job and capture the output
# result = subprocess.run(f"echo \"{command}\" | at now + 1 minutes", shell=True, capture_output=True, text=True)

# # Print the output (which includes the job ID)
# print(result.stdout)
# print(result.stderr)

# # Need to parse stderr to get the job number
# std_err = result.stderr
# std_err = std_err.split("\n")[-2]
# import re

# print(std_err)

# print(re.search(r"[0-9]+", std_err).group(0))

# # import time
# # time.sleep(65)