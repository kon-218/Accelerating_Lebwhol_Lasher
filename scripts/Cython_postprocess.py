import os
import matplotlib.pyplot as plt

# Define the folder path where your slurm files are located
folder_path = '/home/user/Documents/Fourth_year/SciComp/miniProject1/Accelerating_Lebwhol_Lasher/slurm_outputs/slurm_LL'

data_dict = {}
# Initialize empty lists to store size and time data
sizes = []
times = []

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".out"):  # Assuming your slurm files have a ".out" extension
        with open(os.path.join(folder_path, filename), 'r') as file:
            lines = file.readlines()[3:]  # Skip the first three lines

            for line in lines:
                if "LebwohlLasher.py: Size:" in line:
                    parts = line.split()
                    #print(parts)
                    size = int(parts[2].strip(","))
                    time = float(parts[-2].strip())  # Extract time from the line
                    print(f"Size: {size}, Time: {time}")
                    sizes.append(size)
                    times.append(time)

# Print extracted data for verification
for size, time in zip(sizes, times):
    print(f"Size: {size}, Time: {time}")

plt.figure()
# for n_processes, data in data_dict.items():
#     if n_processes%4==0:
plt.scatter(sizes, times)
plt.xlabel('Size')
plt.ylabel('Time')
plt.title(f'Time vs Size LebwohlLasher')
plt.savefig(f'figs/LL_OG.png')