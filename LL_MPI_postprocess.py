import os
import matplotlib.pyplot as plt
import numpy as np


data_dict= {}
data_dict_vec_jit = {}
data_dict_vec = {}
data_dict_OG = {}
data_dict_vec_2 = {}
# Initialize empty lists to store size and time data
sizes,sizes_vec,sizes_vec_jit, sizes_OG, sizes_vec_2= [],[],[],[],[]
times,times_vec,times_vec_jit,times_OG,times_vec_2 = [],[],[],[],[]
n_processes_list,n_processes_vec,n_processes_vec_jit = [],[],[]

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
                    sizes_OG.append(size)
                    times_OG.append(time)

# Define the folder path where your slurm files are located
folder_path = '/home/user/Documents/Fourth_year/SciComp/miniProject1/Accelerating_Lebwhol_Lasher/slurm_outputs/slurm_MPI'

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".out"):  # Assuming your slurm files have a ".out" extension
        with open(os.path.join(folder_path, filename), 'r') as file:
            lines = file.readlines()[3:]  # Skip the first three lines

            for line in lines:
                print(line)
                if "MPI_LebwohlLasher.py: Size:" in line:
                    parts = line.split()
                    print(parts)
                    size = int(parts[2].strip(","))
                    time = float(parts[-2].strip())  # Extract time from the line
                    print(f"Size: {size}, Time: {time}")
                    sizes.append(size)
                    times.append(time)
                elif "Num Procs:" in line:
                    n_processes = int(line.split(':')[1].strip())  # Extract number of processors from the line
                    n_processes_list.append(n_processes)
            
            if n_processes is not None and size is not None and time is not None:
                if n_processes not in data_dict:
                    data_dict[n_processes] = {'sizes': [], 'times': []}
                data_dict[n_processes]['sizes'].append(size)
                data_dict[n_processes]['times'].append(time)
print("sadafgasdgdssa",data_dict)

# Print extracted data for verification
cmap = plt.get_cmap('gnuplot')
norm = plt.Normalize(min(n_processes_list), max(n_processes_list))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig,ax = plt.subplots(1,1)
print(sizes,times,n_processes_list)
for size, time,procs in zip(sizes, times,n_processes_list):

    print(f"Size: {size}, Time: {time}, Procs: {procs} AAAAAAAAAAAAA")
    #if procs == 10:
    color = cmap(norm(procs))
    ax.scatter(size,time,c=color,label=procs,s=8)
    plt.xlabel('Size')
    plt.ylabel('Time')

ax.scatter(sizes_OG, times_OG, s=10,c="black")
coefficients = np.polyfit(sizes_OG, times_OG, 2)
polynomial = np.poly1d(coefficients)

# Generate x values for the fitted curve
x_values = np.linspace(min(sizes), max(sizes), 100)

# Calculate corresponding y values
y_values = polynomial(x_values)

# Plot the fitted curve
ax.plot(x_values, y_values, color='black', linewidth=0.7)

# Add colorbar
sm.set_array([])  # You can set an empty array or use your procs list
cbar = plt.colorbar(sm,ax=ax)
cbar.set_label('Number of Processes')
plt.title("MPI")
plt.savefig(f'figs/LL_MPI.png')

folder_path = '/home/user/Documents/Fourth_year/SciComp/miniProject1/Accelerating_Lebwhol_Lasher/slurm_outputs/slurm_MPI_Numba'
# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".out"):  # Assuming your slurm files have a ".out" extension
        with open(os.path.join(folder_path, filename), 'r') as file:
            lines = file.readlines()[3:]  # Skip the first three lines

            for line in lines:
                if "MPI_Vec_LL.py: Size:" in line:
                    parts = line.split()
                    print(parts)
                    size = int(parts[2].strip(","))
                    time = float(parts[-2].strip())  # Extract time from the line
                    print(f"Size: {size}, Time: {time}")
                    sizes_vec.append(size)
                    times_vec.append(time)
                elif "Num Procs:" in line:
                    n_processes = int(line.split(':')[1].strip())  # Extract number of processors from the line
                    n_processes_vec.append(n_processes)


            if n_processes is not None and size is not None and time is not None:
                if n_processes not in data_dict_vec:
                    data_dict_vec[n_processes] = {'sizes': [], 'times': []}
                data_dict_vec[n_processes]['sizes'].append(size)
                data_dict_vec[n_processes]['times'].append(time)
                
# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".out"):  # Assuming your slurm files have a ".out" extension
        with open(os.path.join(folder_path, filename), 'r') as file:
            lines = file.readlines()[3:]  # Skip the first three lines

            for line in lines:                
                if "MPI_Vec_Jit_LL.py: Size:" in line:
                    parts = line.split()
                    print(parts)
                    size = int(parts[2].strip(","))
                    time = float(parts[-2].strip())  # Extract time from the line
                    print(f"Size: {size}, Time: {time}")
                    sizes_vec_jit.append(size)
                    times_vec_jit.append(time)
                elif "Num Procs:" in line:
                    n_processes = int(line.split(':')[1].strip())  # Extract number of processors from the line
                    n_processes_vec_jit.append(n_processes)
            
            if n_processes is not None and size is not None and time is not None:
                if n_processes not in data_dict_vec_jit:
                    data_dict_vec_jit[n_processes] = {'sizes': [], 'times': []}
                data_dict_vec_jit[n_processes]['sizes'].append(size)
                data_dict_vec_jit[n_processes]['times'].append(time)

# Print extracted data for verification
for size, time in zip(sizes_OG, times_OG):
    print(f"Size: {size}, Time: {time}")


fig,ax=plt.subplots(1,1)
# for n_processes, data in data_dict.items():
#     if n_processes%4==0:
ax.scatter(sizes_OG, times_OG, s =8)
ax.set_xlabel('Size')
ax.set_ylabel('Time')
ax.set_title(f'Time vs Size LebwohlLasher')
plt.savefig(f'figs/LL_OG.png')

folder_path = '/home/user/Documents/Fourth_year/SciComp/miniProject1/Accelerating_Lebwhol_Lasher/slurm_outputs/slurm_Vec'

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
                if "Vec_LL.py: Size:" in line:
                    parts = line.split()
                    #print(parts)
                    size = int(parts[2].strip(","))
                    time = float(parts[-2].strip())  # Extract time from the line
                    print(f"Size: {size}, Time: {time}")
                    sizes_vec_2.append(size)
                    times_vec_2.append(time)


fig,ax = plt.subplots(1,1)
for size, time, procs in zip(sizes_vec, times_vec,n_processes_vec):
    print(f"Size: {size}, Time: {time}, Procs: {procs}")
    #if procs == 10:
    color = cmap(norm(procs))
    ax.scatter(size,time,c=color,label=procs,s=8)
    plt.xlabel('Size')
    plt.ylabel('Time')

ax.scatter(sizes_vec_2, times_vec_2, s=10,c="black")
coefficients = np.polyfit(sizes_vec_2, times_vec_2, 2)
polynomial = np.poly1d(coefficients)

# Generate x values for the fitted curve
x_values = np.linspace(min(sizes_vec_2), max(sizes_vec_2), 100)

# Calculate corresponding y values
y_values = polynomial(x_values)

# Plot the fitted curve
ax.plot(x_values, y_values, color='black', linewidth=0.7)

# Add colorbar
sm.set_array([])  # You can set an empty array or use your procs list
cbar = plt.colorbar(sm,ax=ax)
cbar.set_label('Number of Processes')
plt.title("MPI + Numba Vectorization")
plt.savefig(f'figs/LL_MPI_Vec.png')

fig,ax = plt.subplots(1,1)
for size, time,procs in zip(sizes_vec_jit, times_vec_jit,n_processes_vec_jit):
    print(f"Size: {size}, Time: {time}, Procs: {procs}")
    #if procs == 10:
    color = cmap(norm(procs))
    ax.scatter(size,time,c=color,label=procs,s=8)
    plt.xlabel('Size')
    plt.ylabel('Time')

# Add colorbar
sm.set_array([])  # You can set an empty array or use your procs list
cbar = plt.colorbar(sm,ax=ax)
cbar.set_label('Number of Processes')
plt.title("MPI + Numba Jit & Vectorization")


# fig,ax = plt.subplots(1,1)
# for n_processes, data in data_dict.items():
#     if n_processes==10:
#         ax.scatter(data['sizes'], data['times'],label=n_processes)
#         plt.xlabel('Size')
#         plt.ylabel('Time')

# handles, labels = ax.get_legend_handles_labels()
# # sort both labels and handles by labels
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
#ax.legend(handles, labels)

plt.savefig(f'figs/LL_MPI_Vec_Jit.png')
#plt.savefig(f'figs/LL_MPI_Numba_Jit.png')
