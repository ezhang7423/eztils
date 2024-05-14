"""For getting information about running processes
"""

import nvitop


def get_gpu_processes_commands():
    gpu_processes = []
    # Iterate over each GPU
    for gpu in nvitop.Device.all():
        # Get the processes running on the GPU
        processes = gpu.processes()
        for process in processes.values():
            # Get process details

            pid = process.pid
            command = process.command()
            gpu_processes.append((pid, command))

    return gpu_processes


if __name__ == "__main__":
    gpu_processes = get_gpu_processes_commands()
    if gpu_processes:
        print("Processes running on GPUs:")
        for pid, command in gpu_processes:
            print(f"PID: {pid}, Command: {command}")
    else:
        print("No processes found running on GPUs.")
