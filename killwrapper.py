import subprocess
import signal
import os
import time

def run_process_and_wait():
    # Define the command to run the script
    command = ['python3', 'wrapper.py']

    # Start the process
    process = subprocess.Popen(command, preexec_fn=os.setsid)

    print(f"Started process with PID: {process.pid}")

    try:
        # Wait indefinitely until Ctrl-C is pressed
        while True:
            time.sleep(1)
    except:
        print("\nStopping the process and its subprocesses...")

        # Send SIGKILL to the process group to ensure termination of the process and its children
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)

        # Wait for the process to terminate
        process.wait()

        print("Process and its subprocesses terminated.")

if __name__ == "__main__":
    run_process_and_wait()
