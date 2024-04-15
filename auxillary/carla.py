import subprocess
import time
import os
import signal

CARLA_SH = "/opt/carla-simulator/CarlaUE4.sh"
CONFIG_FP = "/opt/carla-simulator/PythonAPI/util/config.py"


print("Checking existing carla processes...")
try:
    subprocess.run(['pkill', '-f', 'carla-simulator'], check=True)
    subprocess.run(['pkill', '-f', 'scenario'], check=True)
    print("Processes killed.")
except subprocess.CalledProcessError as e:
    print("None found")
print("Starting carla...")

proc = subprocess.Popen(CARLA_SH, shell=True, preexec_fn=os.setsid)

time.sleep(3)
subprocess.run(["python3", CONFIG_FP, "-m", "/Game/map_package/Maps/shanghai_intl_circuit/shanghai_intl_circuit"])

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nCtrl-C caught, stopping carla...")
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    proc.wait()

    print("Carla terminated.")

print("Checking existing carla processes...")
try:
    subprocess.run(['pkill', '-f', 'carla-simulator'], check=True)
    subprocess.run(['pkill', '-f', 'scenario'], check=True)
    print("Processes killed.")
except subprocess.CalledProcessError as e:
    print("None found")
    