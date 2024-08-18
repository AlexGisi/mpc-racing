# mpc-racing
This repository contains an implementation of MPC-based racing in a Carla simulation. Features include
- Smooth parameterization of racing tracks
- Model predictive control formulation which maximizes the vehicle's progress along path
- Custom combined PID and pure pursuit racing-tuned controllers for comparison
- Optimization of MPC parameters using genetic algorithm
- Kinematic, dynamic, and blended bicycle models with custom extensions to accurately model the simulation vehicle

The simulation is built on the GRAIC platform for the CARLA simulator.

## Install
First, follow the instructions to install GRAIC [here](https://docs.google.com/document/d/1O0thKd-WcQzPpEvyfJZmjEr0xCWvgUkzzftlyZxOi_A/edit?usp=sharing). I found the following modifications are necessary:

1. Get the carla install by following instructions at https://github.com/carla-simulator/carla/issues/7017#issuecomment-1908462106

2. Update networkx to 2.8: `pip install networkx==2.8`

3. Add scenario_runner-0.9.13/srunner/tests/carla_mocks to pythonpath: `export PYTHONPATH=$PYTHONPATH:/opt/scenario_runner-0.9.13/srunner/tests/carla_mocks`

4. install tkinter: `sudo apt-get install python3-tk`

If you did not install CARLA to `/opt/carla-simulator`, then change the parameters in `auxillary/carla.py` to match your install location. Then when you run `python3 auxillary/carla.py`, it should load the CARLA simulator on the Shanghai map.

## Use
How to run (without scenarios):
1. Start carla with `python3 auxillary/carla.py`. 
2. Run the GRAIC wrapper using the script `python3 automatic_control_GRAIC.py`.

Note the `main` branch implements the model predictive controller, while the `pp` branch implements the pure pursuit controller with the best parameters found via GA.

## Project structure 
The `control` directory holds mpc-related files. the `models` directory holds files related
to the development of the vehicle models (these are re-implemented in casadi symbolics in 
`control/mpc.py`). The `splines` directory holds files related to the parameterization of
the centerline and the lane boundaries. The `script` directory contains testing and visualization
scripts for various parts of the project.

## Contributors
- [@eeshanzele](https://github.com/eeshanzele)
- [@soumilsg](https://github.com/Soumilsg)
- [@ztmuiuc](https://github.com/ztmuiuc)
