# for akshay
note the final work is across the branches `pp` and `mpc`. The `pp` branch implements the pure
pursuit controller with the best parameters found via GA. the `mpc` branch implements the mpc in 
agent.py. 

to run the final agent: `git checkout pp`.

the `control` directory holds mpc-related files. the `models` directory holds files related
to the development of the vehicle models (these are re-implemented in casadi symbolics in 
`control/mpc.py`). the `splines` directory holds files related to the parameterization of
the centerline and the lane boundaries. the `script` directory contains testing and visualization
scripts for various parts of the project. 

we did not consider obstacles in our problem statement, hence we did not include a video
with scenarios. the extension is reasonably straightforward; we discuss this in the video.

running shanghai video:
https://drive.google.com/file/d/1-f9xNyzWwORthsTG67wBQts_fH9zjB1e/view?usp=sharing


# autobots-race
How to run:
1. Start carla and change to the shanghai map, for example using the script `auxillary/carla.py`. 
2. Run the GRAIC wrapper, for example using the script `killwrapper.py`.

# GRAIC 2023
Installation Documentation could be found [Here](https://docs.google.com/document/d/1O0thKd-WcQzPpEvyfJZmjEr0xCWvgUkzzftlyZxOi_A/edit?usp=sharing)

1. All you need to submit is `agent.py`. And all you implementation should be contained in this file.
2. If you want to change to different map, just modify line 6 in `wrapper.py`. 5 maps (shanghai_intl_circuit, t1_triple, t2_triple, t3, t4) have been made available to public, while we hold some hidden maps for testing.
3. If you would like to test your controller without the scenarios, comment out line 10 in `wrapper.py`.

Benchmark score using our very naive controllers on below tracks with **no scenarios**:

| Track | Score|
|-----|--------|
| triple_t1 | 45 |
| triple_t2 | 74 |
| t3 | 82 |
| t4 | 57 |
| shanghai_intl_circuit | 122 |
