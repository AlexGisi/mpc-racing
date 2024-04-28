### data-lanebounds.csv
The first file I took which recorded the location of the lanes.

### easy-drive.csv
Using pure pursuit (78bacf9f339a8766d6edba9804bb11213468085b) with k=0.2

### medium-drive.csv
Using pure pursuit (78bacf9f339a8766d6edba9804bb11213468085b) with k=0.6

### fast-drive.csv
Using pure pursuit (78bacf9f339a8766d6edba9804bb11213468085b) with k=1.2

### with-dt.csv
Recorded using pd controller on the `id` branch with sinusoidal throttle. Includes the dt column, which is the 
simulation timestep from the **previous** step.
