import os
import imageio

# Directory where the images are stored
SAVEDIR = "/home/alex/Pictures/OL-AUTO"

# List of files sorted by integer in filename
file_list = sorted(
    [os.path.join(SAVEDIR, file) for file in os.listdir(SAVEDIR) if file.endswith('.png')],
    key=lambda x: int(os.path.splitext(x)[0].split('/')[-1])
)

# Create a GIF
output_gif_path = os.path.join(SAVEDIR, "output_animation2.gif")
with imageio.get_writer(output_gif_path, mode='I', duration=1.5, loop=0) as writer:
    for filename in file_list:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF created at {output_gif_path}")
