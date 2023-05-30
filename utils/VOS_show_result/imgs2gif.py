from PIL import Image

input_filenames = []
# loop through each image file and add it to the list
# Load the images and append them to a list
for i in range(0, 24):
    filename = "/Users/pinxue/Documents/LocalData/collision_demo_video/0/rgba_{:05d}.png".format(i)
    img = Image.open(filename)
    input_filenames.append(img)

output_filename = 'animation.gif'

# Save the list of images as an animated GIF file
input_filenames[0].save(output_filename, save_all=True, append_images=input_filenames[1:], duration=1, loop=0)
