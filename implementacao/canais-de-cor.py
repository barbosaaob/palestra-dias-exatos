import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# create colormaps
nbeans = 100
red_cmap = LinearSegmentedColormap.from_list("red_cmap", [(0,0,0), (1,0,0)], N=nbeans)
green_cmap = LinearSegmentedColormap.from_list("green_cmap", [(0,0,0), (0,1,0)], N=nbeans)
blue_cmap = LinearSegmentedColormap.from_list("blue_cmap", [(0,0,0), (0,0,1)], N=nbeans)

img = plt.imread("rio.png")
fig = plt.imshow(img)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# Red
fig = plt.imshow(img[:,:,0])
fig.set_cmap(red_cmap)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# Green
fig = plt.imshow(img[:,:,1])
fig.set_cmap(green_cmap)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# Blue
fig = plt.imshow(img[:,:,2])
fig.set_cmap(blue_cmap)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()
