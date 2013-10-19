Spotted Stingray
================
Names: Jason Ye, Amy Tsai

Platform: OSX 

Location: Amy Tsai is submitting onto the OSX machines.

Extra Features

Refraction: Materials can be given a refraction index with lines such as the following
refraction 1
refractionIndex 1.3
Any number following "refraction" greater than 0 just means there will be refraction. The number following "refractionIndex" is in the index of refraction of the material. It should be greater than 1.

Soft Shadowing: Area lights can be instantiated with lines such as the following
area 1 3 -1 -1 3 -1 1 3 -3 -1 3 -3 1 1 1 20 20
We first name off the x, y, z coordinates of the 4 points that determine the area of the light. (Should be square or rectangle) This should be in order of upper left, upper right, lower left, lower right. The next 3 numbers are the rgb values of the light. The last 2 numbers are the number of horizontal samples, and then the number of vertical samples.
There's also another constructor where you can call 4 already declared vertices rather than name off the xyz coordinates of 4 points.

Anti aliasing: