import math
import numpy as np

class Sphere:

    def __init__(self, radius):
        self.radius = radius
        self.diameter = radius*2
        self.sphere_map = None
        self.z_map = None
        self.mask_map = None

        self.sphere_map, self.mask_map, self.z_map =  self.draw_sphere(self.radius)

    def draw_sphere(self, radius=None):

        radius = radius if radius is not None else self.radius
        diameter = radius*2 if radius is not None else self.diameter

        sphere_map = np.zeros(shape=(diameter,diameter))
        z_map = np.zeros(shape=(diameter,diameter))
        mask_map = np.zeros(shape=(diameter,diameter))

        center_offset = 0.5 if radius%2 == 0 else 0
        radius_squared = radius*radius

        for x in range(-radius, radius):

            x_plus = x+center_offset
            x_squared = x_plus**2

            for y in range(-radius, radius):

                y_plus = y+center_offset
                y_squared = y_plus**2

                if x_squared + y_squared < radius_squared:

                    z = math.sqrt(radius_squared - x_squared - y_squared)
                    z_map[x+radius][y+radius] = round(z)
                    mask_map[x+radius][y+radius] = 1
                    sphere_map[x+radius][y+radius] = min(z/radius,1)

        return sphere_map, mask_map, z_map
    
    #distort a texture over a surface by warping texture map based on surface height gradient
    #TODO: set min/max x and y limits so we don't have to iterate over the entire grid
    def distort_texture(self, texture, z_map):

        horizontal_distortion = np.zeros(shape=(texture.shape))
        vertical_distortion = np.zeros(shape=(texture.shape))

        for row, arr in enumerate(texture):

            prev_horizontal = None
            prev_vertical = None

            for col, val in enumerate(arr):

                if prev_horizontal is not None and prev_vertical is not None:
                    
                    row_z_diff = (prev_horizontal - z_map[row][col])/2
                    col_z_diff = (prev_vertical - z_map[col][row])/2

                    row_z_diff = math.ceil(row_z_diff) if row_z_diff < 0 else math.floor(row_z_diff)
                    col_z_diff = math.ceil(col_z_diff) if col_z_diff < 0 else math.floor(col_z_diff)                      

                    if 0 <= int(row_z_diff + col) < (self.radius*2)-1:
                        horizontal_distortion[row][col] = texture[row][int(row_z_diff + col)]
                    else: 
                        horizontal_distortion[row][col] = 0

                    if 0 <= int(col_z_diff + row) < (self.radius*2)-1:
                        vertical_distortion[col][row] = texture[col][int(col_z_diff + row)]
                    else: 
                        vertical_distortion[col][row] = 0

                prev_horizontal = z_map[row][col]
                prev_vertical = z_map[col][row]

        return horizontal_distortion*vertical_distortion

    def draw_texture(self, x, y, r, texture, distort=False):

        #If there is no surface to draw texture on, return False
        if self.sphere_map is None:
            return False

        #cut out a spherical/circular section of texture map
        spot_sphere, spot_mask, spot_z = self.draw_sphere(r)

        texture_map = np.zeros(shape=(self.diameter, self.diameter))
        texture_array = []

        #this gets us the boundaries based on our (global) 32x32 grid's coordinate system
        left, right = x-r, x+r
        bottom, top = y-r, y+r

        #left and right values correspond to global coordinates
        for w in range(left, right):
            
            #spot_x value start from 0
            spot_x = w - left

            for h in range(bottom, top):

                #spot_y value start from 0
                spot_y = h - bottom

                if 0 <= w < len(self.mask_map) and 0 <= h < len(self.mask_map):

                    #spot_x and spot_y correspond to the (local) coordinate system of the circular section
                    if spot_mask[spot_x][spot_y] > 0 and self.mask_map[w][h] > 0:
                        
                        texture_val = self.sphere_map[w][h]*texture[w][h]

                        if texture_val > 0.1:
                            texture_map[w][h] = texture_val
                            texture_array.append((w,h))

        if distort:
            distorted_texture_map = self.distort_texture(texture_map, self.z_map)
            texture_map = distorted_texture_map

        return texture_map, texture_array




