import numpy as np

class Block_Panel():

    def __init__(self, grid_bounds, row_index, col_index, vertex_index):

        self.row_index = row_index
        self.col_index = col_index

        self.width = grid_bounds['block']['width']
        self.height = grid_bounds['block']['height']

        self.block_origin = (
                                grid_bounds['from']['x'] + self.width*self.row_index, 
                                grid_bounds['from']['y'] + self.height*self.col_index
                            )

        self.block_termin = (
                                self.block_origin[0] + self.width, 
                                self.block_origin[1] + self.height
                            )

        self.shape_props =  [
                                self.block_origin[0], self.block_origin[1],
                                self.block_termin[0], self.block_origin[1],
                                self.block_termin[0], self.block_termin[1],
                                self.block_origin[0], self.block_termin[1]
                            ]

        self.color_props =  np.array([
                                0,0,0,
                                0,0,0,
                                0,0,0,
                                0,0,0
                            ],dtype='int16')

        self.threshold =  np.array([
                                0,0,0,
                                0,0,0,
                                0,0,0,
                                0,0,0
                            ],dtype='int16')

        self.vertex_index = vertex_index

        #TODO: make rate a 1x3 array but broadcastable over 1x12 array
        self.rate = None

    #TODO: create function to affect neighbors based on activation rules
    def update_neighbors(self):
        pass
