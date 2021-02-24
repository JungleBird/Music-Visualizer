from pyglet import graphics, gl
from block_panel import Block_Panel
import numpy as np
from functools import partial 
import time
import copy
import math
#import ctypes

class Block_Grid:
    
    #TODO: add starting/ending point (to draw grids within grids, starting from pixel value (x,y))
    def __init__(self,
                 origin = [0,0],
                 grid_width = None,
                 grid_height = None,
                 row_block_number = None,
                 col_block_number = None):

        #self.block_grid_batch = graphics.Batch()
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.blocks_per_row = row_block_number
        self.blocks_per_col = col_block_number

        self.row_v = (row_block_number+1)*2
        self.col_v = (col_block_number+1)*2

        self.grid_bounds = {
                            'from': {'x':origin[0], 'y':origin[1]},
                            'to': {'x':origin[0]+grid_width, 'y':origin[1]+grid_height},
                            'block': {
                                        'width': grid_width//row_block_number,
                                        'height': grid_height//col_block_number
                                     }
                           }
        
        self.grid_background_vertex_list = []
        
        self.row_line_group = graphics.OrderedGroup(1)
        self.col_line_group = graphics.OrderedGroup(1)
        self.block_panel_group = graphics.OrderedGroup(0)
        self.grid_background_group = graphics.OrderedGroup(2)
        self.line_group = graphics.OrderedGroup(2)

        self.row_line_vertices = []
        self.col_line_vertices = []
        self.block_vertices = []

        self.row_markers = []
        self.col_markers = []

        self.row_line_colors = []
        self.col_line_colors = []
        self.block_colors = []

        self.block_panels = {}
        self.grid_set = False
        self.block_set = False
        self.block_vertex_list = None

        self.zero_color = np.array([
                                0,0,0,
                                0,0,0,
                                0,0,0,
                                0,0,0
                            ],dtype='int16')

        self.active_blocks = set()

        self.update_loaded = False

    #TODO: use one loop for both lines and colors / maybe vectorize functions
    def set_grid(self, batch):

        if not self.grid_set:
            self.row_line_vertices = [row_line for y in range(self.blocks_per_row+1) for row_line in [self.grid_bounds['from']['x'], self.grid_bounds['from']['y']+y*self.grid_bounds['block']['height'], self.grid_bounds['to']['x'], self.grid_bounds['from']['y']+y*self.grid_bounds['block']['height']]]
            self.row_line_colors = [row_color for row in range(self.blocks_per_row+1) for row_color in [64,64,64, 64,64,64]]    
            self.row_markers = self.row_line_vertices[3::4]

            self.col_line_vertices = [col_line for x in range(self.blocks_per_col+1) for col_line in [self.grid_bounds['from']['x']+x*self.grid_bounds['block']['width'], self.grid_bounds['from']['y'], self.grid_bounds['from']['x']+x*self.grid_bounds['block']['width'], self.grid_bounds['to']['y']]]
            self.col_line_colors = [col_color for col in range(self.blocks_per_col+1) for col_color in [64,64,64, 64,64,64]]
            self.col_markers = self.col_line_vertices[::4]

            self.grid_set = True

        if batch:
            batch.add(self.col_v, gl.GL_LINES, self.col_line_group,
                ('v2i/static', (self.col_line_vertices)),
                ('c3B/dynamic', (self.col_line_colors))
            )
            
            batch.add(self.row_v, gl.GL_LINES, self.row_line_group,
                ('v2i/static', (self.row_line_vertices)),
                ('c3B/dynamic', (self.row_line_colors))
            )
    
    #UPDATE GRID COLORS... NEED GRID OBJECT
    def update_grid(self):
        pass

    #TODO: UPDATE THIS TO BE MORE LIKE THE UPDATE_BLOCKS FUNCTION
    def set_blocks(self, batch):

        if self.grid_set:
            #Generate origin point for block panels inside of grid
            #self.block_panel_array = [[(self.grid_bounds['from']['x']+x*self.grid_bounds['block_width'], self.grid_bounds['from']['y']+y*self.grid_bounds['block_height']) for x in range(self.blocks_per_row+1)] for y in range(self.blocks_per_col+1)]

            for y in range(self.blocks_per_col):
                for x in range(self.blocks_per_row):
                    vertex_index = 12*(x + y*self.blocks_per_row)
                    block_panel = Block_Panel(self.grid_bounds, x, y, vertex_index)

                    self.block_panels[(x,y)] = block_panel
                    #probably faster to use a put method instead of extending array?
                    self.block_vertices.extend(block_panel.shape_props)
                    self.block_colors.extend(block_panel.color_props)

            self.block_set = True

            if batch:
                self.block_vertex_list = batch.add(self.blocks_per_row*self.blocks_per_col*4, gl.GL_QUADS, self.block_panel_group, 
                    (('v2i/dynamic', self.block_vertices)), 
                    (('c3B/stream', self.block_colors))
                )

    #TODO: MOVE OPERATIONS OVER TO THE BLOCK_PANEL OBJECT WHERE POSSIBLE
    def update_blocks(self):

        def update_active_blocks(s):
            try:
                block = self.block_panels[s]

                if block.rate is not None:
                    block.color_props += block.rate
                    block.color_props[block.color_props < 0] = 0
                    block.color_props = np.where(block.color_props > block.threshold, block.threshold, block.color_props)

                if np.array_equal(block.color_props, block.threshold):
                    block.rate = -8

                if np.equal(block.color_props, 0).all():
                    block.rate = None
                    self.active_blocks.remove(s)

                self.block_vertex_list.colors[block.vertex_index:block.vertex_index+12] = block.color_props
                self.block_panels[s] = block
                return 1
            except:
                return 0

        current_active_blocks = copy.copy(self.active_blocks)
        updated = np.fromiter(map(update_active_blocks, current_active_blocks), dtype='int16')

        if not updated.all():
            e = np.where(updated == 0)
            print('failed to update blocks', e)

        return updated

    def check_grid_bounds(self, x, y):
        return self.grid_bounds['from']['x'] <= x <= self.grid_bounds['to']['x'] and self.grid_bounds['from']['y'] <= y <= self.grid_bounds['to']['y']

    def get_block_index(self, w, h):
        x, y = (w-self.grid_bounds['from']['x'])//self.grid_bounds['block']['width'], (h-self.grid_bounds['from']['y'])//self.grid_bounds['block']['height']
        return x, y

    def toggle_block(self, x, y, target, rate=None):
        self.block_panels[(x,y)].threshold = target
        self.block_panels[(x,y)].rate = rate if rate else target
        self.active_blocks.add((x,y))

    def click_block(self, w, h, target, rate=None):
        #values w, h from mouse button click
        x, y = (w-self.grid_bounds['from']['x'])//self.grid_bounds['block']['width'], (h-self.grid_bounds['from']['y'])//self.grid_bounds['block']['height']
        self.block_panels[(x,y)].threshold = target
        self.block_panels[(x,y)].rate = rate if rate else target
        self.active_blocks.add((x,y))


    def activate_blocks(self, points, color, texture=None, rate=None):

        def activate(xy):
            try:
                x, y = xy
                if y < 0: return 1 #used for FFT mode, otherwise we won't encounter a negative value

                block = self.block_panels[(x,y)]
                block.rate = rate if rate else color

                color_target = color
                if texture is not None:
                    color_target = (color_target*texture[x][y]).astype(int)
                    
                block.threshold[:] = color_target[:]
                self.block_panels[(x,y)] = block
                self.active_blocks.add((x,y))

                return 1
            except:
                return 0

        activated = np.fromiter(map(activate, points), dtype='int16')
        
        if not activated.all():
            e = np.where(activated == 0)
            print('failed to update blocks', e)

        return activated

    def activate_black_blocks(self, target, rate=None):
        black_blocks = list(set(self.block_panels.keys()).difference(self.active_blocks))
        self.activate_blocks(black_blocks, target)

    #?????????
    def draw_line_function(self, line_function, batch=None):

        dimlight = np.array([
                    48, 48, 48,
                    48, 48, 48,
                    48, 48, 48,
                    48, 48, 48
                ],dtype='int16')

        line_colors = np.array([
                            255, 255, 25
                        ],dtype='int16')

        line_vertex = []
        num_line_vertex = len(self.col_markers)

        prev_point = None

        xpoints = range(self.grid_bounds['from']['x'], self.grid_bounds['to']['x'], 4)

        for x in xpoints:
            y = line_function(x,self.grid_bounds['from']['y'])
            i, j = self.get_block_index(x, y)

            #TODO: add visited set and determine color fill opacity from mirror points method
            if 0 <= i < 32 and 0 <= j < 32:
                self.toggle_block(i, j, dimlight)

        for x in self.col_markers:
            y = line_function(x,self.grid_bounds['from']['y'])
            line_vertex.extend([x,y])

        line_color_vertex = [color for x in range(num_line_vertex) for color in line_colors]

        if batch:
            batch.add(num_line_vertex, gl.GL_LINE_STRIP, self.line_group,
            ('v2i/static', (line_vertex)),
            ('c3B/static', (line_color_vertex))
            )
