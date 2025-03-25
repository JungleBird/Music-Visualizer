import pyglet
from block_grid import Block_Grid
from perlin_noise import Perlin_Noise
from audio_parser import Audio_Parser
from component_analyzer import Component_Analyzer
from sphere import Sphere
import random
import time as time
import numpy as np
from rbm import RBM
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
import math

filename = '[file path to .wav file goes here]'

green = np.array([
            30, 132, 73,
            30, 132, 73,
            30, 132, 73,
            30, 132, 73
         ],dtype='int16')

teal = np.array([
            14, 123, 136,
            14, 123, 136,
            14, 123, 136,
            14, 123, 136
        ],dtype='int16')

orange = np.array([
            255, 102, 25,
            255, 102, 25,
            255, 102, 25,
            255, 102, 25
         ],dtype='int16')

red = np.array([
            255, 0, 0,
            255, 0, 0,
            255, 0, 0,
            255, 0, 0
         ],dtype='int16')

pink = np.array([
            255, 51, 255,
            255, 51, 255,
            255, 51, 255,
            255, 51, 255
         ],dtype='int16')

purple = np.array([
            98, 76, 153,
            98, 76, 153,
            98, 76, 153,
            98, 76, 153
         ],dtype='int16')

dimlight = np.array([
            32, 32, 32,
            32, 32, 32,
            32, 32, 32,
            32, 32, 32
        ],dtype='int16')

perlinNoise = Perlin_Noise(0, 12, 32, 2)
perlinNoiseData = (perlinNoise.generate_perlin() + 1)
perlinNoiseData = perlinNoiseData/np.amax(perlinNoiseData)

iterations = 20
num_components = 6
component_draw_size = 2
componentAnalyzer = Component_Analyzer(num_components, iterations)

basis_vector = [6, 9, 6]
x_offset = 16
y_offset = 16
grid_rate = 16
colors = [purple, pink, red, orange]
colorPaletteSize = len(colors)

def rotate_vector(v,theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return np.dot(R,v)

class Visual_Music_Player(pyglet.window.Window):

    def __init__(self, width, height, fft_mode=False):
        super(Visual_Music_Player,self).__init__(width, height, fullscreen = False)

        self.width = width
        self.height = height
        self.fft_mode = fft_mode
        self.window_batch = pyglet.graphics.Batch()
        self.grid = None
        self.audio_parser = None
        self.feature_buffer = None
        self.feature_buffer_size = 16
        
        #currently we're sticking to using a sphere
        self.surface_geometry = Sphere(16)

        #move this to an RBM class
        self.r = RBM(num_visible = 128, num_hidden = 32)
        self.r2 = RBM(num_visible = 32, num_hidden = 4)

    def load_audio_parser(self, file_path, chunk):
        self.audio_parser = Audio_Parser(file_path, chunk)

    def load_grid(self, x_blocks, y_blocks):
        #TODO: validate so grid size/number of blocks = integer
        self.grid = Block_Grid([1,120], 704, 704, x_blocks, y_blocks)
        self.grid.set_grid(self.window_batch)
        self.grid.set_blocks(self.window_batch)

    def on_mouse_press(self, x, y, button, modifiers):
        self.grid.click_block(x, y, orange)

    def on_draw(self):
        self.clear()
        self.window_batch.draw()

    def update(self, dt):
        if self.grid.active_blocks:
            self.grid.update_blocks()

        if self.audio_parser:
            buffer_data = self.audio_parser.play_chunk()

            if buffer_data is not None:
                fft_data, fftx = self.audio_parser.audio_fft(buffer_data, self.audio_parser.sample_rate, self.audio_parser.chunk_size)


                if self.fft_mode:
                    parts, energy = self.audio_parser.partition_chunk(fft_data)
                    indexed_vector = enumerate(parts)
                    self.grid.activate_blocks(indexed_vector, orange)

                    if energy > 128:
                        self.grid.activate_black_blocks(dimlight)

                else:

                    parts, energy = self.audio_parser.partition_chunk(fft_data)
                    indexed_vector = enumerate(parts)
                    self.grid.activate_blocks(indexed_vector, teal)

                    #audio perception is logarithmically scaled
                    #bin sizes start small for lower frequencies and progressively get larger for higher frequencies
                    fft_reduced_0 = self.audio_parser.downsample(fft_data[:148], 2)
                    fft_reduced_1 = self.audio_parser.downsample(fft_data[148:292], 4)
                    fft_reduced_2 = self.audio_parser.downsample(fft_data[292:436], 8)

                    fft_reduced = np.append(fft_reduced_0, fft_reduced_1)
                    fft_reduced = np.append(fft_reduced, fft_reduced_2)

                    fft_reduced[fft_reduced < self.audio_parser.chunk_size] = 0
                    fft_reduced = fft_reduced//self.audio_parser.chunk_size

                    #take the log of the frequency bin values to keep them between 0 and 1
                    #Riemann Boltzmann Machine inputs take values between 0 and 1
                    fft_reduced = np.log1p(fft_reduced)


                    if self.feature_buffer is None:
                        self.feature_buffer = np.array([fft_reduced])
                    else:

                        #TODO: implement fast, circular FIFO buffer in numpy with minimal memory interactions by using an array of index pointers
                        if self.feature_buffer.shape[0] >= self.feature_buffer_size:
                            self.feature_buffer[:-1] = self.feature_buffer[1:]
                            self.feature_buffer[-1] = fft_reduced
                        else:
                            self.feature_buffer = np.append(self.feature_buffer, [fft_reduced], axis=0)

                        #Process feature buffer when filled
                        if self.feature_buffer.shape[0] >= self.feature_buffer_size:
                            w, h, n = componentAnalyzer.mu_method(self.feature_buffer)
                            features = np.where(h > 0, h, 0)
                            basis = np.where(w > 0, w, 0)

                            #Train RBM model layers
                            self.r.train(features, max_epochs=10)
                            next_features = self.r.run_visible(features)
                            self.r2.train(next_features, max_epochs=10)

                            #Generate output layer of RBM model
                            output_layer = np.zeros(shape=(num_components,4))

                            for index, feature in enumerate(features):
                                latent_layer0 = self.r.run_visible(np.array([feature]))
                                latent_layer1 = self.r2.run_visible(latent_layer0)
                                output_layer[index][:] = latent_layer1[:]

                            #Convert output layer nodes into quaternion values and rotate the basis vector
                            quaternions = list(map(lambda hidden: Rotation.from_quat(hidden), output_layer))
                            spatial_vectors = list(map(lambda quat: quat.apply(basis_vector), quaternions))
                            xyz_vectors = list(map(lambda v: [int(v[1]), int(v[0]), v[2]*math.pi/64], spatial_vectors))
                            xy_rotated = list(map(lambda xyz: rotate_vector([xyz[0], xyz[1]], xyz[2]), xyz_vectors))

                            #Activate blocks on grid
                            for index, xy in enumerate(xy_rotated):

                                t_data, t_pos = self.surface_geometry.draw_texture(int(xy[0]) + x_offset, int(xy[1]) + y_offset, component_draw_size, perlinNoiseData)
                                self.grid.activate_blocks(t_pos, colors[index%colorPaletteSize], t_data, grid_rate)


                            #Truncate the feature buffer to avoid re-processing previous features
                            self.feature_buffer = self.feature_buffer[8:]

chunk_size = 1024*2
music_visualizer = Visual_Music_Player(705, 961, fft_mode=False)
music_visualizer.load_audio_parser(filename, chunk_size)
music_visualizer.load_grid(32, 32)

pyglet.clock.schedule(music_visualizer.update)
pyglet.app.run()

