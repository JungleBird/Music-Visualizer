import numpy as np
import matplotlib.pyplot as plt


class Perlin_Noise:

    def __init__(self, x1, x2, div, seed):

        self.size = np.linspace(x1,x2,div,endpoint=False)
        self.seed = seed
        self.perlin_data = None
        self.x, self.y = np.meshgrid(self.size,self.size)

        
    def generate_perlin(self, x=None, y=None, seed=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        seed = self.seed if seed is None else seed

        # permutation table
        np.random.seed(seed)
        p = np.arange(256,dtype=int)
        np.random.shuffle(p)
        p = np.stack([p,p]).flatten()

        # coordinates of the top-left
        xi = x.astype(int)
        yi = y.astype(int)

        # internal coordinates
        xf = x - xi
        yf = y - yi

        # fade factors
        u = self.fade(xf)
        v = self.fade(yf)

        # noise components
        n00 = self.gradient(p[p[xi]+yi],xf,yf)
        n01 = self.gradient(p[p[xi]+yi+1],xf,yf-1)
        n11 = self.gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
        n10 = self.gradient(p[p[xi+1]+yi],xf-1,yf)

        # combine noises
        x1 = self.lerp(n00,n10,u)
        x2 = self.lerp(n01,n11,u)

        perlin_data = self.lerp(x1,x2,v)
        normalized_perlin_data = (perlin_data+1)/np.amax(perlin_data)

        if self.perlin_data is None:
            self.perlin_data = normalized_perlin_data

        return normalized_perlin_data

    def lerp(self, a, b, x):
        return a + x * (b-a)

    def fade(self, t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    #gradient converts h to the right gradient vector and return the dot product with (x,y)
    def gradient(self, h, x, y):
        vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
        g = vectors[h%4]
        return g[:,:,0] * x + g[:,:,1] * y