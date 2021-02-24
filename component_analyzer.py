import numpy as np

class Component_Analyzer():

    def __init__(self, num_components, update_iterations):
        self.num_components = num_components
        self.update_iterations = update_iterations

    def random_initialization(self, A,rank):
        number_of_documents = A.shape[0]
        number_of_terms = A.shape[1]
        W = np.random.uniform(1,2,(number_of_documents,rank))
        H = np.random.uniform(1,2,(rank,number_of_terms))
        return W,H
                        
    def mu_method(self,A,components=None,iterations=None):
        
        if components is None:
            components = self.num_components

        if iterations is None:
            iterations = self.update_iterations

        W ,H = self.random_initialization(A,components)

        norms = []
        e = 1.0e-10
        for n in range(iterations):
            # Update H
            W_TA = W.T@A
            W_TWH = W.T@W@H+e
            for i in range(np.size(H, 0)):
                for j in range(np.size(H, 1)):
                    H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]
            # Update W
            AH_T = A@H.T
            WHH_T =  W@H@H.T+ e

            for i in range(np.size(W, 0)):
                for j in range(np.size(W, 1)):
                    W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]

            norm = np.linalg.norm(A - W@H, 'fro')
            norms.append(norm)
        return np.matrix(W), np.matrix(H), norms