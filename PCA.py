import numpy as np

class PCA():
    def __init__(self, q, explain_per=95, pca_type="pca1") -> None:
        assert pca_type in ['pca1', 'pca2']
        self.q = q
        self.explain_per = explain_per
        self.pca_type = pca_type
        self.proj_matrix = None
    
    def __pca1(self, data):
        eig_vals, eig_vectors = np.linalg.eig(np.cov(data[:,:self.q],rowvar = False, ddof = 0))
        eig_vals = eig_vals.real
        eig_vectors = eig_vectors.real
        e_indices = np.argsort(eig_vals)[::-1]
        eigenvectors_sorted = eig_vectors[:,e_indices]
        variance_explained = np.array([100*i/sum(eig_vals) for i in eig_vals])#percentage of variance_explained
        n_take = np.where(np.cumsum(variance_explained)>self.explain_per) # chose st total percentage of variance_explained > 99%
        self.n_take = np.min(n_take)+1
        print('n components chosen:', self.n_take)
        self.proj_matrix = eigenvectors_sorted[:, :self.n_take]
    
    def __pca2(self, data):
        U, S, VT = np.linalg.svd(data[:,:self.q], full_matrices=False)
        n_take = np.where(np.cumsum(S)/sum(S)>self.explain_per/100) 
        n_take = np.min(n_take)+1
        print('n components chosen:', n_take)
        self.n_take = n_take
        V = VT.T
        self.proj_matrix = V[:, :n_take]
    
    def fit(self, data):
        if self.pca_type == "pca1":
            self.__pca1(data)
        else:
            self.__pca2(data)
    
    def fit_transform(self, data):
        if self.pca_type == "pca1":
            self.__pca1(data)
            reduced_matrix = data[:, :self.q]@ self.proj_matrix
        else:
            self.__pca2(data)
            reduced_matrix = data[:, :self.q].dot(self.proj_matrix)

        return reduced_matrix
    
    def transform(self, data):
        assert self.proj_matrix is not None
        not_reduced_part = data[:, self.q:]
        if self.pca_type == "pca1":
            reduced_matrix = data[:, :self.q]@ self.proj_matrix
            reduced_matrix = np.hstack((reduced_matrix, not_reduced_part))
        else:
            reduced_matrix = data[:, :self.q].dot(self.proj_matrix)     
            reduced_matrix = np.hstack((reduced_matrix, not_reduced_part))
        return reduced_matrix