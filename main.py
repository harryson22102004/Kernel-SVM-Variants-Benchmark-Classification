import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
 
def rbf_kernel(X1, X2, gamma=1.0):
    sq = np.sum(X1**2,1,keepdims=True) + np.sum(X2**2,1) - 2*X1@X2.T
    return np.exp(-gamma*sq)
 
def polynomial_kernel(X1, X2, degree=3, coef=1):
    return (X1@X2.T + coef)**degree
 
def sigmoid_kernel(X1, X2, alpha=0.01, c=0):
    return np.tanh(alpha*(X1@X2.T) + c)
def laplacian_kernel(X1, X2, gamma=1.0):
    dist = np.sum(np.abs(X1[:,None,:]-X2[None,:,:]),axis=-1)
    return np.exp(-gamma*dist)
 
datasets = [
    ('moons',  *make_moons(n_samples=400, noise=0.2, random_state=42)),
    ('circles',*make_circles(n_samples=400, noise=0.1, factor=0.5, random_state=42)),
    ('linear', *make_classification(n_samples=400, n_features=10, random_state=42)),
]
 
kernels = [('RBF',     SVC(kernel='rbf', C=1.0, gamma='scale')),
           ('Poly(3)', SVC(kernel='poly', degree=3, C=1.0)),
           ('Sigmoid', SVC(kernel='sigmoid', C=0.1)),
           ('Linear',  SVC(kernel='linear', C=1.0))]
 
scaler = StandardScaler()
print(f"{'Dataset':8s} | {'Kernel':10s} | Accuracy")
print("-"*40)
for dname, X, y in datasets:
    Xs = scaler.fit_transform(X)
    for kname, model in kernels:
        scores = cross_val_score(model, Xs, y, cv=5)
        print(f"{dname:8s} | {kname:10s} | {scores.mean():.3f} ± {scores.std():.3f}")
