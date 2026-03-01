# https://stepik.org/a/246258


# Singular Value Decomposition (SVD) Solver Course

> 🚀 **Professional course project for understanding and implementing Singular Value Decomposition from scratch in Python**

---

## 🔥 Project Overview

This repository explains and implements **Singular Value Decomposition (SVD)** with:

- Mathematical derivation
- Geometric interpretation
- Matrix factorization implementation
- Numerical algorithms
- Applications in Machine Learning
- Low-rank approximation

The project is optimized for:

- ✅ GitHub search
- ✅ Google indexing
- ✅ Data science portfolio
- ✅ Academic visibility

---

## Keywords

```

singular value decomposition
svd from scratch
matrix factorization
linear algebra
numerical linear algebra
low rank approximation
pca using svd
eigen decomposition
machine learning math
python svd implementation

```

---

## 📚 Mathematical Foundation

Singular Value Decomposition factorizes a matrix:

$$
A \in \mathbb{R}^{m \times n}
$$

into:

$$
A = U \Sigma V^T
$$

Where:

- $$U \in \mathbb{R}^{m \times m}$$ — left singular vectors
- $$\Sigma \in \mathbb{R}^{m \times n}$$ — diagonal matrix of singular values
- $$V \in \mathbb{R}^{n \times n}$$ — right singular vectors

---

### 🔎 Eigenvalue Connection

SVD is computed via:

$$
A^T A = V \Lambda V^T
$$

$$
AA^T = U \Lambda U^T
$$

Singular values:

$$
\sigma_i = \sqrt{\lambda_i}
$$

---

## ⚡ Low-Rank Approximation

Best rank-$k$ approximation:

$$
A_k = U_k \Sigma_k V_k^T
$$

Minimizes Frobenius norm error:

$$
\min_{rank(B)=k} ||A - B||_F
$$

This is the foundation of:

- PCA
- Data compression
- Recommendation systems
- Noise reduction

---

## 🧠 Project Structure

```

svd-solver-course/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── svd_numpy.py
│   ├── svd_from_scratch.py
│   ├── power_method.py
│
├── examples/
│   └── demo.py
│
├── docs/
│   ├── derivation.md
│   ├── geometry.md
│
├── images/
│   └── svd_visualization.png
│
└── index.html

````

---

## 🐍 Example Implementation

### SVD From Scratch (Basic Version)

```python
import numpy as np

class SVDSolver:

    def compute(self, A):
        ATA = A.T @ A
        eigenvalues, V = np.linalg.eigh(ATA)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]

        singular_values = np.sqrt(np.maximum(eigenvalues, 0))
        Sigma = np.diag(singular_values)

        U = A @ V @ np.linalg.inv(Sigma)

        return U, Sigma, V.T
````

---

## 🚀 Applications

* Dimensionality reduction
* PCA computation
* Image compression
* Recommendation systems
* Noise filtering
* Feature extraction

