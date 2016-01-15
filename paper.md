---
title: 'Convolutional Neural Networks on Graph'
author:
- Michaël Defferrard^1^
  \footnote{Corresponding author - michael.defferrard@epfl.ch}
- Xavier Bresson^1^
- Pierre Vandergheynst^1^
include-before: ^1^ LTS2, Ecole Polytechnique Fédérale de Lausanne,
  Lausanne, Switzerland
keywords:
- graph signal processing
- deep learning
abstract: Disrupting Deep Learning.
date: \today
header-includes:
  \DeclareMathOperator*{\diag}{diag}
  \DeclareMathOperator*{\argmin}{arg\,min}
  \renewcommand{\L}{\mathcal{L}}
  \newcommand{\R}{\mathbb{R}}
---

# Introduction

Large networks allow to learn complex relationships. Although they overfit
easily if the training set is small because there is a lot of parameters to
learn.

Convolutional networks offer a great reduction of parameters on regular
Euclidean domain. This work proposes to generalize it to the discrete model of
manifolds, graphs. Such an approach was proposed by [@bruna_spectral_2013;
@henaff_deep_2015], we will refine it with proper graph signal processing tools.


# Related work

Comparison with fully connected NN (Reuters used in [@srivastava_dropout_2014])

* \+ less parameters via convolution
* ? accuracy

Comparison with first generation graph CNN [@henaff_deep_2015].

* \+ better accuracy
* = same number of parameters
* \- graph construction vs supervised estimation
* Theoretical:
	* \+ proper spectral graph signal processing tools
	* \+ kNN graph (sparse weight matrix) vs fully connected graph. SVD
	  complexity for Fourier basis is O(n^3), we avoid it.
	* \+ METIS vs naive agglomeration
	* \+ Chebyshev polynomials vs splines to approximate filters Filtering from
	  $O(n^2)$ to $O(|E| K)$ where $K$ is the polynomial order approximately
	  $O(n)$ if the weight matrix is sparse, i.e. grows linearly with $N$ This
	  addresses the first limitation mentioned in [@henaff_deep_2015].

# Method

## Spectral graph theory

A graph $G = (V, E, W)$ is defined by a set $V$ of $|V| = M$ nodes, a set of
edges $E$ with their associated weight matrix $W \in \R^{M \times M}$. Two
nodes $v_i$ and $v_j$ are connected if $W_{ij} > 0$.

The non-normalized graph Laplacian is given by
$$\L = D - W$$
where $D$ is the diagonal degree matrix defined as $D_{ii} = \sum_j W_{ij}$.
The normalized graph Laplacian is then given by
$$\L = I - D^{-1/2} W D^{-1/2}$$
where $I$ is the $M \times M$ identity matrix.

In analogy to the real line Fourier transform, a Fourier basis $U =
\{u_\ell\}_{\ell=0}^{\ell=M-1}$ is given by the eigenvectors of the Laplacian
$$\L u_\ell = \lambda_\ell u_\ell$$
with their associated eigenvalues $\lambda_\ell$ [@shuman_emerging_2013;
@hammond_wavelets_2011]. Assuming the graph is connected, we may order the
eigenvalues such that
$$0 = \lambda_0 < \lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_{M-1}.$$
We just diagonalized the Laplacian as
$$\L = U \diag(\lambda) U^T$$
where $\diag(\lambda)$ denotes a diagonal matrix of eigenvalues.

For any signal $x \in \R^{M}$ defined on the vertices of $G$, its graph Fourier transform $\hat{x}$ is defined by
$$\hat{x}(\ell) = \langle u_\ell , x \rangle
= \sum_{m=0}^{M-1} u_\ell(m) x(m).$$

## Graph filter learning

Given a graph $G$, a set $X = \{x_i\}_{i=0}^{N-1}$ of $N$ source graph signals
and their associated set $Y = \{y_i\}_{i=0}^{N-1}$ of target signals, we want
to learn the coefficients $c$ of a graph filter such as to minimize the
convex reconstruction error
$$L = \frac{1}{N} \sum_{i=0}^{N-1}
\| U \diag(c) U^T x_i - y_i \|_2^2 = \frac{1}{N}
\| U \diag(c) U^T X - Y \|_F^2$$ {#eq:loss}
where $\|\cdot\|_2^2$ denotes the squared $\ell_2$ norm and $\|\cdot\|_F^2$ the
squared Frobenius norm.

Note that the non-parametrization of the coefficients $c$ w.r.t. $\lambda$ does
omit all information about frequencies, while we know that the lower
frequencies are more important for clustering [ref NCut].

Rewriting [@eq:loss] in the spectral domain (in a matrix form) gives
$$L = \frac{1}{N} \| \diag(c) U^T X - U^T Y \|_F^2 =
\frac{1}{N} \sum_{i=0}^{M-1} \|c_i(U^TX)_{i,\cdot} - (U^TY)_{i,\cdot} \|_2^2$$
where the right hand side has been decomposed w.r.t. the scalar coefficients
$c_i$. The gradient for each coefficient is then given by
$$\nabla_{c_i} L = \frac{2}{N}
( c_i (U^T X)_{i,\cdot} - (U^T Y)_{i,\cdot} ) (X^T U)_{\cdot,i}$$
and can be rewritten in a vector form as
$$\nabla_{c} L = \frac{2}{N}
\left( U^T X \circ ( c \circ U^T X - U^T Y ) \right) 1_N$$ {#eq:gradient}
where $1_N$ denotes a unit vector of length $N$.

A direct solution is given by the optimality condition $\nabla_{c}L=0$ such that
$$c^o = \argmin_c L = (U^T X \circ U^T Y) 1_N \circ
\left( (U^T X \circ U^T X) 1_N \right)^{-1}.$$
Note that this method is impractical for large $M$ and $N$ (sufficiently
large for $X$ and $Y$ to not fit in memory). A Stochastic Gradient Descent
based on [@eq:gradient] will do the trick.

There is however two major computational drawbacks to this optimization
process: (1) the Fourier transform $U^TX$ of a set of signals costs $O(M^2N)$
operations and (2) the eigenvalue decomposition $\L=U\diag(\lambda)U^T$ costs
$O(M^3)$. The total cost of filtering is thus $O(M^2 \max(M,N))$, a problem
stated in [@henaff_deep_2015].

### Fast algorithm

The idea is to approximate the filter (in the spectral domain) by a truncated
Chebyshev expansion and to recursively compute the Chebyshev polynomials from
the Laplacian, avoiding the Fourier basis altogether [@hammond_wavelets_2011]. 

Recall that the Chebyshev polynomial $T_k(x)$ of order $k$ may be generated by
the stable recurrence relation $T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$ with $T_0
= 1$ and $T_1 = x$. These polynomials form an orthogonal basis for $L^2([-1,1],
dy / \sqrt{1-y^2})$, the Hilbert space of square integrable functions with
respect to a measure.

Our filter coefficients can thus be approximated by the expansion
$$ c = \sum_{k=0}^{K-1} c^c_k T_k(\tilde{\lambda}), $$
where $c^c$ denotes a vector of Chebyshev coefficients, $K-1$ is the polynomial
order and $\tilde{\lambda} = 2\lambda/\lambda_{N-1}-1$ is a vector of scaled
eigenvalues. This approximation reduces the number of coefficients to learn
from $M$ to $K$.

The trick to avoid the Fourier basis is to express the polynomials $T_k$ as
functions of the scaled Laplacian $\tilde{\L} = 2\L/\lambda_{N-1}-I$. Note that
the spectrum of the normalized Laplacian is bounded by $2$, such that the
scaling can simply be $\hat{\L} = L - I$, tolerating some imprecision in the
approximation. The approximate filtering function is thus given by
$$ U\diag(c)U^T =
\sum_{k=0}^{K-1} U c^c_k T_k(\tilde{\lambda}) U^T =
\sum_{k=0}^{K-1} c^c_k T_k(\tilde{\L}). $$ {#eq:approximation}

Inserting [@eq:approximation] into [@eq:loss] we obtain
$$ L =
\frac{1}{N} \sum_{k=0}^{K-1} \| c^c_k T_k X - Y \|_F^2 =
\frac{1}{N} \| \bar{Y} c^c - \bar{y} \|_2^2 $$
where $\bar{y} \in \R^{MN}$ is the vectorized matrix $Y$ and the $k^\text{th}$
column of $\bar{Y} \in \R^{MN \times K}$ is the vectorized matrix $\hat{Y}_k =
T_k X$. The gradient is then given by
$$ \nabla_{c^c} L = \frac{2}{N} \bar{Y}^T (\bar{Y} c^c - \bar{y}). $$
The optimality condition $\bar{Y} c^c = \bar{y}$ is largely over-determined as
$K << MN$ but the least-square approximate solution is optimal.

Using the recurrence
$$ \hat{Y}_k = 2\tilde{\L} \hat{Y}_{k-1} - \hat{Y}_{k-2} $$
with $\hat{Y}_0 = X$ and $\hat{Y}_1 = \tilde{\L} X$, the computation of
$\bar{Y}$ from $X$ costs $O(K|E|N) = O(KMN)$ operations if the number of edges
$|E|$ is proportional to the number of nodes $M$, e.g. for kNN graphs. As the
cost of the product $\bar{Y}c^c$ is similar, the entire filtering operation has
a computational cost of $O(KMN)$ whereas the straightforward implementation
using the Fourier basis had a cost of $O(M^2\max(M,N)$. To further save
computations, at the expense of memory, one may store $\bar{Y}$ while applying
different filters, e.g. in the case of learning through SGD.

## Convolution of graph signals

which gives
$$a_j^{(l)'} = f(U_{l-1}^T \hat{H}_{l,i,j} U_{l-1} a_i^{(l-1)} + b_{l,i,j,\cdot})$$ {#eq:spectral_mult}
in the spectral domain. See [@eq:spectral_mult].

## Architecture

Image.

# Experiments

## Filter learning

Example from the Juypiter notebook.

Ground truth filter.
where $c_\ell = g(\lambda_\ell)$ is the evaluated filter.

(randomly
generated) graph signals and a filter $g(\lambda)$ for
some $t$, we want to learn the filter coefficients $C$, a diagonal matrix, such
that
$$\min_C \sum_{i=1}^N \| U^T C U x_i - y_i \|_2^2$$
where $U$ are the eigenvectors of the graph Laplacian $L$ and $y_i = x_i * g +
\varepsilon, \varepsilon \sim \mathcal{N}(0,\epsilon)$ is the target signal (the
filtered signal $x_i$ with additive Gaussian noise).

# Conclusion

That was awesome.

# References
