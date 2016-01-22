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
  \newcommand{\Xh}{\hat{X}}
  \newcommand{\Yh}{\hat{Y}}
  \newcommand{\st}{\ \text{s.t.} \,}
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

A graph $G = (V, E, W)$ is defined by a set $V$ of $|V| = M$ nodes and a set
$E$ of weighted edges. The connectivity of the graph is captured by the
adjacency matrix $W \in \R^{M \times M}$ which entry $W_{i,j}$ denotes the
weight of the edge $(u_i, v_i) \in E$ which connects the vertex $v_i \in V$ to
$v_j \in V$. The weight $W_{i,j} = 0$ if the vertices are not connected, i.e.
$(u_i, v_i) \notin E$. Assuming an undirected graph, $W$ is a symmetric matrix.
A graph signal is any signal $x \in \R^M$ defined on the vertices of $G$.

The combinatorial graph Laplacian is given by
$$ \L = D - W \in \R^{M \times M} $$
where $D \in \R^{M \times M}$ is the diagonal degree matrix defined as $D_{i,i}
= \sum_j W_{i,j}$. Note that it is a difference operator such that
$$ (\L x)_i = \sum_j W_{i,j} (x_i - x_j). $$
The normalized graph Laplacian is given by
$$ \L = I_M - D^{-1/2} W D^{-1/2} \in \R^{M \times M} $$
where $I_M \in \R^{M \times M}$ is the identity matrix.

In analogy to the real line Fourier Transform, a Fourier basis is given by the
eigenvectors of the Laplacian
$$ \L u_i = \lambda_i u_i \st \|u_i\|_2=1, \ i = 0, \ldots, M-1, $$
with their associated eigenvalues $\lambda_i$ [@shuman_emerging_2013;
@hammond_wavelets_2011]. Assuming the graph is connected, we may order the
vector $\lambda = [\lambda_0, \ldots, \lambda_{M-1}]^T \in \R^M$ of eigenvalues
such that
$$ 0 = \lambda_0 < \lambda_1 \leq \ldots \leq \lambda_{M-1} = \lambda_{max}. $$
As the Laplacian is a real symmetric matrix, the eigenvalues are real and the
eigenvectors orthonormal. The Laplacian is indeed diagonalized by the Fourier
basis $U = [u_0, \ldots, u_{M-1}] \in \R^{M \times M}$ such that
$$\L = U \diag(\lambda) U^T$$
where $\diag(\lambda)$ denotes a diagonal matrix of eigenvalues.

The Graph Fourier Transform $\hat{x} \in \R^M$ of any graph signal $x$ is
defined as
$$ \hat{x} = U^T x =
[\langle u_0, x \rangle, \ldots, \langle u_{M-1}, x \rangle]^T $$
while its inverse is given by
$$ x = U \hat{x}. $$

It follows that any signal $x$ can be filtered by an operator $g(\L) \in \R^{M
\times M}$ such that
$$ y = g(\L) x = U \diag(c) U^T x $$
where $c = g(\lambda) \in \R^M$ is a vector of filter coefficients.

## Graph filter learning

Given a graph $G$, a set $X = [x_0, \ldots, x_{N-1}] \in \R^{M \times N}$ of
$N$ source graph signals of dimensionality $M$ and their associated set $Y =
[y_0, \ldots, y_{N-1}] \in \R^{M \times N}$ of target signals, we want to learn
the coefficients of a graph filter $g$ such as to minimize the convex
reconstruction error
$$ L =
\frac{1}{N} \sum_{i=0}^{N-1} \| g(\L) x_i - y_i \|_2^2 =
\frac{1}{N} \| U \diag(c) U^T X - Y \|_F^2 $$ {#eq:loss}
where $\|\cdot\|_2^2$ denotes the squared $\ell_2$ norm and $\|\cdot\|_F^2$ the
squared Frobenius norm.

Note that the non-parametrization of the coefficients $c$ w.r.t. $\lambda$ does
omit all information about frequencies, while we know that the lower
frequencies are more important for clustering [ref NCut].

Rewriting [@eq:loss] in the spectral domain while decomposing it w.r.t. the
scalar coefficients $c_i$ gives
$$ L =
\frac{1}{N} \| \diag(c) U^T X - U^T Y \|_F^2 =
\frac{1}{N} \sum_{i=0}^{M-1} \|c_i\Xh_{i,\cdot} - \Yh_{i,\cdot} \|_2^2 $$
where $\Xh = U^TX \in \R^{M \times N}$ and $\Yh = U^TY \in \R^{M \times N}$ are
the spectral representations of the signals $X$ and $Y$. The gradient for each
coefficient is then given by
$$ \nabla_{c_i} L =
\frac{2}{N} ( c_i \Xh_{i,\cdot} - \Yh_{i,\cdot} ) \Xh^T_{\cdot,i} $$
and can be rewritten in a vectorized form as
$$ \nabla_{c} L =
\frac{2}{N} \diag \left( (\diag(c) \Xh - \Yh) \Xh^T \right) =
\frac{2}{N} \left( \Xh \odot ( c1_N^T \odot \Xh - \Yh ) \right) 1_N
$$ {#eq:gradient}
where $1_N$ denotes a unit vector of length $N$ and $\odot$ the element-wise
Hadamard product. The second form avoids the computation of the useless
off-diagonal elements.

A direct solution is given by the optimality condition $\nabla_{c}L=0$ such that
$$ c^o = \argmin_c L =
(\Xh \odot \Yh) 1_N \oslash (\Xh \odot \Xh) 1_N $$
where $\oslash$ denotes an element-wise division. Note that this method is
impractical for large $M$ and $N$ (sufficiently large for $X$ and $Y$ to not
fit in memory). A Stochastic Gradient Descent based on [@eq:gradient] will do
the trick.

There is however two major computational drawbacks to this optimization
process: (1) the Fourier transform $\Xh=U^TX$ of a set of signals costs
$O(M^2N)$ operations and (2) the eigenvalue decomposition
$\L=U\diag(\lambda)U^T$ costs $O(M^3)$. The total cost of filtering is thus
$O(M^2 \max(M,N))$, a problem already stated in [@henaff_deep_2015].

### Fast algorithm

The idea is to approximate the filter (in the spectral domain) by a truncated
Chebyshev expansion and to recursively compute the Chebyshev polynomials from
the Laplacian, avoiding the Fourier basis altogether [@hammond_wavelets_2011]. 

Recall that the Chebyshev polynomial $T_k(x)$ of order $k$ may be generated by
the stable recurrence relation $T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$ with $T_0
= 1$ and $T_1 = x$. These polynomials form an orthogonal basis for $L^2([-1,1],
dy / \sqrt{1-y^2})$, the Hilbert space of square integrable functions with
respect to a measure.

The graph filter coefficients can thus be approximated by the expansion
$$ c = g(\lambda) \approx \sum_{k=0}^{K-1} c^c_k T_k(\tilde{\lambda}) $$
of polynomial order $K-1$, where $c^c \in \R^K$ denotes a vector of Chebyshev
coefficients and $T_k(\tilde{\lambda}) \in \R^M$ is the Chebyshev polynomial of
order $k$ evaluated at $\tilde{\lambda} = 2\lambda/\lambda_{max}-1 \in \R^M$,
a vector of scaled eigenvalues. This approximation reduces the number of
learned coefficients from $M$ to $K$.

The trick to avoid the Fourier basis is to express the polynomials as functions
of the Laplacian, such that an approximation of the filtering operator is given
by
$$ g(\L) = U\diag(c)U^T \approx
\sum_{k=0}^{K-1} U \diag \left( c^c_k T_k(\tilde{\lambda}) \right) U^T =
\sum_{k=0}^{K-1} c^c_k T_k(\tilde{\L}) $$ {#eq:approximation}
where $T_k(\tilde{\L}) \in \R^{M \times M}$ is the Chebyshev polynomial of
order $k$ evaluated at the scaled Laplacian $\tilde{\L} = 2\L/\lambda_{max}-I$.
Note that the spectrum of the normalized Laplacian is bounded by $2$, such that
the scaling can simply be $\tilde{\L} = \L - I$, tolerating some imprecision in
the approximation.

Inserting [@eq:approximation] into [@eq:loss] we obtain
$$ L =
\frac{1}{N} \| \sum_{k=0}^{K-1} c^c_k T_k(\tilde{\L}) X - Y \|_F^2 =
\frac{1}{N} \| \bar{X} c^c - \bar{y} \|_2^2 $$
where $\bar{y} \in \R^{MN}$ is the vectorized matrix $Y$ and the $k^\text{th}$
column of $\bar{X} \in \R^{MN \times K}$ is the vectorized matrix $\tilde{X}_k
= T_k(\tilde{\L}) X \in \R^{M \times N}$. The gradient is then given by
$$ \nabla_{c^c} L = \frac{2}{N} \bar{X}^T (\bar{X} c^c - \bar{y}). $$
The optimality condition $\bar{X} c^c = \bar{y}$ is largely over-determined as
$K << MN$ but a least-square solver can find an approximate solution.

Using the recurrence
$$ \tilde{X}_k = 2\tilde{\L} \tilde{X}_{k-1} - \tilde{X}_{k-2} $$
with $\tilde{X}_0 = X$ and $\tilde{X}_1 = \tilde{\L} X$, the computation of
$\bar{X}$ given $X$ costs $O(K|E|N) = O(KMN)$ operations if the number of edges
$|E|$ is proportional to the number of nodes $M$, e.g. for kNN graphs. As the
cost of the product $\bar{X}c^c$ is similar, the entire filtering operation has
a computational cost of $O(KMN)$ whereas the straightforward implementation
using the Fourier basis had a cost of $O(M^2\max(M,N)$. To further save
computations, at the expense of memory, one may store $\bar{X}$ while applying
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
