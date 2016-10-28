---
title: 'Convolutional Neural Networks for Graphs'
subtitle: 'Graph Neural Networks'
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
abstract: This paper introduces Graph Neural Network (GNN), a novel framework
  to process data on irregular, i.e. non-Euclidean domains modelized as graphs.
  Even recent developments in Deep Learning who use graph embeddings don't make
  use of the graph structure per se.
date: \today
header-includes:
- \usepackage[acronym]{glossaries}
- \newacronym{SVD}{SVD}{Singular Value Decomposition}
- \newacronym{SGD}{SGD}{Stochastic Gradient Descent}
- \newacronym{PSD}{PSD}{positive semidefinite}
- \DeclareMathOperator*{\diag}{diag}
- \DeclareMathOperator*{\argmin}{arg\,min}
- \DeclareMathOperator*{\spn}{span}
- \renewcommand{\L}{\mathcal{L}}
- \renewcommand{\G}{\mathcal{G}}
- \renewcommand{\H}{\mathcal{H}}
- \newcommand{\K}{\mathcal{K}}
- \newcommand{\V}{\mathcal{V}}
- \newcommand{\E}{\mathcal{E}}
- \newcommand{\W}{\mathcal{W}}
- \newcommand{\R}{\mathbb{R}}
- \newcommand{\N}{\mathbb{N}}
- \newcommand{\Xh}{\hat{X}}
- \newcommand{\Yh}{\hat{Y}}
- \newcommand{\st}{\ \text{s.t.} \,}
- '\newcommand{\norm}[1]{\left\| #1 \right\|}'
---

# Introduction

Large networks allow to learn complex relationships. Although they overfit
easily if the training set is small because there is a lot of parameters to
learn.

Convolutional neural networks (CNN) offer a great reduction of parameters on
regular Euclidean domain. This work proposes to generalize it to the discrete
model of manifolds, graphs. Such an approach was proposed by
[@bruna_spectral_2013; @henaff_deep_2015], we will refine it with proper graph
signal processing tools.

This paper introduces a framework to use CNN with signals defined on weighted
graphs. Our work leverages spectral methods, which are much faster than
a spatial approach would be, in the same sense as doing convolutions as
multiplication in the Fourier domain (with the help of the FFT) is much faster
than multiplying patches on an image.

Another option to do clustering or regression on data graphs is to first embed
the graph and then use the embedding as additional features.

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

# Model

## Spectral Graph Theory

A graph $\G = (\V, \E, \W)$ is defined by a set $V$ of $|V| = M$ nodes and a set
$E$ of weighted edges. The connectivity of the graph is captured by the
adjacency matrix $W \in \R^{M \times M}$ which entry $W_{i,j}$ denotes the
weight of the edge $(v_i, v_j) \in E$ which connects the vertex $v_i \in V$ to
$v_j \in V$. It is set to $0$ if the vertices are not connected, i.e. $(v_i,
v_j) \notin E$. Assuming an undirected graph, $W$ is a symmetric matrix.
A graph signal is any signal $x \in \R^M$ defined on the vertices of $G$.

The combinatorial graph Laplacian is defined as
$$ \L^c := D - W \in \R^{M \times M}, $$
where $D \in \R^{M \times M}$ is the diagonal degree matrix defined as
$D_{i,i} := \sum_j W_{i,j}$. Note that it is a difference operator such that
$$ (\L^c x)_i = \sum_j W_{i,j} (x_i - x_j). $$
The normalized graph Laplacian is defined as
$$ \L^n := I_M - D^{-1/2} W D^{-1/2} \in \R^{M \times M}, $$
where $I_M \in \R^{M \times M}$ is the identity matrix. Finally, the
random-walk graph Laplacian is defined as
$$ \L^{rw} := I_M - D^{-1} W \in \R^{M \times M}. $$
Note that this work is independent of the chosen graph Laplacian $\L$, which
can be any of $\L^c$, $\L^n$ or $\L^{rw}$.

In analogy to the real line Fourier Transform, a Fourier basis is given by the
eigenvectors of the Laplacian
$$ \L u_i = \lambda_i u_i \st \norm{u_i}_2=1,
\ u_i \in \R^M, \ i = 0, \ldots, M-1, $$
with their associated eigenvalues $\lambda_i$. Assuming the graph is connected,
we may order the vector $\lambda := [\lambda_0, \ldots, \lambda_{M-1}]^T \in
\R^M$ of eigenvalues such that
$$ 0 = \lambda_{min} = \lambda_0 < \lambda_1 \leq \ldots \leq
\lambda_{M-1} = \lambda_{max}. $$
As the Laplacian is a real symmetric and \gls{PSD} matrix, the eigenvalues are
real and positive, and the eigenvectors are orthonormal. The Laplacian is
indeed diagonalized by the Fourier basis $U := [u_0, \ldots, u_{M-1}] \in \R^{M
\times M}$ such that
$$ \L = U \Lambda U^T, $$ {#eq:lap_diag}
where $\Lambda := \diag(\lambda) \in \R^{M \times M}$ denotes a diagonal matrix
of eigenvalues and $U^T$ is the matrix transpose of $U$. See
[@chung_spectral_1997] for details on spectral graph theory.

The Graph Fourier Transform $\hat{x} \in \R^M$ of any graph signal $x$ is
given by
$$ \hat{x} = U^T x =
[\langle u_0, x \rangle, \ldots, \langle u_{M-1}, x \rangle]^T, $$
where $\langle \cdot , \cdot \rangle$ denotes the standard $\ell_2$ inner
product [@shuman_emerging_2013]. The inverse transform is then given by $$
x = U \hat{x}. $$

It follows that any signal $\hat{x}$ can be filtered in the spectral domain by
$$ \hat{y} = g_\theta(\Lambda) \hat{x}, $$
where the operator $g_\theta(\Lambda) \in \R^{M \times M}$, a matrix function,
yields a diagonal matrix of $M$ Fourier coefficients parametrized by $\theta$.
The filtering operation can equivalently take place in the vertex domain as
$$ y = U g_\theta(\Lambda) U^T x = g_\theta(\L) x $$
where the operator $g_\theta(\L) = g_\theta(U \Lambda U^T)
= U g_\theta(\Lambda) U^T \in \R^{M \times M}$ is akin to a convolution with
the filter $Ug_\theta(\Lambda)$ in the vertex domain. Note that the filtering
operation is defined as a multiplication in the spectral domain because
a convolution cannot be defined in the vertex domain [@shuman_emerging_2013].

## Graph Filter Learning

Given a graph $G$, a set $X = [x_0, \ldots, x_{N-1}] \in \R^{M \times N}$ of
$N$ source graph signals of dimensionality $M$ and their associated set $Y
= [y_0, \ldots, y_{N-1}] \in \R^{M \times N}$ of target signals, we want to
learn the parameters $\theta \in \R^M$ of a graph filter $g_\theta(\Lambda) :=
\diag(\theta)$ such as to minimize the convex mean square error
$$ L =
\frac{1}{N} \sum_{i=0}^{N-1} \norm{ g_\theta(\L) x_i - y_i }_2^2 =
\frac{1}{N} \norm{ U \diag(\theta) U^T X - Y }_F^2, $$ {#eq:loss}
where $\norm{\cdot}_2^2$ denotes the squared $\ell_2$ norm and
$\norm{\cdot}_F^2$ the squared Frobenius norm.

Note that, while being the most flexible definition of $g_\theta(\Lambda)$, the
independence of the Fourier coefficients $\theta$ from the eigenvalues
$\Lambda$ does omit all information about frequencies. As we know that the
lower frequencies are more important for clustering [ref NCut], we could have
designed a parametric filter with less parameters.

Rewriting [@eq:loss] in the spectral domain while decomposing it w.r.t. the
scalar coefficients $\theta_i$ gives
$$ L =
\frac{1}{N} \norm{ \diag(\theta) U^T X - U^T Y }_F^2 =
\frac{1}{N} \sum_{i=0}^{M-1} \norm{\theta_i\Xh_{i,\cdot}-\Yh_{i,\cdot}}_2^2, $$
where $\Xh = U^TX \in \R^{M \times N}$ and $\Yh = U^TY \in \R^{M \times N}$ are
the spectral representations of the signals $X$ and $Y$. The gradient for each
coefficient is then given by
$$ \nabla_{\theta_i} L =
\frac{2}{N} ( \theta_i \Xh_{i,\cdot} - \Yh_{i,\cdot} ) \Xh^T_{\cdot,i} $$
and can be rewritten in a vectorized form as
$$ \nabla_{\theta} L =
\frac{2}{N} \diag \left( (\diag(\theta) \Xh - \Yh) \Xh^T \right) =
\frac{2}{N} \left( ( \theta 1_N^T \odot \Xh - \Yh ) \odot \Xh \right) 1_N,
$$ {#eq:gradient}
where $1_N$ denotes a unit vector of length $N$ and $\odot$ the element-wise
Hadamard product. The second form avoids the computation of the useless
off-diagonal elements.

A closed-form solution is given by the optimality condition
$\nabla_{\theta}L=0$ such that
$$ \theta^* = \argmin_\theta L =
(\Xh \odot \Yh) 1_N \oslash (\Xh \odot \Xh) 1_N, $$ {#eq:direct}
where $\oslash$ denotes an element-wise division. Note that this method is
impractical for large $M$ and $N$ (sufficiently large for $X$ and $Y$ to not
fit in memory). A \gls{SGD} based on [@eq:gradient] is then necessary.

There are however two major computational drawbacks to this optimization
process: 1) the Fourier transform $\Xh=U^TX$ of a set of signals costs
$O(M^2N)$ operations and 2) the eigenvalue decomposition
$\L = U \Lambda U^T$ costs $O(M^3)$. The total cost of filtering is thus $O(M^2
\max(M,N))$, a problem already stated in [@henaff_deep_2015]. We propose to
overcome these computational issues using efficient numerical approximations
such as the Chebyshev polynomials or the Lanczos algorithm.

### Chebyshev Parametrization

The main idea, to avoid the Fourier basis, is twofold: 1) parametrize the
filter, in the spectral domain, as a truncated Chebyshev expansion and 2)
recursively compute the Chebyshev polynomials from the Laplacian. This
approximate method for spectral graph filtering was first proposed in
[@hammond_wavelets_2011] for a fast wavelet transform on graphs.

Recall that the Chebyshev polynomial $T_k(x)$ of order $k$ may be generated by
the stable recurrence relation $T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$ with $T_0
= 1$ and $T_1 = x$. These polynomials form an orthogonal basis for $L^2([-1,1],
dy / \sqrt{1-y^2})$, the Hilbert space of square integrable functions with
respect to the measure $dy/\sqrt{1-y^2}$.

The graph filter $g_\theta(\Lambda)$ can thus be constructed from the truncated
expansion
$$ g_\theta(\Lambda) := \sum_{k=0}^{K-1} \theta_k T_k(\tilde{\Lambda}) $$
of polynomial order $K-1$, where the parameter $\theta \in \R^K$, $K \ll M$, is a vector of
Chebyshev coefficients and $T_k(\tilde{\Lambda}) \in \R^{M \times M}$ is the
Chebyshev polynomial of order $k$ evaluated at $\tilde{\Lambda} := 2 \Lambda
/ \lambda_{max} - I_M \in \R^{M \times M}$, a diagonal matrix of scaled
eigenvalues (so that they lie in $[-1,1]$). While reducing the number of
parameters from $M$ to $K$, this parametrization enforces smoothness in the
spectral domain, which translates to localization in the vertex domain. It can
indeed be shown that $(\L^k)_{i,j}=0$ if the shortest-path between vertices
$v_i$ and $v_j$ is longer than $k$ edges [@hammond_wavelets_2011], which limits
the influence of a $K^\text{th}$ order filter to $K$ hopes. This is often
a desired property, e.g. to learn local features for classification.

To avoid the Fourier basis $U$, we express the polynomials as functions of the
Laplacian $\L$ instead of its eigenvalues $\Lambda$ using [@eq:lap_diag], such
that the filtering operator in the vertex domain is given by
$$ g_\theta(\L) = U g_\theta(\Lambda) U^T =
\sum_{k=0}^{K-1} U \theta_k T_k(\tilde{\Lambda}) U^T =
\sum_{k=0}^{K-1} \theta_k T_k(\tilde{\L}), $$ {#eq:chebyshev}
where $T_k(\tilde{\L}) \in \R^{M \times M}$ is the Chebyshev polynomial of
order $k$ evaluated at the scaled Laplacian $\tilde{\L} := 2 \L / \lambda_{max}
- I_M$. Note that the spectrum of the normalized Laplacian is bounded by $2$
[@chung_spectral_1997], such that the scaling can simply be $\tilde{\L} = \L
- I_M$, tolerating some imprecision in the approximation due to the loss of
a fraction $2-\lambda_{max}$ of the domain.

Inserting [@eq:chebyshev] into [@eq:loss] we obtain
$$ L =
\frac{1}{N} \norm{ \sum_{k=0}^{K-1} \theta_k T_k(\tilde{\L}) X - Y }_F^2 =
\frac{1}{N} \norm{ \bar{X} \theta - \bar{y} }_2^2, $$ {#eq:loss_c}
where $\bar{y} \in \R^{MN}$ is the vectorized matrix $Y$ and the $k^\text{th}$
column of $\bar{X} \in \R^{MN \times K}$ is the vectorized matrix $\tilde{X}_k
:= T_k(\tilde{\L}) X \in \R^{M \times N}$. The gradient is then given by
$$ \nabla_\theta L =
\frac{2}{N} \bar{X}^T (\bar{X} \theta - \bar{y}). $$ {#eq:gradient_c}

While the system $\bar{X} \theta - \bar{y}$ is largely over-determined as $K
\ll MN$, a closed-form solution of [@eq:loss_c] is given by the optimality
condition $\nabla_\theta L = 0$ so that
$$ \theta^* = \argmin_\theta L = \bar{X}^+ \bar{y}, $$ {#eq:direct_c}
where $\bar{X}^+ = (\bar{X}^T\bar{X})^{-1} \bar{X}^T$ is the pseudo-inverse of
$\bar{X}$. The computation of the inverse is fast and stable for small $K$.
Alternative approximate solution methods for larger order $K$ include the
computation of the pseudo-inverse with \gls{SVD}, the use of a least-square
solver or a gradient descent scheme based on [@eq:gradient_c].

Using the recurrence
$$ \tilde{X}_k = 2\tilde{\L} \tilde{X}_{k-1} - \tilde{X}_{k-2} $$
with $\tilde{X}_0 = X$ and $\tilde{X}_1 = \tilde{\L} X$, the computation of
$\bar{X}$ given $X$ costs $O(K|E|N) = O(KMN)$ operations if the number of edges
$|E|$ is proportional to the number of nodes $M$, e.g. for kNN graphs. As the
cost of the product $\bar{X}\theta$ is similar, the entire filtering operation
has a computational cost of $O(KMN)$ whereas the straightforward implementation
using the Fourier basis has a cost of $O(M^2\max(M,N)$. When applying multiple
filters to the same set of signals, as is the case with \gls{SGD}, one may save
computations by storing and querying $\bar{X}$ instead of $X$, at the expense
of $K$ times the memory.

### Lanczos Parametrization

Another applicable parametrization is based on the Lanczos algorithm, an
adaptation of the power iteration to find the largest or smallest eigenvalues
and corresponding eigenvectors of a linear system. It was first introduced for
fast graph filtering in [@susnjara_accelerated_2015].

The algorithm, described in [@gallopoulos_efficient_1992;
@susnjara_accelerated_2015], constructs an orthonormal basis $V = [v_0, \ldots,
v_{K-1}] \in \R^{M \times K}$ of the Krylov subspace $\mathcal{K}_K(\L,x)
= \spn\{ x, \L x, \ldots, \L^{K-1} x \}$ and a tri-diagonal matrix $H = V^T \L
V \in \R^{K \times K}$ with a computational cost of $O(K |E|)$. Note that for
large $K \gtrapprox 30$, the original iterative algorithm may loose the basis
orthogonality such that a necessary orthogonalization step will increase the
complexity [@susnjara_accelerated_2015]. Filtering the signal $x$ with
$g_\theta(\L)$ can then be approximated by an order $K-1$ polynomial as
$$ y = g_\theta(\L) x \approx
V g_\theta(H) V^T x = V Q g_\theta(\Sigma) Q^T V^T x $$ {#eq:lanczos}
where $Q \Sigma Q^T = H$ is the eigendecomposition of $H$. There exist fast
methods for the eigendecomposition of symmetric tri-diagonal matrices []. As
for the Chebyshev approximation, this construction enforces smoothness in the
spectral domain.

Inserting [@eq:lanczos] with a parametrized filter $g_\theta(\Sigma) :=
\diag(\theta)$, $\theta \in \R^K$, $K \ll M$, into [@eq:loss] gives
$$ L = \frac{1}{N} \sum_{n=0}^{N-1}
\norm{V_n Q_n \diag(\theta) Q_n^T V_n^T x_n - y_n}_2^2 =
\frac{1}{N} \norm{\diag(\theta) \hat{X} - \hat{Y}}_F^2 $$
where $\hat{X} = [\hat{x}_0, \ldots, \hat{x}_{N-1}] \in \R^{K \times N}$,
$\hat{Y} = [\hat{y}_0, \ldots, \hat{y}_{N-1}] \in \R^{K \times N}$ and
$\hat{x}_n = Q_n^T V_n^T x_n \in \R^K$, $\hat{y}_n = Q_n^T V_n^T y_n \in \R^K$
are approximate representations of the signals $x_n$, $y_n$ in the orthonormal
basis $V_n Q_n \in \R^{M \times K}$ of $\mathcal{K}_K(\L,x_n)$. Similarly to
[@eq:loss], the gradient is given by [@eq:gradient] and a closed-form solution
by [@eq:direct].

The expression [@eq:lanczos] can be simplified by setting $v_0 :=
x / \norm{x}_2$ (the first basis vector can be set to an arbitrary unit length
vector) such that $V^T x = \norm{x}_2 e_1$ and
$$ V g_\theta(H) V^T x =
\norm{x}_2 V Q \diag(\theta) Q^T e_1 =
\norm{x}_2 V Q \diag(q) \theta $$
where $e_1 \in \R^K$ denotes the first unit vector and $q = Q^T e_1 \in \R^K$
is the first row of $Q$. Inserting into [@eq:loss] gives
$$ L =
\frac{1}{N} \sum_{n=0}^{N-1} \norm{\tilde{X}_n \theta - y_n}_2^2 =
\frac{1}{N} \sum_{k=0}^{K-1} \norm{\bar{X}_{\cdot,k} \theta_k - \bar{y}}_2^2 =
\frac{1}{N} \norm{\bar{X} \theta - \bar{y} }_2^2 $$
where $\bar{y} \in \R^{NM}$ is the vectorized matrix $Y$ and $\bar{X} :=
[\tilde{X}_0, \ldots, \tilde{X}_{N-1}]^T \in \R^{NM \times K}$ is a stack of
$N$ matrices $\tilde{X}_n := \norm{x_n}_2 V_n Q_n \diag(q_n) \in \R^{M \times
K}$ where $V_n$, $Q_n$ and $q_n$ are derived from $x_n$. The second form (in
terms of independent coefficients $\theta_k$) is valid because $\bar{X}$ is
orthogonal, i.e $\bar{X}^T \bar{X}$ is a diagonal matrix, so that the solution
[@eq:direct_c] can be written as
$$ \theta^* = \argmin_\theta L =
\underbrace{\sum_{n=0}^{N-1} \Big( \norm{x_n}_2 \diag(q_n) \Big)^{-2}}
_{(\bar{X}^T\bar{X})^{-1}} \bar{X}^T \bar{y} $$ {#eq:direct_l}
where $\bar{X}^T \bar{y} = [\langle \bar{X}_{\cdot,0}, \bar{y} \rangle, \ldots,
\langle \bar{X}_{\cdot,K-1}, \bar{y} \rangle]^T \in \R^K$ is the projection of
$\bar{y}$ onto the $K$ basis vectors $\bar{X}_{\cdot,k} \in \R^{MN}$. While
this closed-form evaluation is fast and stable, a gradient descent scheme can
be used with [@eq:gradient_c].

While the time and space complexities are similar, this method has two
advantages over the Chebyshev approximation: 1) it does not require the
normalization of the Laplacian spectrum (thus the estimation of
$\lambda_{max}$) and 2) the optimization is easier as the parameters $\theta_k$
are independent of each other thanks to the orthogonal basis. By easier we mean
that 1) the closed-form solution [@eq:direct_l] does not involve the inversion
of a large matrix and 2) \gls{SGD} converges to a better solution with less
iterations. See [@susnjara_accelerated_2015] for a discussion of the
approximation quality and a comparison with the Chebyshev approximation.

## Graph coarsening

The graph partitioning problem has been widely studied. Applications include
VLSI design, load balancing for parallel computations, network analysis, and
optimal scheduling. The goal is to partition the vertices of a graph into
a certain number of disjoint sets of approximately the same size so that a cut
metric is minimized. This problem is NP-complete even for several restricted
classes of graphs, and there is no constant factor approximation algorithm for
general graphs [Bui and Jones 1992]. While notable developments in exact
algorithms and heuristics have been done, only the introduction of the
general-purpose multilevel methods during the 1990s has provided a breakthrough
in efficiency and quality [Safro 2009 2012]. Graph coarsening, where the problem
instance is gradually mapped to smaller ones to reduce the original complexity,
i.e., the graph underlying the problem is compressed, is the first step of
these multilevel methods.

Existing multilevel algorithms for combinatorial optimization problems (such as
k-partitioning, linear ordering, clustering, and segmentation) can be divided
into two classes: contraction-based schemes and algebraic multigrid
(AMG)-inspired schemes.

Alternative standard algorithms for graph coarsening include the Kron reduction
[], Label Propagation [], spectral clustering [] or a sampling [Nath
uncertainty]. A standard approach for graph coarsening is the maximal match, as
used by METIS for graph partitioning [?].

The assumption behind METIS is that a (near) optimal partitioning on a coarser
graph implies a good partitioning in the finer graph. In general, it only holds
true when the degree of nodes in the graph is bounded by a constant [Karypis
95]. Some real-life graphs, e.g. social networks or WWW, are right-skewed, i.e.
there exists hub vertices which have very large degrees. These graphs follow
a power-law degree distribution.

# Applications

## Regression

Solving a linear regression problem on graph signals is akin to minimize
[@eq:loss] where $Y$ is the dependent variable and $X$ the independent
observations.

## Representation learning

Representation learning is the problem of finding an appropriate (to some
criteria) representation $y_i$ of a sample $x_i$ defined as
$$ \min_{A,Y} \norm{ AX - Y }_F^2 + \tau P(Y) $$
where $A \in \R^{M \times M}$ is some operator and $P(Y)$ is some prior on $Y$.
Take $P(Z)=\norm{Z}_F^2$ and you get a Thikhonov prior. Take $P(Y)
= \norm{Y}_1$ and you get sparse coding (with dictionary learning).

## Classification

There are dependent vectors $X = \{x_i\}_{i=0}^{N-1} \in \R^{M \times N}$ we
want to predict from observations $Y = \{y_i\}_{i=0}^{N-1} \in \R^{M \times
N}$.

In the classification case, $y_i \in \N$ is a discrete number which represents
a class, e.g. $y_i=-1$ indicates that sample $x_i$ belongs to class 1 and
$y_i=+1$ to class 2.

### Representation learning

Three ways to reduce the number of parameters:

1. Graph coarsening
2. No weights: loss of spatiality (convolutional only)
3. Non-linear activation function

But $\theta$ and $w$ are redundant, so
$$ L = \norm{\overline{\bar{X} \theta}^T 1_{FM} - y}_2^2 $$
$$ \bar{X} \in \R^{NM \times K} $$
$$ \theta \in \R^{K \times F} $$

$$ \nabla_\theta L = \bar{X}^T (\bar{X} \theta - z) $$
$$ \nabla_\theta L = (\overline{\bar{X} \theta}^T 1_{FM} - y) $$

$$ \bar{X} 1_{FM} (\overline{\bar{X} \theta}^T 1_{FM} - y) $$
$$ (\bar{X} 1_{FM} \overline{\bar{X}}^T 1_{FM})^{-1} \bar{X} 1_{FM} y $$

### Linear classification

Empirical risk minimization (statistical learning).

The $\ell_2$ regularized least-square model, also known as ridge regression,
is defined as
$$ L =
\frac{1}{N} \sum_{i=0}^{N-1} |f_w(x_i) - y_i|^2 + \tau \norm{f}_\H^2,
$$ {#eq:rls}
where $f_w \in \H$ is some classification function parametrized by $w$ and $\H$
is an Hilbert space.

Introducing the linear classifier $f(x) = w^T x$ into [@eq:rls] gives
$$ L = \frac{1}{N} \norm{X^T w - y}_2^2  + \tau \norm{w}_2^2, $$
where $w \in \R^M$ are the weights to learn. Note that there is no bias because
the data can be centered around zero (we know the lower and upper bounds on
luminosity values).

Gradient
$$ \nabla_w L = \frac{2}{N} X (X^T w - y) + 2 \tau w $$ {#eq:gradient_rls}

Optimality condition $\nabla L = 0$
$$ w^* = \argmin_w \nabla L =
\left(X X^T + \tau N I_M \right)^{-1} X y $$ {#eq:sol_rls}

### Feature learning

Introducing graph filter learning as a scheme for feature learning gives the
objective
$$ L = \frac{1}{N} \sum_{i=0}^{N-1}
\Big| f_w \Big( A_\theta(\L) x_i \Big) - y_i \Big|^2 +
\tau_R \norm{f_w}_\H^2, $$
where the filter bank $A_\theta(\L) = \bigcup_{i=0}^{F-1} g_{\theta_i}(\L) \in
\R^{FM \times M}$ is a block-diagonal matrix composed of $F$ filters
$g_{\theta_i}(\L)$ which extract different features.

Using a linear classifier and the Lanczos parametrization introduced in
\ref{lanczos-parametrization} gives
$$ L =
\frac{1}{N} \norm{\overline{\bar{X} \theta}^T w - y}_2^2 +
\tau_R \norm{w}_2^2, $$ {#eq:rls_gfl}
where $w \in \R^{FM}$ are the weights to learn, $\bar{X} \in \R^{NM
\times K}$ is the Lanczos basis described in \ref{lanczos-parametrization},
$\theta \in \R^{K \times F}$ are the $K$ coefficients of the $F$ filters and
$\overline{\bar{X} \theta} \in \R^{FM \times N}$ is a rearrangement of the
matrix $\bar{X} \theta \in \R^{NM \times F}$. Note that there is now $F$ times
more weigts to learn, i.e. $w \in \R^{FM}$.

$\bar{X} \theta \in \R^{NM \times F}$

$$ \frac{2}{N} \bar{X} w (\overline{\bar{X} \theta}^T w - y) $$

$$ (\bar{X} w \overline{\bar{X}}^T w)^{-1} \bar{X}wy $$

### Splitting scheme

The splitting scheme
$$ L =
\frac{1}{N} \norm{Z^T w - y}_2^2 +
\frac{\tau_F}{N} \norm{\overline{\bar{X} \theta} - Z}_F^2 +
\tau_R \norm{w}_2^2 $$
may be used to minimize the non-convex objective [@eq:rls_gfl] for $\theta$ and
$w$, where $Z = [z_0, \ldots, z_{N-1}] \in \R^{FM \times N}$ are the signals in
feature space. Solving the overall non-convex problem requires to iteratively
solve the following three convex sub-problems:

1. The gradient and solution to $\min_\theta L$ are given by [@eq:gradient_c]
and [@eq:direct_l], substituting $\bar{y}$ by $\bar{Z} \in \R^{NM \times F}$,
a rearrangement of $Z$.

2. The gradient of $\min_Z L$ is
$$ \nabla_Z L =
\frac{2}{N} w (w^T Z - y^T) + \frac{2\tau_F}{N} (Z - \overline{\bar{X} \theta}) $$
and the closed-form solution is
$$ Z^* = \argmin_Z L =
\left( \tau_F I_{FM} + w w^T \right)^{-1}
\left( \tau_F \overline{\bar{X} \theta} + w y^T \right). $$

3. The gradient and solution to $\min_w L$ are given by [@eq:gradient_rls]
and [@eq:sol_rls], substituting $X$ by $Z$.

### Graph coarsening

To reduce again the number of parameters in $w$, the graph should be coarsened
to trade feature resolution at the price of vertex resolution.

### Kernel methods

The same can be done for kernel methods.

We want to learn a classification function $f(x) = \sum_i \K(x,x_i) \alpha_i
\in \R$, where $\K(\cdot,\cdot)$ is some kernel function, which maps a sample
$x$ to a label $y$. The loss function to minimize over the training set is then
given by
$$ L = \sum_i |f(x_i) - y_i|^2 + \tau \norm{f}_{H_\K}^2 =
\norm{ \K \alpha - y }_2^2 + \tau \alpha^T \K \alpha, $$
where $\K_{i,j} = \K(x_i, x_j)$ is the kernel function evaluated at all
training samples and $\alpha \in \R^N$ are the parameters.

$$ \nabla_\alpha L = \K (\K \alpha - y) + \tau \K \alpha $$
$\K$ is symmetric.

$$ \alpha^* = \argmin_\alpha L = (\K + \tau I_N)^{-1} y $$

### Feature learning

A general formulation of the loss for a classification problem is given by
$$ \min_{\theta_A, \theta_h} \left\{ L = \frac{1}{N} \sum_{i=0}^{N-1}
\left| h_{\theta_h}(A_{\theta_A}(\L) x_i) - y_i \right|^2 +
\tau_A \Omega_A(A_{\theta_A}) + \tau_h \Omega_h(h_{\theta_h}) \right\} $$
where $A$ is an operator which transforms the input signals into a suitable,
i.e. learned, representation and $h$ is a classification function. $\Omega_A$
and $\Omega_h$ are regularizations, $\tau_A$ and $\tau_h$ are hyper-parameters.

The SVM classification function $h \in H_\K$ is defined as
$$ h_{\theta_h}(x) = \sum_{i=0}^{N-1} \alpha_i \K(x, x_i) + \beta,
\ \theta_h = (\alpha_i, \beta), $$
where $\K(\cdot,\cdot)$ is some kernel function. It is regularized by
$$ \Omega_h = \norm{h}_{H_\K}^2. $$

In our graph filters learning setting, the mapping $A \in \R^{FM \times M}$ is
given by the filter bank
$$ A_{\theta_A}(\L) = \bigcup_{f=0}^{F-1} g_{\theta_f}(\L),
\ \theta_A = (\theta_f), $$
a block-diagonal matrix composed of $F$ filters, which extracts multiple
features from the input signal.

While the optimization complexity would increase a lot if we were to minimize
for the model complexity $K$, we can easily include a penalty of the form
$$ \Omega_A = \sum_f \sum_i |\theta_f(\lambda_i) w(\lambda_i)|^2 $$
where $w(\lambda_i) \sim 1 / \lambda_i$.

Learning complexity: there is $FK$ parameters in $\theta_A$ and $N+1$
parameters in $\theta_h$.

## Going deep

Define convolutional neural networks on irregular domains by learning multiple
layers of graph filters.

Image of the architecture.

# Experiments

## Filter learning

Example from the Juypiter notebook.

Ground truth filter.
where $c_\ell = g(\lambda_\ell)$ is the evaluated filter.

(randomly
generated) graph signals and a filter $g(\lambda)$ for
some $t$, we want to learn the filter coefficients $C$, a diagonal matrix, such
that
$$\min_C \sum_{i=1}^N \norm{ U^T C U x_i - y_i }_2^2$$
where $U$ are the eigenvectors of the graph Laplacian $L$ and $y_i = x_i * g +
\varepsilon, \varepsilon \sim \mathcal{N}(0,\epsilon)$ is the target signal (the
filtered signal $x_i$ with additive Gaussian noise).

### Non-parametrized filter

### Spline Parametrization

The authors of [@bruna_spectral_2013; @henaff_deep_2015] learned graph filters
parametrized by cubic splines.

### Chebyshev Parametrization

### Lanczos Parametrization

## Classification

### Sampled MNIST

### Text documents

# Discussion

# Conclusion and Outlook

That was awesome.

# References
