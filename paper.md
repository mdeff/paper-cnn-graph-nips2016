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
