% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/XVAE_utils.R
\name{wendland}
\alias{wendland}
\title{Wendland Radial Basis Function}
\usage{
wendland(d, r)
}
\arguments{
\item{d}{A vector or matrix of nonnegative distances.}

\item{r}{A nonnegative scalar representing the radius of influence.}
}
\value{
A vector or matrix of Wendland function values, where the function is zero for distances greater than \code{r}.
}
\description{
This function evaluates the Wendland radial basis function with parameters \code{s = 2} and \code{k = 1}.
It is compactly supported, meaning the function value is zero for distances greater than \code{r}.
}
\details{
The Wendland radial basis function is commonly used in spatial statistics and machine learning for interpolation and smoothing.
This implementation assumes parameters \code{s = 2} and \code{k = 1}.
}
\examples{
# Example: Compute Wendland function values for a vector of distances
distances <- c(0, 0.5, 1, 1.5, 2)
radius <- 1
wendland(distances, radius)

}
