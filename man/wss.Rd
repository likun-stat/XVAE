% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/XVAE_utils.R
\name{wss}
\alias{wss}
\title{Calculate Total Within-Cluster Sum of Squares (WSS) for k-Means Clustering}
\usage{
wss(k, df)
}
\arguments{
\item{k}{Integer. The number of clusters for the k-means algorithm.}

\item{df}{Data frame or matrix. The dataset to be clustered, where rows are observations
and columns are features.}
}
\value{
Numeric. The total within-cluster sum of squares (WSS) for the specified \code{k}.
}
\description{
This function computes the total WSS for a specified number of clusters (\code{k})
on a given dataset. It is commonly used to evaluate clustering performance
and select an optimal number of clusters.
}
\examples{
wss(3, iris[, -5])
}
