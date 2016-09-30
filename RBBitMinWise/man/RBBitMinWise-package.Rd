\name{RBBitMinWise-package}
\alias{RBBitMinWise-package}
\alias{RBBitMinWise}
\docType{package}
\title{
  Providing useful functions for use in bBit Minwise Hash calculations
}
\description{
  This file is designed to implement the methods, not first, descibed in the following paper https://arxiv.org/pdf/1308.1269v3.pdf
  It implements method 2.2 for performing bBit Minwise Hashing on sparse binary data sets.
  Created by Thomas Webster under supervision of Dr Rajen Shah and supported by the Bridgewater program.
}
\details{
  File contains two functions of use.
  bBitCompress(...) takes the data set and uses bBit Minwise Hashing to produce a reduction in dimensionality.
  It outputs two files, one contains the compressed data set and the second the permutations used to produce it.
  The permutation file is required to process new data before using a model base on the compressed data set for predictions.
  vectorProcess(...) takes a new row of data and uses the permutations to replicate the compression.
  It returns a vector of variable length, entries corespond to positions where the values are 1. Other values are assumed to be 0.
}
\author{
Thomas Webster, tw432@cam.ac.uk

Maintainer: Thomas Webster, tw432@cam.ac.uk
}
\references{
  Some relevant papers on the topic.
  https://arxiv.org/pdf/1308.1269v3.pdf
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/wfc0398-liPS.pdf
  
}
\keyword{ package }