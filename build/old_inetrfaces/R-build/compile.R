library(Rcpp)
setwd("~/githubProject/G2S/build/R-build/")
Sys.setenv("PKG_CXXFLAGS"="-std=c++11 -I/opt/local/include -I/Users/mathieugravey/githubProject/G2S/include/")
Sys.setenv("PKG_LIBS"="-L/opt/local/lib -lzmq -ljsoncpp")

sourceCpp("../../src/g2sRExtension.cpp")