library(Rcpp)
curdir=getwd();
compileFileDir=normalizePath(getSrcDirectory(function(x) {x}))
setwd(compileFileDir)
Sys.setenv("PKG_CXXFLAGS"=paste("-std=c++17 -I",compileFileDir,sep=""))
Sys.setenv("PKG_LIBS"="-L/opt/homebrew/lib -lzmq")

sourceCpp("../../src_interfaces/R_interface.cpp")
setwd(curdir)


