library(Rcpp)
curdir=getwd();
compileFileDir=normalizePath(getSrcDirectory(function(x) {x}))
setwd(compileFileDir)
Sys.setenv("PKG_CXXFLAGS"=paste("-std=c++17 -I/opt/local/include -I",compileFileDir,"/../../include/ -I",compileFileDir,"/../../include_interfaces/",sep=""))
Sys.setenv("PKG_LIBS"="-L/opt/local/lib -lzmq -ljsoncpp")

sourceCpp("../../src_interfaces/R_interface.cpp")
setwd(curdir)

