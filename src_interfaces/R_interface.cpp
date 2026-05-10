#include "R_interface.hpp"
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::RObject g2sInterface(SEXP args)
{
	Rcpp::List rargs(args); // wrap SEXP as List
	InterfaceTemplateR interfaceTemplateR;
	return interfaceTemplateR.runStandardCommunicationR(rargs);
}
