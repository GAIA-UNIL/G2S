#include "R_interface.hpp"

// [[Rcpp::export]]
Rcpp::RObject g2sInterface(Rcpp::List args)
{
	
	InterfaceTemplateR interfaceTemplateR;
	return interfaceTemplateR.runStandardCommunicationR( args );
	
}
