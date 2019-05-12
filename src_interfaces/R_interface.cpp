#include "R_interface.hpp"

// [[Rcpp::export]]
Rcpp::RObject g2sInterface(Rcpp::List args)
{
	
	InerfaceTemplateR inerfaceTemplateR;
	return inerfaceTemplateR.runStandardCommunicationR( args );
	
}
