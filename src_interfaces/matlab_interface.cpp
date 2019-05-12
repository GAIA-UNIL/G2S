#include "matlab_interface.hpp"

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	
	InerfaceTemplateMatlab inerfaceTemplateMatlab;
	inerfaceTemplateMatlab.runStandardCommunicationMatlab( nlhs, plhs,  nrhs, prhs );
	mexEvalString("drawnow");
}
