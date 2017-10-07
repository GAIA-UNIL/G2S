#ifndef SAMPLING_MODULE_HPP
#define SAMPLING_MODULE_HPP

class SamplingModule {
public:
	struct matchLocation{
		unsigned TI;
		unsigned index;
	};
	enum convertionType{
		P0=0,
		P1=1,
		P2=2
	};
protected:
	std::vector<ComputeDeviceModule*> *_cdmV;
	g2s::DataImage* _kernel;
public:
	SamplingModule(std::vector<ComputeDeviceModule *> *cdmV, g2s::DataImage *kernel){
		_cdmV=cdmV;
		_kernel=kernel;
	};
	~SamplingModule(){};

	virtual matchLocation sample(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false)=0;
};

#endif // SAMPLING_MODULE_HPP