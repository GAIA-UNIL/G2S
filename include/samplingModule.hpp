#ifndef SAMPLING_MODULE_HPP
#define SAMPLING_MODULE_HPP

class SamplingModule {
public:
	struct matchLocation{
		unsigned TI;
		unsigned index;
	};

	struct narrownessMeasurment{
		matchLocation candidate;
		float narrowness;
	};

	enum convertionType{
		P0=0,
		P1=1,
		P2=2
	};
protected:
	std::vector<ComputeDeviceModule*> *_cdmV;
	g2s::DataImage* _kernel;
	std::function<float(float*, unsigned int *, unsigned int * , unsigned int )> _narrownessFunction;
public:
	SamplingModule(std::vector<ComputeDeviceModule *> *cdmV, g2s::DataImage *kernel){
		_cdmV=cdmV;
		_kernel=kernel;
	};
	~SamplingModule(){};

	void setNarrownessFunction(std::function<float(float*, unsigned int *, unsigned int * , unsigned int )> narrownessFunction){
		_narrownessFunction=narrownessFunction;
	}

	virtual matchLocation sample(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false)=0;
	virtual narrownessMeasurment narrowness(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false)=0;
};

#endif // SAMPLING_MODULE_HPP