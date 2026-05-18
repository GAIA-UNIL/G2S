/*
 * G2S
 * Copyright (C) 2018, Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL (University of Lausanne)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef SAMPLING_MODULE_HPP
#define SAMPLING_MODULE_HPP

#include <functional>
#include <cstddef>
#include "typeDefine.hpp"
#include "pathIndexType.hpp"

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

	struct simValues{
		unsigned index;
		std::vector<float> values;
	};

	struct SampleContext{
		unsigned currentCell=0;
		g2s_path_index_t pathIndex=0;
		unsigned variableOfInterest=0;
		unsigned globalSeed=0;
		const std::vector<int>* kernelFlatIndexVector=nullptr;
		bool fullSimulation=false;
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

	virtual bool useRawNeighborValues() const { return false; }
	virtual bool strictInformedNeighbors() const { return false; }
	virtual int resolveTiId(float rawValue, unsigned tiCount) const {
		(void)tiCount;
		return int(rawValue);
	}
	virtual void setSampleContext(const SampleContext& context){
		(void)context;
	}

	virtual matchLocation sample(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, matchLocation verbatimRecord, unsigned moduleID=0, bool fullStationary=false, unsigned variableOfInterest=0, float localk=0.f, int idTI4Sampling=-1, g2s::DataImage* localKernel=nullptr)=0;
	virtual narrownessMeasurment narrowness(std::vector<std::vector<int>> neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false)=0;
};

#endif // SAMPLING_MODULE_HPP
