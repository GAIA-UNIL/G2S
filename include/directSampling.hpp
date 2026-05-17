#ifndef DIRECT_SAMPLING_HPP
#define DIRECT_SAMPLING_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "samplingModule.hpp"
#include "DataImage.hpp"

class DirectSamplingModule: public SamplingModule {
private:
	std::vector<g2s::DataImage>* _tis=nullptr;
	std::vector<g2s::DataImage::VariableType> _types;
	std::vector<float> _continuousNormPowerByVariable;
	float _threshold=0.f;
	float _maxExplorationRatio=1.f;
	bool _circularTI=false;

	static inline uint64_t mix64(uint64_t value){
		value+=0x9e3779b97f4a7c15ULL;
		value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
		value=(value^(value>>27))*0x94d049bb133111ebULL;
		return value^(value>>31);
	}

	static inline uint32_t floatBits(float value){
		uint32_t bits=0;
		std::memcpy(&bits,&value,sizeof(float));
		return bits;
	}

	static inline SampleContext& threadSampleContext(){
		static thread_local SampleContext context;
		return context;
	}

	static inline unsigned permutationBitCount(size_t count){
		unsigned bits=0;
		size_t value=1;
		while(value<count && bits<sizeof(size_t)*8){
			value<<=1;
			bits++;
		}
		return bits;
	}

	static inline size_t permutePowerOfTwo(size_t value, unsigned bits, uint64_t key){
		if(bits==0){
			return 0;
		}
		const size_t mask=(bits>=sizeof(size_t)*8) ? std::numeric_limits<size_t>::max() : ((size_t(1)<<bits)-size_t(1));
		size_t x=value&mask;
		const size_t mul1=static_cast<size_t>(mix64(key^0x243f6a8885a308d3ULL))|size_t(1);
		const size_t mul2=static_cast<size_t>(mix64(key^0x13198a2e03707344ULL))|size_t(1);
		const size_t add1=static_cast<size_t>(mix64(key^0xa4093822299f31d0ULL));
		const size_t add2=static_cast<size_t>(mix64(key^0x082efa98ec4e6c89ULL));
		const unsigned shift1=std::max(1u,bits/2u);
		const unsigned shift2=std::max(1u,bits/3u+1u);
		x=(x+add1)&mask;
		x^=(x>>shift1);
		x=(x*mul1)&mask;
		x^=(x>>shift2);
		x=(x+add2)&mask;
		x=(x*mul2)&mask;
		x^=(x>>shift1);
		return x&mask;
	}

	static inline size_t permutedCandidate(size_t scan, size_t candidateCount, uint64_t key){
		if(candidateCount<=1){
			return 0;
		}
		const unsigned bits=permutationBitCount(candidateCount);
		size_t value=scan;
		do {
			value=permutePowerOfTwo(value,bits,key);
		} while(value>=candidateCount);
		return value;
	}

	inline bool centerIsValid(g2s::DataImage& ti, unsigned index, unsigned variableOfInterest, const SampleContext& context) const{
		if(index>=ti.dataSize()/ti._nbVariable){
			return false;
		}
		if(context.fullSimulation){
			return variableOfInterest<ti._nbVariable && std::isfinite(ti._data[index*ti._nbVariable+variableOfInterest]);
		}
		for (unsigned variable = 0; variable < ti._nbVariable; ++variable)
		{
			if(!std::isfinite(ti._data[index*ti._nbVariable+variable])){
				return false;
			}
		}
		return true;
	}

	inline float kernelWeight(g2s::DataImage* kernel, unsigned neighborIndex, unsigned variable, const SampleContext& context) const{
		if(kernel==nullptr || kernel->isEmpty()){
			return 1.f;
		}
		unsigned flatIndex=0;
		if(context.kernelFlatIndexVector!=nullptr && neighborIndex<context.kernelFlatIndexVector->size() && (*context.kernelFlatIndexVector)[neighborIndex]>=0){
			flatIndex=static_cast<unsigned>((*context.kernelFlatIndexVector)[neighborIndex]);
		}else{
			unsigned center=0;
			for (int dim = int(kernel->_dims.size())-1; dim>=0; --dim)
			{
				center=center*kernel->_dims[dim]+kernel->_dims[dim]/2;
			}
			flatIndex=center;
		}
		const unsigned kernelVariable=(kernel->_nbVariable==1 ? 0 : std::min(variable,kernel->_nbVariable-1));
		const float weight=kernel->_data[flatIndex*kernel->_nbVariable+kernelVariable];
		if(!std::isfinite(weight)){
			return 0.f;
		}
		return std::fabs(weight);
	}

	inline bool tiIndexWithDelta(g2s::DataImage& ti, unsigned& outIndex, unsigned centerIndex, const std::vector<int>& offset) const{
		if(ti.indexWithDelta(outIndex, centerIndex, offset, nullptr)){
			return true;
		}
		if(!_circularTI || ti._dims.empty()){
			return false;
		}
		std::vector<unsigned> coord(ti._dims.size(),0);
		unsigned remaining=centerIndex;
		for (int dim = int(ti._dims.size())-1; dim>=0; --dim)
		{
			coord[dim]=remaining%ti._dims[dim];
			remaining/=ti._dims[dim];
		}
		for (size_t dim = 0; dim < ti._dims.size(); ++dim)
		{
			const int size=int(ti._dims[dim]);
			int value=int(coord[dim])+(dim<offset.size() ? offset[dim] : 0);
			value%=size;
			if(value<0){
				value+=size;
			}
			coord[dim]=static_cast<unsigned>(value);
		}
		outIndex=0;
		for (size_t dim = 0; dim < ti._dims.size(); ++dim)
		{
			outIndex=outIndex*ti._dims[dim]+coord[dim];
		}
		return true;
	}

	inline float scoreCandidate(
		g2s::DataImage& ti,
		unsigned candidateIndex,
		std::vector<std::vector<int> >& neighborArrayVector,
		std::vector<std::vector<float> >& neighborValueArrayVector,
		unsigned variableOfInterest,
		g2s::DataImage* kernel,
		const SampleContext& context) const{

		if(!centerIsValid(ti,candidateIndex,variableOfInterest,context)){
			return std::numeric_limits<float>::infinity();
		}

		std::vector<float> continuousPower(_types.size(),0.f);
		std::vector<float> continuousSupport(_types.size(),0.f);
		std::vector<float> categoricalMismatch(_types.size(),0.f);
		std::vector<float> categoricalSupport(_types.size(),0.f);

		for (size_t neighbor = 0; neighbor < neighborArrayVector.size(); ++neighbor)
		{
			unsigned tiNeighborIndex=0;
			if(!tiIndexWithDelta(ti,tiNeighborIndex,candidateIndex,neighborArrayVector[neighbor])){
				for (unsigned variable = 0; variable < ti._nbVariable && variable < neighborValueArrayVector[neighbor].size() && variable < _types.size(); ++variable)
				{
					const float observed=neighborValueArrayVector[neighbor][variable];
					if(std::isfinite(observed) && kernelWeight(kernel,static_cast<unsigned>(neighbor),variable,context)>0.f){
						return std::numeric_limits<float>::infinity();
					}
				}
				continue;
			}
			for (unsigned variable = 0; variable < ti._nbVariable && variable < neighborValueArrayVector[neighbor].size() && variable < _types.size(); ++variable)
			{
				const float observed=neighborValueArrayVector[neighbor][variable];
				const float expected=ti._data[tiNeighborIndex*ti._nbVariable+variable];
				if(!std::isfinite(observed)){
					continue;
				}
				const float weight=kernelWeight(kernel,static_cast<unsigned>(neighbor),variable,context);
				if(weight<=0.f){
					continue;
				}
				if(!std::isfinite(expected)){
					return std::numeric_limits<float>::infinity();
				}
				if(_types[variable]==g2s::DataImage::Continuous){
					const float p=std::max(1.0e-6f,_continuousNormPowerByVariable[variable]);
					continuousPower[variable]+=weight*std::pow(std::fabs(observed-expected),p);
					continuousSupport[variable]+=weight;
				}else{
					categoricalMismatch[variable]+=weight*(observed==expected ? 0.f : 1.f);
					categoricalSupport[variable]+=weight;
				}
			}
		}

		float totalScore=0.f;
		float activeVariableCount=0.f;
		for (unsigned variable = 0; variable < _types.size(); ++variable)
		{
			if(_types[variable]==g2s::DataImage::Continuous){
				if(continuousSupport[variable]>0.f){
					const float p=std::max(1.0e-6f,_continuousNormPowerByVariable[variable]);
					totalScore+=std::pow(continuousPower[variable]/continuousSupport[variable],1.f/p);
					activeVariableCount+=1.f;
				}
			}else if(categoricalSupport[variable]>0.f){
				totalScore+=categoricalMismatch[variable]/categoricalSupport[variable];
				activeVariableCount+=1.f;
			}
		}

		if(activeVariableCount<=0.f){
			return std::numeric_limits<float>::infinity();
		}
		return totalScore/activeVariableCount;
	}

public:
	DirectSamplingModule(
		std::vector<g2s::DataImage>* tis,
		g2s::DataImage* defaultKernel,
		const std::vector<g2s::DataImage::VariableType>& types,
		const std::vector<float>& continuousNormPowerByVariable,
		float threshold,
		float maxExplorationRatio,
		bool circularTI):
		SamplingModule(nullptr,defaultKernel),
		_tis(tis),
		_types(types),
		_continuousNormPowerByVariable(continuousNormPowerByVariable),
		_threshold(threshold),
		_maxExplorationRatio(maxExplorationRatio),
		_circularTI(circularTI)
	{
		if(_continuousNormPowerByVariable.size()!=_types.size()){
			_continuousNormPowerByVariable.assign(_types.size(),2.f);
		}
		for (size_t i = 0; i < _continuousNormPowerByVariable.size(); ++i)
		{
			if(!std::isfinite(_continuousNormPowerByVariable[i]) || _continuousNormPowerByVariable[i]<=0.f){
				_continuousNormPowerByVariable[i]=2.f;
			}
		}
	}

	bool useRawNeighborValues() const override { return true; }
	bool strictInformedNeighbors() const override { return true; }

	int resolveTiId(float rawValue, unsigned tiCount) const override {
		if(!std::isfinite(rawValue) || rawValue<0.f){
			return -1;
		}
		const float rounded=std::round(rawValue);
		if(std::fabs(rawValue-rounded)>1.0e-5f){
			return -1;
		}
		const int value=static_cast<int>(rounded);
		if(value<0 || value>=static_cast<int>(tiCount)){
			return -1;
		}
		return value;
	}

	void setSampleContext(const SampleContext& context) override {
		threadSampleContext()=context;
	}

	matchLocation sample(std::vector<std::vector<int> > neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, matchLocation verbatimRecord, unsigned moduleID=0, bool fullStationary=false, unsigned variableOfInterest=0, float localk=0.f, int idTI4Sampling=-1, g2s::DataImage* localKernel=nullptr) override {
		(void)verbatimRecord;
		(void)moduleID;
		(void)fullStationary;
		const SampleContext& context=threadSampleContext();
		g2s::DataImage* kernel=(localKernel!=nullptr ? localKernel : _kernel);
		matchLocation best;
		best.TI=0;
		best.index=0;
		if(_tis==nullptr || _tis->empty()){
			return best;
		}

		std::vector<unsigned> allowedTis;
		if(idTI4Sampling>=0 && idTI4Sampling<static_cast<int>(_tis->size())){
			allowedTis.push_back(static_cast<unsigned>(idTI4Sampling));
		}else{
			allowedTis.resize(_tis->size());
			std::iota(allowedTis.begin(),allowedTis.end(),0);
		}

		size_t totalCandidateCount=0;
		for (size_t i = 0; i < allowedTis.size(); ++i)
		{
			totalCandidateCount+=(*_tis)[allowedTis[i]].dataSize()/(*_tis)[allowedTis[i]]._nbVariable;
		}
		if(totalCandidateCount==0){
			return best;
		}

		const float explorationRatio=(localk>0.f && std::isfinite(localk)) ? localk : _maxExplorationRatio;
		const size_t maxScanned=std::max<size_t>(1,std::min<size_t>(totalCandidateCount,static_cast<size_t>(std::ceil(double(totalCandidateCount)*std::max(0.f,explorationRatio)))));
		const uint64_t permutationKey=mix64(uint64_t(context.globalSeed))^mix64(uint64_t(context.pathIndex))^(mix64(uint64_t(variableOfInterest))<<1)^mix64(uint64_t(floatBits(seed)));

		if(neighborArrayVector.empty()){
			for (size_t scan = 0; scan < totalCandidateCount; ++scan)
			{
				size_t flattened=permutedCandidate(scan,totalCandidateCount,permutationKey);
				unsigned tiId=allowedTis.front();
				unsigned index=0;
				for (size_t tiOrder = 0; tiOrder < allowedTis.size(); ++tiOrder)
				{
					tiId=allowedTis[tiOrder];
					const unsigned cellCount=(*_tis)[tiId].dataSize()/(*_tis)[tiId]._nbVariable;
					if(flattened<cellCount){
						index=static_cast<unsigned>(flattened);
						break;
					}
					flattened-=cellCount;
				}
				if(centerIsValid((*_tis)[tiId],index,variableOfInterest,context)){
					best.TI=tiId;
					best.index=index;
					return best;
				}
			}
			return best;
		}

		float bestScore=std::numeric_limits<float>::infinity();
		bool foundAny=false;
		for (size_t scan = 0; scan < maxScanned; ++scan)
		{
			size_t flattened=permutedCandidate(scan,totalCandidateCount,permutationKey);
			unsigned tiId=allowedTis.front();
			unsigned index=0;
			for (size_t tiOrder = 0; tiOrder < allowedTis.size(); ++tiOrder)
			{
				tiId=allowedTis[tiOrder];
				const unsigned cellCount=(*_tis)[tiId].dataSize()/(*_tis)[tiId]._nbVariable;
				if(flattened<cellCount){
					index=static_cast<unsigned>(flattened);
					break;
				}
				flattened-=cellCount;
			}

			const float score=scoreCandidate((*_tis)[tiId],index,neighborArrayVector,neighborValueArrayVector,variableOfInterest,kernel,context);
			if(score<bestScore){
				bestScore=score;
				best.TI=tiId;
				best.index=index;
				foundAny=true;
			}
			if(std::isfinite(score) && score<=_threshold){
				best.TI=tiId;
				best.index=index;
				return best;
			}
		}

		if(foundAny){
			return best;
		}

		for (size_t tiOrder = 0; tiOrder < allowedTis.size(); ++tiOrder)
		{
			const unsigned tiId=allowedTis[tiOrder];
			const unsigned cellCount=(*_tis)[tiId].dataSize()/(*_tis)[tiId]._nbVariable;
			for (unsigned index = 0; index < cellCount; ++index)
			{
				if(centerIsValid((*_tis)[tiId],index,variableOfInterest,context)){
					best.TI=tiId;
					best.index=index;
					return best;
				}
			}
		}
		return best;
	}

	narrownessMeasurment narrowness(std::vector<std::vector<int> > neighborArrayVector, std::vector<std::vector<float> > neighborValueArrayVector,float seed, unsigned moduleID=0, bool fullStationary=false) override {
		(void)neighborArrayVector;
		(void)neighborValueArrayVector;
		(void)seed;
		(void)moduleID;
		(void)fullStationary;
		narrownessMeasurment result;
		result.candidate.TI=0;
		result.candidate.index=0;
		result.narrowness=0.f;
		return result;
	}
};

#endif
