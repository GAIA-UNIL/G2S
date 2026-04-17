#ifndef ANCHOR_SAMPLING_MODULE_HPP
#define ANCHOR_SAMPLING_MODULE_HPP

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "computeDeviceModule.hpp"
#include "DataImage.hpp"
#include "samplingModule.hpp"

struct AnchorSamplingExpandedVariable{
	enum class Kind{
		Continuous,
		Categorical
	};

	Kind kind=Kind::Continuous;
	unsigned originalVariable=0;
};

class AnchorSamplingData {
public:
	std::vector<unsigned> _dims;
	unsigned _nbVariable=0;
	unsigned _nbExpandedVariable=0;
	unsigned _nbTI=0;
	bool _hasMask=false;

	std::vector<AnchorSamplingExpandedVariable> _expandedVariables;
	std::vector<float> _expandedValues;
	std::vector<std::uint8_t> _centerValidVector;
	std::vector<std::uint8_t> _centerValidScalar;
	std::vector<float> _maskWeights;

	inline unsigned cellCount() const{
		unsigned count=1;
		for (size_t i = 0; i < _dims.size(); ++i)
		{
			count*=_dims[i];
		}
		return count;
	}

	inline size_t expandedIndex(unsigned cellIndex, unsigned expandedVariable, unsigned ti) const{
		return (static_cast<size_t>(cellIndex)*_nbExpandedVariable+expandedVariable)*_nbTI+ti;
	}

	inline size_t scalarValidityIndex(unsigned cellIndex, unsigned variable, unsigned ti) const{
		return (static_cast<size_t>(cellIndex)*_nbVariable+variable)*_nbTI+ti;
	}

	inline size_t vectorValidityIndex(unsigned cellIndex, unsigned ti) const{
		return static_cast<size_t>(cellIndex)*_nbTI+ti;
	}

	inline size_t maskIndex(unsigned cellIndex, unsigned ti) const{
		return static_cast<size_t>(cellIndex)*_nbTI+ti;
	}

	static inline AnchorSamplingData build(
		const std::vector<g2s::DataImage> &TIs,
		const std::vector<std::vector<float> > &categoriesValues,
		const g2s::DataImage *maskImage=nullptr)
	{
		AnchorSamplingData stack;
		if(TIs.empty()){
			return stack;
		}

		stack._dims=TIs[0]._dims;
		stack._nbVariable=TIs[0]._nbVariable;
		stack._nbTI=TIs.size();
		stack._hasMask=(maskImage!=nullptr);

		unsigned categoriesIndex=0;
		for (unsigned variable = 0; variable < stack._nbVariable; ++variable)
		{
			if(TIs[0]._types[variable]==g2s::DataImage::Continuous){
				AnchorSamplingExpandedVariable info;
				info.kind=AnchorSamplingExpandedVariable::Kind::Continuous;
				info.originalVariable=variable;
				stack._expandedVariables.push_back(info);
			}else{
				const unsigned numberOfCategory=categoriesValues[categoriesIndex].size();
				for (unsigned category = 0; category < numberOfCategory; ++category)
				{
					AnchorSamplingExpandedVariable info;
					info.kind=AnchorSamplingExpandedVariable::Kind::Categorical;
					info.originalVariable=variable;
					stack._expandedVariables.push_back(info);
				}
				categoriesIndex++;
			}
		}
		stack._nbExpandedVariable=stack._expandedVariables.size();

		const unsigned numberOfCell=stack.cellCount();
		stack._expandedValues.resize(static_cast<size_t>(numberOfCell)*stack._nbExpandedVariable*stack._nbTI,0.f);
		stack._centerValidVector.resize(static_cast<size_t>(numberOfCell)*stack._nbTI,0);
		stack._centerValidScalar.resize(static_cast<size_t>(numberOfCell)*stack._nbVariable*stack._nbTI,0);
		if(maskImage){
			stack._maskWeights.resize(static_cast<size_t>(numberOfCell)*stack._nbTI,0.f);
		}

		for (unsigned ti = 0; ti < stack._nbTI; ++ti)
		{
			unsigned expandedVariableIndex=0;
			unsigned localCategoriesIndex=0;
			for (unsigned cell = 0; cell < numberOfCell; ++cell)
			{
				bool fullVectorValid=true;
				for (unsigned variable = 0; variable < stack._nbVariable; ++variable)
				{
					const float value=TIs[ti]._data[cell*stack._nbVariable+variable];
					const bool valid=!std::isnan(value);
					stack._centerValidScalar[stack.scalarValidityIndex(cell,variable,ti)]=valid;
					fullVectorValid&=valid;
				}
				stack._centerValidVector[stack.vectorValidityIndex(cell,ti)]=fullVectorValid;

				if(maskImage){
					stack._maskWeights[stack.maskIndex(cell,ti)]=maskImage->_data[cell*maskImage->_nbVariable+ti];
				}
			}

			expandedVariableIndex=0;
			localCategoriesIndex=0;
			for (unsigned variable = 0; variable < stack._nbVariable; ++variable)
			{
				if(TIs[ti]._types[variable]==g2s::DataImage::Continuous){
					for (unsigned cell = 0; cell < numberOfCell; ++cell)
					{
						const float value=TIs[ti]._data[cell*stack._nbVariable+variable];
						stack._expandedValues[stack.expandedIndex(cell,expandedVariableIndex,ti)]=std::isnan(value) ? 0.f : value;
					}
					expandedVariableIndex++;
				}else{
					const std::vector<float> &categoryValues=categoriesValues[localCategoriesIndex];
					for (unsigned category = 0; category < categoryValues.size(); ++category)
					{
						for (unsigned cell = 0; cell < numberOfCell; ++cell)
						{
							const float value=TIs[ti]._data[cell*stack._nbVariable+variable];
							const float encoded=(!std::isnan(value) && value==categoryValues[category]) ? 1.f : 0.f;
							stack._expandedValues[stack.expandedIndex(cell,expandedVariableIndex,ti)]=encoded;
						}
						expandedVariableIndex++;
					}
					localCategoriesIndex++;
				}
			}
		}

		return stack;
	}
};

class AnchorSamplingModule {
private:
	const AnchorSamplingData *_stack=nullptr;
	g2s::DataImage *_kernel=nullptr;
	g2s::DataImage *_referenceGrid=nullptr;
	float _k=1.f;
	bool _completeTIs=true;
	bool _circularTI=false;

public:
	AnchorSamplingModule(
		const AnchorSamplingData *stack,
		g2s::DataImage *referenceGrid,
		g2s::DataImage *kernel,
		float k,
		bool completeTIs,
		bool circularTI):
		_stack(stack),
		_kernel(kernel),
		_referenceGrid(referenceGrid),
		_k(k),
		_completeTIs(completeTIs),
		_circularTI(circularTI)
	{
	}

	inline SamplingModule::matchLocation sample(
		unsigned currentCell,
		std::vector<std::vector<int> > &neighborArrayVector,
		std::vector<std::vector<float> > &neighborValueArrayVector,
		float seed,
		bool fullStationary=false,
		unsigned variableOfInterest=0,
		float localk=0.f,
		g2s::DataImage *localKernel=nullptr) const
	{
		g2s::DataImage *kernel=(localKernel!=nullptr ? localKernel : _kernel);
		if(localk<=0.f){
			localk=_k;
		}

		std::vector<unsigned> candidateOrder(_stack->_nbTI);
		std::iota(candidateOrder.begin(),candidateOrder.end(),0);
		if(fullStationary && !candidateOrder.empty()){
			std::mt19937 generator;
			generator.seed(static_cast<unsigned>(std::floor(seed*float(UINT_MAX))));
			std::shuffle(candidateOrder.begin(),candidateOrder.end(),generator);
			const unsigned retained=std::max<unsigned>(1,static_cast<unsigned>(std::ceil(_stack->_nbTI/std::max(localk,1.f))));
			candidateOrder.resize(std::min<unsigned>(retained,candidateOrder.size()));
		}

		struct CandidateScore{
			unsigned ti=0;
			float score=-INFINITY;
			float weight=std::nanf("0");
		};

		std::vector<CandidateScore> candidates;
		candidates.reserve(candidateOrder.size());

		unsigned kernelCenter=0;
		for (int i = int(kernel->_dims.size()-1); i >=0 ; --i)
		{
			kernelCenter=kernelCenter*kernel->_dims[i]+kernel->_dims[i]/2;
		}

		for (size_t orderIndex = 0; orderIndex < candidateOrder.size(); ++orderIndex)
		{
			const unsigned ti=candidateOrder[orderIndex];
			const bool centerValid=(variableOfInterest==UINT_MAX)
				? bool(_stack->_centerValidVector[_stack->vectorValidityIndex(currentCell,ti)])
				: bool(_stack->_centerValidScalar[_stack->scalarValidityIndex(currentCell,variableOfInterest,ti)]);
			if(!centerValid){
				continue;
			}

			float maskWeight=std::nanf("0");
			if(_stack->_hasMask){
				maskWeight=_stack->_maskWeights[_stack->maskIndex(currentCell,ti)];
				if(std::isnan(maskWeight)){
					continue;
				}
			}

			float score=0.f;
			float support=0.f;
			float penalty=0.f;

			for (size_t neighbor = 0; neighbor < neighborArrayVector.size(); ++neighbor)
			{
				unsigned indexInKernel=kernelCenter;
				if(!kernel->indexWithDelta(indexInKernel,kernelCenter,neighborArrayVector[neighbor])){
					continue;
				}

				unsigned anchoredNeighborCell=0;
				const bool inBounds=_referenceGrid->indexWithDelta(anchoredNeighborCell,currentCell,neighborArrayVector[neighbor],nullptr);
				if(!inBounds && !_circularTI){
					continue;
				}

				for (unsigned expandedVariable = 0; expandedVariable < _stack->_nbExpandedVariable; ++expandedVariable)
				{
					if(expandedVariable>=neighborValueArrayVector[neighbor].size()){
						continue;
					}

					const float neighborValue=neighborValueArrayVector[neighbor][expandedVariable];
					if(std::isnan(neighborValue)){
						continue;
					}

					const float kernelWeight=kernel->_data[indexInKernel*kernel->_nbVariable+expandedVariable];
					if(std::isnan(kernelWeight) || kernelWeight==0.f){
						continue;
					}

					const AnchorSamplingExpandedVariable &expandedInfo=_stack->_expandedVariables[expandedVariable];
					if(!_stack->_centerValidScalar[_stack->scalarValidityIndex(anchoredNeighborCell,expandedInfo.originalVariable,ti)]){
						continue;
					}

					const float tiValue=_stack->_expandedValues[_stack->expandedIndex(anchoredNeighborCell,expandedVariable,ti)];
					if(_completeTIs){
						if(expandedInfo.kind==AnchorSamplingExpandedVariable::Kind::Continuous){
							const float diff=tiValue-neighborValue;
							score-=kernelWeight*diff*diff;
						}else if(neighborValue>0.5f){
							score+=kernelWeight*tiValue;
						}
					}else{
						const float supportWeight=std::fabs(kernelWeight);
						if(expandedInfo.kind==AnchorSamplingExpandedVariable::Kind::Continuous){
							const float diff=tiValue-neighborValue;
							penalty+=supportWeight*diff*diff;
							support+=supportWeight;
						}else if(neighborValue>0.5f){
							penalty+=supportWeight*(1.f-tiValue);
							support+=supportWeight;
						}
					}
				}
			}

			if(!_completeTIs){
				if(support<=0.f){
					continue;
				}
				score=-(std::fabs(penalty)/(support*support*support*support));
			}

			CandidateScore candidate;
			candidate.ti=ti;
			candidate.score=score;
			candidate.weight=maskWeight;
			candidates.push_back(candidate);
		}

		if(candidates.empty()){
			SamplingModule::matchLocation result;
			result.TI=0;
			result.index=currentCell;
			return result;
		}

		const unsigned retainedCount=std::min<unsigned>(std::max<unsigned>(1,static_cast<unsigned>(std::ceil(localk))),candidates.size());
		std::nth_element(candidates.begin(),candidates.begin()+retainedCount-1,candidates.end(),
			[](const CandidateScore &lhs, const CandidateScore &rhs){
				return lhs.score>rhs.score;
			});
		candidates.resize(retainedCount);
		std::sort(candidates.begin(),candidates.end(),
			[](const CandidateScore &lhs, const CandidateScore &rhs){
				return lhs.score>rhs.score;
			});

		unsigned selectedPosition=0;
		if(_stack->_hasMask){
			float weightSum=0.f;
			for (size_t i = 0; i < candidates.size(); ++i)
			{
				if(candidates[i].weight>0.f){
					weightSum+=candidates[i].weight;
				}
			}

			if(weightSum>0.f){
				const float threshold=seed*weightSum;
				float cumulative=0.f;
				for (size_t i = 0; i < candidates.size(); ++i)
				{
					if(candidates[i].weight<=0.f){
						continue;
					}
					cumulative+=candidates[i].weight;
					if(threshold<=cumulative){
						selectedPosition=i;
						break;
					}
				}
			}else{
				selectedPosition=std::min<unsigned>(static_cast<unsigned>(std::floor(seed*candidates.size())),candidates.size()-1);
			}
		}else{
			selectedPosition=std::min<unsigned>(static_cast<unsigned>(std::floor(seed*candidates.size())),candidates.size()-1);
		}

		SamplingModule::matchLocation result;
		result.TI=candidates[selectedPosition].ti;
		result.index=currentCell;
		return result;
	}
};

#endif // ANCHOR_SAMPLING_MODULE_HPP
