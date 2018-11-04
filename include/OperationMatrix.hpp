#ifndef OPERATION_MATRIX_HPP
#define OPERATION_MATRIX_HPP

#include "typeDefine.hpp"
#include <vector>

namespace g2s{

class OperationMatrix
{

private:
	std::vector<float> _matrixData;
	int _numberOfVariable;

public:
	OperationMatrix(int numberOfVariable){
		_numberOfVariable=numberOfVariable;
		_matrixData = std::vector<float>(numberOfVariable * numberOfVariable, 0.f);
	}

	~OperationMatrix(){
	}

	unsigned getNumberOfVariable(){
		return _numberOfVariable;
	}

	void setVariableAt(int variableA, int variableB, float value){
		_matrixData[variableA+variableB*_numberOfVariable]=value;
	}

	float getVariableAt(int variableA, int variableB){
		return _matrixData[variableA+variableB*_numberOfVariable];
	}

	float getVariableCumulatedAlongA(int variableA){
		float cummulated=0;
		for (int i = 0; i < _numberOfVariable; ++i)
		{
			cummulated+=_matrixData[variableA+i*_numberOfVariable];
		}
		return cummulated;
	}

	bool needVariableAlongA(int variableA){
		bool needed=0;
		for (int i = 0; i < _numberOfVariable; ++i)
		{
			needed|=(_matrixData[variableA+i*_numberOfVariable]!=0.f);
		}
		return needed;
	}

	float getVariableCumulatedAlongB(int variableB){
		float cummulated=0;
		for (int i = 0; i < _numberOfVariable; ++i)
		{
			cummulated+=_matrixData[i+variableB*_numberOfVariable];
		}
		return cummulated;
	}

	bool needVariableAlongB(int variableB){
		bool needed=0;
		for (int i = 0; i < _numberOfVariable; ++i)
		{
			needed|=(_matrixData[i+variableB*_numberOfVariable]!=0.f);
		}
		return needed;
	}
	
};

}

#endif // OPERATION_MATRIX_HPP