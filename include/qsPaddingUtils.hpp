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

#ifndef QS_PADDING_UTILS_HPP
#define QS_PADDING_UTILS_HPP

#include <vector>
#include <algorithm>
#include <cstring>
#include "DataImage.hpp"
#include "pathIndexType.hpp"

namespace qs_padding_utils {

inline bool hasPadding(const std::vector<unsigned>& padding){
	return std::any_of(padding.begin(),padding.end(),[](unsigned v){return v>0;});
}

inline std::vector<unsigned> paddedDims(const std::vector<unsigned>& dims, const std::vector<unsigned>& padding){
	std::vector<unsigned> out(dims.size(),0);
	for (size_t i = 0; i < dims.size(); ++i)
	{
		out[i]=dims[i]+2*padding[i];
	}
	return out;
}

inline unsigned coordToIndex(const std::vector<unsigned>& coord, const std::vector<unsigned>& dims){
	unsigned result=0;
	for (int i = int(std::min(coord.size(),dims.size()))-1; i >=0 ; --i)
	{
		result*=dims[i];
		result+=coord[i];
	}
	return result;
}

inline std::vector<unsigned> indexToCoord(unsigned index, const std::vector<unsigned>& dims){
	std::vector<unsigned> coord(dims.size(),0);
	for (size_t i = 0; i < dims.size(); ++i)
	{
		coord[i]=index%dims[i];
		index/=dims[i];
	}
	return coord;
}

inline unsigned dataSizeFromDims(const std::vector<unsigned>& dims, unsigned nbVariable){
	unsigned size=nbVariable;
	for (size_t i = 0; i < dims.size(); ++i)
	{
		size*=dims[i];
	}
	return size;
}

inline unsigned mapCellIndexToPadded(unsigned cellIndex,
	const std::vector<unsigned>& originalDims,
	const std::vector<unsigned>& padding,
	const std::vector<unsigned>& paddedDimsArray){
	std::vector<unsigned> coord=indexToCoord(cellIndex,originalDims);
	for (size_t i = 0; i < coord.size(); ++i)
	{
		coord[i]+=padding[i];
	}
	return coordToIndex(coord,paddedDimsArray);
}

inline g2s::DataImage padDataImageWithValue(const g2s::DataImage& input, const std::vector<unsigned>& padding, float fillValue){
	const std::vector<unsigned> outDims=paddedDims(input._dims,padding);
	std::vector<unsigned> outDimsMutable(outDims);
	g2s::DataImage out(outDimsMutable.size(),outDimsMutable.data(),input._nbVariable);
	out._types=input._types;
	out._encodingType=input._encodingType;

	std::fill(out._data,out._data+out.dataSize(),fillValue);

	const unsigned inCellCount=dataSizeFromDims(input._dims,input._nbVariable)/input._nbVariable;
	for (unsigned inCell = 0; inCell < inCellCount; ++inCell)
	{
		const unsigned outCell=mapCellIndexToPadded(inCell,input._dims,padding,outDims);
		memcpy(out._data+outCell*out._nbVariable,input._data+inCell*input._nbVariable,sizeof(float)*out._nbVariable);
	}
	return out;
}

inline g2s::DataImage cropDataImageCenter(const g2s::DataImage& input, const std::vector<unsigned>& padding, const std::vector<unsigned>& outputDims){
	std::vector<unsigned> outputDimsMutable(outputDims);
	g2s::DataImage out(outputDimsMutable.size(),outputDimsMutable.data(),input._nbVariable);
	out._types=input._types;
	out._encodingType=input._encodingType;

	const unsigned outCellCount=out.dataSize()/out._nbVariable;
	for (unsigned outCell = 0; outCell < outCellCount; ++outCell)
	{
		std::vector<unsigned> coord=indexToCoord(outCell,outputDims);
		for (size_t i = 0; i < coord.size(); ++i)
		{
			coord[i]+=padding[i];
		}
		const unsigned inCell=coordToIndex(coord,input._dims);
		memcpy(out._data+outCell*out._nbVariable,input._data+inCell*input._nbVariable,sizeof(float)*out._nbVariable);
	}
	return out;
}

inline void mapSimulationPathToPadded(g2s_path_index_t* path,
	g2s_path_index_t pathSize,
	bool fullSimulation,
	unsigned nbVariable,
	const std::vector<unsigned>& originalDims,
	const std::vector<unsigned>& padding){
	const std::vector<unsigned> outDims=paddedDims(originalDims,padding);
	for (g2s_path_index_t i = 0; i < pathSize; ++i)
	{
		if(fullSimulation){
			const unsigned variable=unsigned(path[i]%nbVariable);
			const unsigned cell=unsigned(path[i]/nbVariable);
			const unsigned paddedCell=mapCellIndexToPadded(cell,originalDims,padding,outDims);
			path[i]=g2s_path_index_t(paddedCell)*g2s_path_index_t(nbVariable)+g2s_path_index_t(variable);
		}else{
			const unsigned paddedCell=mapCellIndexToPadded(unsigned(path[i]),originalDims,padding,outDims);
			path[i]=g2s_path_index_t(paddedCell);
		}
	}
}

} // namespace qs_padding_utils

#endif // QS_PADDING_UTILS_HPP
