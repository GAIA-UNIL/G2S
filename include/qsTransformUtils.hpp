#ifndef QS_TRANSFORM_UTILS_HPP
#define QS_TRANSFORM_UTILS_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "DataImage.hpp"

namespace qs_transform_utils {

inline uint32_t floatBits(float value){
	uint32_t bits=0;
	std::memcpy(&bits,&value,sizeof(float));
	return bits;
}

struct TransformContext{
	g2s::DataImage* rotationMap=nullptr;
	g2s::DataImage* scaleMap=nullptr;
	unsigned rank=0;

	bool enabled() const{
		return rotationMap!=nullptr || scaleMap!=nullptr;
	}
};

struct NodeTransform{
	float scale=1.f;
	float rotation[4]={0.f,0.f,0.f,1.f};
	bool hasRotation=false;
	bool identity=true;
};

struct TransformKey{
	int kernelIndex=-1;
	unsigned rank=0;
	uint32_t values[5]={0u,0u,0u,0u,0u};

	bool equals(const TransformKey& other) const{
		if(kernelIndex!=other.kernelIndex || rank!=other.rank){
			return false;
		}
		for (int i = 0; i < 5; ++i)
		{
			if(values[i]!=other.values[i]){
				return false;
			}
		}
		return true;
	}
};

struct TransformCacheEntry{
	bool valid=false;
	TransformKey key;
	std::vector<std::vector<int> > transformedPath;
};

struct ThreadTransformCache{
	std::vector<TransformCacheEntry> entries;
};

inline float mapValue(const g2s::DataImage* image, unsigned cellIndex, unsigned variableIndex){
	if(image==nullptr || variableIndex>=image->_nbVariable){
		return std::nanf("0");
	}
	return image->_data[cellIndex*image->_nbVariable+variableIndex];
}

inline NodeTransform readNodeTransform(const TransformContext& context, unsigned cellIndex, TransformKey& key){
	NodeTransform transform;
	key.rank=context.rank;
	key.values[0]=floatBits(transform.scale);
	for (int i = 0; i < 4; ++i)
	{
		key.values[i+1]=floatBits(transform.rotation[i]);
	}

	if(context.scaleMap!=nullptr){
		const float scale=mapValue(context.scaleMap,cellIndex,0);
		if(std::isfinite(scale) && scale>0.f){
			transform.scale=scale;
		}
	}

	if(context.rotationMap!=nullptr && context.rank==2){
		const float angle=mapValue(context.rotationMap,cellIndex,0);
		if(std::isfinite(angle)){
			transform.rotation[0]=angle;
			transform.hasRotation=true;
		}
	}else if(context.rotationMap!=nullptr && context.rank==3){
		const float qx=mapValue(context.rotationMap,cellIndex,0);
		const float qy=mapValue(context.rotationMap,cellIndex,1);
		const float qz=mapValue(context.rotationMap,cellIndex,2);
		const float qw=mapValue(context.rotationMap,cellIndex,3);
		const double norm2=double(qx)*double(qx)+double(qy)*double(qy)+double(qz)*double(qz)+double(qw)*double(qw);
		if(std::isfinite(qx) && std::isfinite(qy) && std::isfinite(qz) && std::isfinite(qw) && norm2>1.0e-20){
			const double invNorm=1.0/std::sqrt(norm2);
			transform.rotation[0]=float(double(qx)*invNorm);
			transform.rotation[1]=float(double(qy)*invNorm);
			transform.rotation[2]=float(double(qz)*invNorm);
			transform.rotation[3]=float(double(qw)*invNorm);
			transform.hasRotation=true;
		}
	}

	transform.identity=(transform.scale==1.f);
	if(context.rank==2){
		transform.identity&=(!transform.hasRotation || transform.rotation[0]==0.f);
	}else if(context.rank==3){
		transform.identity&=(!transform.hasRotation ||
			(transform.rotation[0]==0.f && transform.rotation[1]==0.f && transform.rotation[2]==0.f && transform.rotation[3]==1.f));
	}

	key.values[0]=floatBits(transform.scale);
	for (int i = 0; i < 4; ++i)
	{
		key.values[i+1]=floatBits(transform.rotation[i]);
	}
	return transform;
}

inline int roundedOffset(double value){
	return static_cast<int>(std::lround(value));
}

inline std::vector<int> transformOffset(const std::vector<int>& input, unsigned rank, const NodeTransform& transform){
	std::vector<int> output(rank,0);
	const double scale=double(transform.scale);
	if(rank==2){
		const double x=(input.size()>0 ? double(input[0]) : 0.0)*scale;
		const double y=(input.size()>1 ? double(input[1]) : 0.0)*scale;
		if(transform.hasRotation){
			const double c=std::cos(double(transform.rotation[0]));
			const double s=std::sin(double(transform.rotation[0]));
			output[0]=roundedOffset(c*x-s*y);
			output[1]=roundedOffset(s*x+c*y);
		}else{
			output[0]=roundedOffset(x);
			output[1]=roundedOffset(y);
		}
		return output;
	}

	const double x=(input.size()>0 ? double(input[0]) : 0.0)*scale;
	const double y=(input.size()>1 ? double(input[1]) : 0.0)*scale;
	const double z=(input.size()>2 ? double(input[2]) : 0.0)*scale;
	if(rank==3 && transform.hasRotation){
		const double qx=transform.rotation[0];
		const double qy=transform.rotation[1];
		const double qz=transform.rotation[2];
		const double qw=transform.rotation[3];
		const double xx=qx*qx;
		const double yy=qy*qy;
		const double zz=qz*qz;
		const double xy=qx*qy;
		const double xz=qx*qz;
		const double yz=qy*qz;
		const double wx=qw*qx;
		const double wy=qw*qy;
		const double wz=qw*qz;

		output[0]=roundedOffset((1.0-2.0*(yy+zz))*x + 2.0*(xy-wz)*y + 2.0*(xz+wy)*z);
		output[1]=roundedOffset(2.0*(xy+wz)*x + (1.0-2.0*(xx+zz))*y + 2.0*(yz-wx)*z);
		output[2]=roundedOffset(2.0*(xz-wy)*x + 2.0*(yz+wx)*y + (1.0-2.0*(xx+yy))*z);
	}else{
		output[0]=roundedOffset(x);
		output[1]=roundedOffset(y);
		output[2]=roundedOffset(z);
	}
	return output;
}

inline const std::vector<std::vector<int> >* effectivePath(
	const TransformContext* context,
	ThreadTransformCache& cache,
	const std::vector<std::vector<int> >& basePath,
	int kernelIndex,
	unsigned cellIndex){
	if(context==nullptr || !context->enabled()){
		return &basePath;
	}

	TransformKey key;
	key.kernelIndex=kernelIndex;
	const NodeTransform transform=readNodeTransform(*context,cellIndex,key);
	if(transform.identity){
		return &basePath;
	}

	const size_t cacheIndex=static_cast<size_t>(kernelIndex<0 ? 0 : kernelIndex);
	if(cache.entries.size()<=cacheIndex){
		cache.entries.resize(cacheIndex+1);
	}
	TransformCacheEntry& entry=cache.entries[cacheIndex];
	if(entry.valid && entry.key.equals(key)){
		return &entry.transformedPath;
	}

	entry.key=key;
	entry.valid=true;
	entry.transformedPath.clear();
	entry.transformedPath.reserve(basePath.size());
	for (size_t i = 0; i < basePath.size(); ++i)
	{
		entry.transformedPath.push_back(transformOffset(basePath[i],context->rank,transform));
	}
	return &entry.transformedPath;
}

} // namespace qs_transform_utils

#endif
