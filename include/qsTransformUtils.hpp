#ifndef QS_TRANSFORM_UTILS_HPP
#define QS_TRANSFORM_UTILS_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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
	g2s::DataImage* rotationToleranceMap=nullptr;
	g2s::DataImage* scaleMap=nullptr;
	g2s::DataImage* scaleToleranceMap=nullptr;
	unsigned rank=0;
	unsigned globalSeed=0;

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

struct EffectivePath{
	const std::vector<std::vector<int> >* simulationPath=nullptr;
	const std::vector<std::vector<int> >* matchingPath=nullptr;
};

inline float mapValue(const g2s::DataImage* image, unsigned cellIndex, unsigned variableIndex){
	if(image==nullptr || variableIndex>=image->_nbVariable){
		return std::nanf("0");
	}
	return image->_data[cellIndex*image->_nbVariable+variableIndex];
}

inline uint64_t mix64(uint64_t value){
	value+=0x9e3779b97f4a7c15ULL;
	value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
	value=(value^(value>>27))*0x94d049bb133111ebULL;
	return value^(value>>31);
}

inline float hashUnit(uint64_t a, uint64_t b, uint64_t c, uint64_t d){
	const uint64_t value=mix64(a^(mix64(b)+0x9e3779b97f4a7c15ULL+(a<<6)+(a>>2))^mix64(c)^mix64(d));
	const uint32_t top=static_cast<uint32_t>(value>>40);
	return float(double(top)/double(0x1000000u));
}

inline float drawUniform(float center, float tolerance, const TransformContext& context, uint64_t pathIndex, unsigned variableIndex, unsigned component){
	if(!std::isfinite(tolerance) || tolerance<=0.f){
		return center;
	}
	const float u=hashUnit(context.globalSeed,pathIndex,variableIndex,component);
	return center + (2.f*u-1.f)*tolerance;
}

inline float quantize(float value, float step){
	if(!std::isfinite(value) || step<=0.f){
		return value;
	}
	return std::round(value/step)*step;
}

inline NodeTransform readNodeTransform(const TransformContext& context, unsigned cellIndex, TransformKey& key, uint64_t pathIndex=0, unsigned variableIndex=0){
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
			float logScale=std::log(scale);
			if(context.scaleToleranceMap!=nullptr){
				const float tolerance=mapValue(context.scaleToleranceMap,cellIndex,0);
				if(std::isfinite(tolerance) && tolerance>0.f){
					logScale=drawUniform(logScale,tolerance,context,pathIndex,variableIndex,5);
					logScale=quantize(logScale,0.05f);
				}
			}
			transform.scale=std::exp(logScale);
		}
	}

	if(context.rotationMap!=nullptr && context.rank==2){
		const float angle=mapValue(context.rotationMap,cellIndex,0);
		if(std::isfinite(angle)){
			float sampledAngle=angle;
			if(context.rotationToleranceMap!=nullptr){
				const float tolerance=mapValue(context.rotationToleranceMap,cellIndex,0);
				if(std::isfinite(tolerance) && tolerance>0.f){
					sampledAngle=drawUniform(angle,tolerance,context,pathIndex,variableIndex,0);
					const float degree=3.14159265358979323846f/180.f;
					sampledAngle=quantize(sampledAngle,degree);
				}
			}
			transform.rotation[0]=sampledAngle;
			transform.hasRotation=true;
		}
	}else if(context.rotationMap!=nullptr && context.rank==3){
		float qx=mapValue(context.rotationMap,cellIndex,0);
		float qy=mapValue(context.rotationMap,cellIndex,1);
		float qz=mapValue(context.rotationMap,cellIndex,2);
		float qw=mapValue(context.rotationMap,cellIndex,3);
		if(context.rotationToleranceMap!=nullptr){
			const float tx=mapValue(context.rotationToleranceMap,cellIndex,0);
			const float ty=mapValue(context.rotationToleranceMap,cellIndex,1);
			const float tz=mapValue(context.rotationToleranceMap,cellIndex,2);
			const float tw=mapValue(context.rotationToleranceMap,cellIndex,3);
			if(std::isfinite(tx) && tx>0.f) qx=drawUniform(qx,tx,context,pathIndex,variableIndex,0);
			if(std::isfinite(ty) && ty>0.f) qy=drawUniform(qy,ty,context,pathIndex,variableIndex,1);
			if(std::isfinite(tz) && tz>0.f) qz=drawUniform(qz,tz,context,pathIndex,variableIndex,2);
			if(std::isfinite(tw) && tw>0.f) qw=drawUniform(qw,tw,context,pathIndex,variableIndex,3);
			const float qStep=0.01f;
			qx=quantize(qx,qStep);
			qy=quantize(qy,qStep);
			qz=quantize(qz,qStep);
			qw=quantize(qw,qStep);
		}
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

inline EffectivePath effectivePath(
	const TransformContext* context,
	ThreadTransformCache& cache,
	const std::vector<std::vector<int> >& basePath,
	int kernelIndex,
	unsigned cellIndex,
	uint64_t pathIndex=0,
	unsigned variableIndex=0){
	if(context==nullptr || !context->enabled()){
		return EffectivePath{&basePath,&basePath};
	}

	TransformKey key;
	key.kernelIndex=kernelIndex;
	const NodeTransform transform=readNodeTransform(*context,cellIndex,key,pathIndex,variableIndex);
	if(transform.identity){
		return EffectivePath{&basePath,&basePath};
	}

	const size_t cacheIndex=static_cast<size_t>(kernelIndex<0 ? 0 : kernelIndex);
	if(cache.entries.size()<=cacheIndex){
		cache.entries.resize(cacheIndex+1);
	}
	TransformCacheEntry& entry=cache.entries[cacheIndex];
	if(entry.valid && entry.key.equals(key)){
		return EffectivePath{&entry.transformedPath,&basePath};
	}

	entry.key=key;
	entry.valid=true;
	entry.transformedPath.clear();
	entry.transformedPath.reserve(basePath.size());
	for (size_t i = 0; i < basePath.size(); ++i)
	{
		entry.transformedPath.push_back(transformOffset(basePath[i],context->rank,transform));
	}
	return EffectivePath{&entry.transformedPath,&basePath};
}

} // namespace qs_transform_utils

#endif
