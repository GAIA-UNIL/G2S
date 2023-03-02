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

#ifndef fKS_HPP
#define fKS_HPP

#include <iostream>
#include <cmath>
#if __arm64__
	#include <arm_neon.h>
#else
	#include <immintrin.h>
#endif
#include <limits>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>

#define restrict
namespace fKst {

template<typename T>
inline __attribute__((always_inline)) void addValueS(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int i){
	short position=k-2;
	while ((data[i]<output[position]) && (position>-1) ){
		output[position+1]=output[position];
		position--;
	}
	output[position+1]=data[i];
}

template<typename T>
inline __attribute__((always_inline)) void addValueS(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, unsigned int i){
	short position=k-2;
	while ((data[i]<output[position]) && (position>-1) ){
		output[position+1]=output[position]; 
		positionValue[position+1]=positionValue[position]; 
		position--;
	}
	output[position+1]=data[i];
	positionValue[position+1]=i;
}

template<typename T, typename urgT>
inline __attribute__((always_inline)) void addValueS(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, urgT generator, unsigned &cpt, unsigned int i){
	short position=k-2;
	short positionLikeLast=k-2;
	while ((output[k-1]==output[positionLikeLast]) && (position>-1) ){
		positionLikeLast--;
	}
	int xes=(k-positionLikeLast-1);
	int x=k-xes+int(floor(generator()*(xes)));

	std::swap(positionValue[x],positionValue[k-1]);


	if(data[i]<output[k-1]){
		if(xes==1)
			cpt=0;
		else
			cpt++;

		while ((data[i]<output[position]) && (position>-1) ){
			output[position+1]=output[position]; 
			positionValue[position+1]=positionValue[position]; 
			position--;
		}
		output[position+1]=data[i];
		positionValue[position+1]=i;

	}else{
		cpt++;
		if((xes+cpt)*generator()<xes){
			positionValue[position+1]=i;
		}
	}

}

template<typename T>
inline void findKsmallest(const T* data,const unsigned int N,const unsigned short k, T* restrict output){

	for (unsigned int i = 0; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, i);
		}
	}
}

template<typename T>
inline void findKsmallest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue){

	for (unsigned int i = 0; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, i);
		}
	}

}

template<typename T, typename urgT>
inline void findKsmallest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;

	for (unsigned int i = 0; i < N; ++i)
	{
		if(data[i]<=output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, generator, cpt, i);
		}
	}

}

#if __SSE4_1__
inline void findKsmallest128(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(__m128)/sizeof(float);

	__m128 bigest=_mm_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m128 dataVector=_mm_loadu_ps(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castps_si128(_mm_cmplt_ps(dataVector,bigest))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, j);
				}
			}
			bigest=_mm_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, i);
		}
	}
}

inline void findKsmallest128(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m128)/sizeof(float);

	__m128 bigest=_mm_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m128 dataVector=_mm_loadu_ps(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castps_si128(_mm_cmplt_ps(dataVector,bigest))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, j);
				}
			}
			bigest=_mm_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKsmallest128(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){

	unsigned cpt=0;

	unsigned char ratio=sizeof(__m128)/sizeof(float);

	__m128 bigest=_mm_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m128 dataVector=_mm_loadu_ps(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castps_si128(_mm_cmple_ps(dataVector,bigest))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<=output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			bigest=_mm_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<=output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}


inline void findKsmallest128(const double* data,const unsigned int N,const unsigned short k, double* restrict output){

	unsigned char ratio=sizeof(__m128)/sizeof(double);

	__m128d bigest=_mm_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m128d dataVector=_mm_loadu_pd(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castpd_si128(_mm_cmplt_pd(dataVector,bigest))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, j);
				}
			}
			bigest=_mm_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, i);
		}
	}
}

inline void findKsmallest128(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m128)/sizeof(double);

	__m128d bigest=_mm_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m128d dataVector=_mm_loadu_pd(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castpd_si128(_mm_cmplt_pd(dataVector,bigest))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, j);
				}
			}
			bigest=_mm_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKsmallest128(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue, urgT generator){

	unsigned cpt=0;

	unsigned char ratio=sizeof(__m128)/sizeof(double);

	__m128d bigest=_mm_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m128d dataVector=_mm_loadu_pd(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castpd_si128(_mm_cmple_pd(dataVector,bigest))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<=output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			bigest=_mm_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<=output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}


#endif

#if __AVX__

inline void findKsmallest256(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(__m256)/sizeof(float);

	__m256 bigest=_mm256_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m256 dataVector=_mm256_loadu_ps(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castps_si256(_mm256_cmp_ps(dataVector,bigest,_CMP_LT_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, j);
				}
			}
			bigest=_mm256_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, i);
		}
	}
}

inline void findKsmallest256(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m256)/sizeof(float);

	__m256 bigest=_mm256_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m256 dataVector=_mm256_loadu_ps(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castps_si256(_mm256_cmp_ps(dataVector,bigest,_CMP_LT_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, j);
				}
			}
			bigest=_mm256_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKsmallest256(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){

	unsigned cpt=0;

	unsigned char ratio=sizeof(__m256)/sizeof(float);

	__m256 bigest=_mm256_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m256 dataVector=_mm256_loadu_ps(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castps_si256(_mm256_cmp_ps(dataVector,bigest,_CMP_LE_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<=output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			bigest=_mm256_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<=output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}


inline void findKsmallest256(const double* data,const unsigned int N,const unsigned short k, double* restrict output){

	unsigned char ratio=sizeof(__m256)/sizeof(double);

	__m256d bigest=_mm256_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m256d dataVector=_mm256_loadu_pd(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castpd_si256(_mm256_cmp_pd(dataVector,bigest,_CMP_LT_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, j);
				}
			}
			bigest=_mm256_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, i);
		}
	}
}

inline void findKsmallest256(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m256)/sizeof(double);

	__m256d bigest=_mm256_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m256d dataVector=_mm256_loadu_pd(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castpd_si256(_mm256_cmp_pd(dataVector,bigest,_CMP_LT_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, j);
				}
			}
			bigest=_mm256_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, i);
		}
	}
}

template<typename urgT>
inline void findKsmallest256(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue, urgT generator){

	unsigned cpt=0;

	unsigned char ratio=sizeof(__m256)/sizeof(double);

	__m256d bigest=_mm256_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m256d dataVector=_mm256_loadu_pd(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castpd_si256(_mm256_cmp_pd(dataVector,bigest,_CMP_LE_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<=output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			bigest=_mm256_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<=output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}



#endif

#if __AVX512F__

inline void findKsmallest512(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(__m512)/sizeof(float);

	__m512 bigest=_mm512_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m512 dataVector=_mm512_loadu_ps(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_ps_mask(dataVector,bigest,_CMP_LT_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, j);
				}
			}
			bigest=_mm512_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, i);
		}
	}
}

inline void findKsmallest512(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m512)/sizeof(float);

	__m512 bigest=_mm512_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m512 dataVector=_mm512_loadu_ps(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_ps_mask(dataVector,bigest,_CMP_LT_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, j);
				}
			}
			bigest=_mm512_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKsmallest512(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m512)/sizeof(float);

	__m512 bigest=_mm512_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m512 dataVector=_mm512_loadu_ps(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_ps_mask(dataVector,bigest,_CMP_LE_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<=output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			bigest=_mm512_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<=output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

inline void findKsmallest512(const double* data,const unsigned int N,const unsigned short k, double* restrict output){

	unsigned char ratio=sizeof(__m512)/sizeof(double);

	__m512d bigest=_mm512_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m512d dataVector=_mm512_loadu_pd(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_pd_mask(dataVector,bigest,_CMP_LT_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, j);
				}
			}
			bigest=_mm512_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, i);
		}
	}
}

inline void findKsmallest512(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m512)/sizeof(double);

	__m512d bigest=_mm512_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m512d dataVector=_mm512_loadu_pd(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_pd_mask(dataVector,bigest,_CMP_LT_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, j);
				}
			}
			bigest=_mm512_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKsmallest512(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m512)/sizeof(double);

	__m512d bigest=_mm512_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m512d dataVector=_mm512_loadu_pd(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_pd_mask(dataVector,bigest,_CMP_LE_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]<=output[k-1]) //then change
				{
					addValueS(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			bigest=_mm512_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]<=output[k-1]) //then change
		{
			addValueS(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}


#endif


template<typename T>
inline void findKSmallest(const T* data,const unsigned int N,const unsigned short k, T* restrict output){
	std::fill(output,output+k,INFINITY);
#if __AVX512F__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX512F))
	#endif
	{
		findKsmallest512(data, N, k, output);
		return;
	}
#endif

#if __AVX__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX))
	#endif
	{
		findKsmallest256(data, N, k, output);
		return;
	}
#endif

#if __SSE4_1__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_SSE4_1))
	#endif
	{
		findKsmallest128(data, N, k, output);
		return;
	}
#endif

	findKsmallest<T>(data, N, k, output);
	return;

}
template<typename T>
inline void findKSmallest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue){
	std::fill(output,output+k,INFINITY);
	std::fill(positionValue,positionValue+k,UINT_MAX);

#if __AVX512F__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX512F))
	#endif
	{
		findKsmallest512(data, N, k, output, positionValue);
		return;
	}
#endif

#if __AVX__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX))
	#endif
	{
		findKsmallest256(data, N, k, output, positionValue);
		return;
	}
#endif

#if __SSE4_1__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_SSE4_1))
	#endif
	{
		findKsmallest128(data, N, k, output, positionValue);
		return;
	}
#endif

	findKsmallest<T>(data, N, k, output, positionValue);
	return;

}

template<typename T, typename urgT>
inline void findKSmallest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, urgT generator){
	std::fill(output,output+k,INFINITY);
	std::fill(positionValue,positionValue+k,UINT_MAX);

#if __AVX512F__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX512F))
	#endif
	{
		findKsmallest512(data, N, k, output, positionValue, generator);
		return;
	}
#endif

#if __AVX__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX))
	#endif
	{
		findKsmallest256(data, N, k, output, positionValue, generator);
		return;
	}
#endif

#if __SSE4_1__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_SSE4_1))
	#endif
	{
		findKsmallest128(data, N, k, output, positionValue, generator);
		return;
	}
#endif

	findKsmallest<T>(data, N, k, output, positionValue, generator);
	return;

}

}
#endif
