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

#ifndef fKB_HPP
#define fKB_HPP

#include <iostream>
#include <cmath>
// __arm64__ is only defined by Apple compilers; Linux/Windows ARM64 use __aarch64__/_M_ARM64.
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
	#ifndef G2S_ARM64
		#define G2S_ARM64 1
	#endif
	#include <arm_neon.h>
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
	#include <immintrin.h>
#endif
#include <limits>
#include <chrono>
#include <algorithm>
#include <numeric>

#define restrict
namespace fKst {

template<typename T>
	inline /*__attribute__((always_inline))*/ void addValueB(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int i){
		short position=k-2;
		while ((position>-1) && (data[i]>output[position]) ){
			output[position+1]=output[position];
			position--;
		}
		output[position+1]=data[i];
	}

template<typename T>
	inline /*__attribute__((always_inline))*/ void addValueB(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, unsigned int i){
		short position=k-2;
		while ((position>-1) && (data[i]>output[position]) ){
			output[position+1]=output[position]; 
			positionValue[position+1]=positionValue[position]; 
			position--;
		}
		output[position+1]=data[i];
		positionValue[position+1]=i;
	}

template<typename T, typename urgT>
	inline /*__attribute__((always_inline))*/ void addValueB(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, urgT generator, unsigned &cpt, unsigned int i){
		short position=k-2;
		short positionLikeLast=k-2;
		while ((positionLikeLast>-1) && (output[k-1]==output[positionLikeLast]) ){
			positionLikeLast--;
		}
		int xes=(k-positionLikeLast-1);
		int x=k-xes+int(floor(generator()*(xes)));

		std::swap(positionValue[x],positionValue[k-1]);


		if(data[i]>output[k-1]){
			if(xes==1)
				cpt=0;
			else
				cpt++;

			while ((position>-1) && (data[i]>output[position]) ){
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


// The scans below visit the values in order and only act on the ones that reach the
// current threshold output[k-1]. That threshold never decreases during a scan, so a block
// in which no value reaches it can be skipped as a whole: the result, and the sequence of
// generator draws, are exactly those of the plain element-by-element loop.
constexpr unsigned int kScanBlock=16;

template<typename T>
inline bool blockHasValueAbove(const T* data, const T threshold){
	bool any=false;
	for (unsigned int j = 0; j < kScanBlock; ++j) any|=(data[j]>threshold);
	return any;
}

template<typename T>
inline bool blockHasValueAtLeast(const T* data, const T threshold){
	bool any=false;
	for (unsigned int j = 0; j < kScanBlock; ++j) any|=(data[j]>=threshold);
	return any;
}

template<typename T>
	inline void findKbiggest(const T* data,const unsigned int N,const unsigned short k, T* restrict output){
	unsigned int i = 0;
	for (; i + kScanBlock <= N; i += kScanBlock)
	{
		if(!blockHasValueAbove(data+i, output[k-1])) continue;
		for (unsigned int j = i; j < i + kScanBlock; ++j)
			if(data[j]>output[k-1]) addValueB(data, N, k, output, j);
	}
	for (; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, i);
		}
	}
}

template<typename T>
inline void findKbiggest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue){
	unsigned int i = 0;
	for (; i + kScanBlock <= N; i += kScanBlock)
	{
		if(!blockHasValueAbove(data+i, output[k-1])) continue;
		for (unsigned int j = i; j < i + kScanBlock; ++j)
			if(data[j]>output[k-1]) addValueB(data, N, k, output, positionValue, j);
	}
	for (; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, i);
		}
	}
}

template<typename T,typename urgT>
inline void findKbiggest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned int i = 0;
	for (; i + kScanBlock <= N; i += kScanBlock)
	{
		if(!blockHasValueAtLeast(data+i, output[k-1])) continue;
		for (unsigned int j = i; j < i + kScanBlock; ++j)
			if(data[j]>=output[k-1]) addValueB(data, N, k, output, positionValue, generator, cpt, j);
	}
	for (; i < N; ++i)
	{
		if(data[i]>=output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}


#if G2S_ARM64
inline void findKbiggestARM(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(float32x4_t)/sizeof(float);

	float32x4_t smallest=vld1q_dup_f32(output+k-1);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		float32x4_t dataVector=vld1q_f32(data+i);

		if(vmaxvq_u32(vcgeq_f32(dataVector,smallest)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
					if(data[j]>output[k-1]) //then change
			{
					addValueB(data, N, k, output, j);
				}
			}
			smallest=vld1q_dup_f32(output+k-1);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, i);
		}
	}
}

inline void findKbiggestARM(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(float32x4_t)/sizeof(float);

	float32x4_t smallest=vld1q_dup_f32(output+k-1);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		float32x4_t dataVector=vld1q_f32(data+i);

		if(vmaxvq_u32(vcgeq_f32(dataVector,smallest)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, j);
				}
			}
			smallest=vld1q_dup_f32(output+k-1);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, i);
		}
	}
}
template <typename urgT>
inline void findKbiggestARM(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(float32x4_t)/sizeof(float);

	float32x4_t smallest=vld1q_dup_f32(output+k-1);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		float32x4_t dataVector=vld1q_f32(data+i);

		if(vmaxvq_u32(vcgeq_f32(dataVector,smallest)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>=output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=vld1q_dup_f32(output+k-1);
		}
    }

    for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
    {
        if(data[i]>=output[k-1]) //then change
        {
        	addValueB(data, N, k, output, positionValue, generator, cpt, i);
        }
    }
}

#endif

#if __SSE4_1__
inline void findKbiggest128(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(__m128)/sizeof(float);

	__m128 smallest=_mm_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m128 dataVector=_mm_loadu_ps(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castps_si128(_mm_cmpgt_ps(dataVector,smallest))))
			{
				for (unsigned int j = i; j < i+ratio; ++j)
			{
					if(data[j]>output[k-1]) //then change
					{
						addValueB(data, N, k, output, j);
					}
				}
			smallest=_mm_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, i);
		}
	}
}

inline void findKbiggest128(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m128)/sizeof(float);

	__m128 smallest=_mm_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m128 dataVector=_mm_loadu_ps(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castps_si128(_mm_cmpgt_ps(dataVector,smallest))))
			{
				for (unsigned int j = i; j < i+ratio; ++j)
			{
					if(data[j]>output[k-1]) //then change
					{
						addValueB(data, N, k, output, positionValue, j);
					}
				}
			smallest=_mm_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKbiggest128(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m128)/sizeof(float);

	__m128 smallest=_mm_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m128 dataVector=_mm_loadu_ps(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castps_si128(_mm_cmpge_ps(dataVector,smallest))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>=output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>=output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

inline void findKbiggest128(const double* data,const unsigned int N,const unsigned short k, double* restrict output){

	unsigned char ratio=sizeof(__m128)/sizeof(double);

	__m128d smallest=_mm_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m128d dataVector=_mm_loadu_pd(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castpd_si128(_mm_cmpgt_pd(dataVector,smallest))))
			{
				for (unsigned int j = i; j < i+ratio; ++j)
			{
					if(data[j]>output[k-1]) //then change
					{
						addValueB(data, N, k, output, j);
					}
				}
			smallest=_mm_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, i);
		}
	}
}

inline void findKbiggest128(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m128)/sizeof(double);

	__m128d smallest=_mm_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m128d dataVector=_mm_loadu_pd(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castpd_si128(_mm_cmpgt_pd(dataVector,smallest))))
			{
				for (unsigned int j = i; j < i+ratio; ++j)
			{
					if(data[j]>output[k-1]) //then change
					{
						addValueB(data, N, k, output, positionValue, j);
					}
				}
			smallest=_mm_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKbiggest128(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m128)/sizeof(double);

	__m128d smallest=_mm_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m128d dataVector=_mm_loadu_pd(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castpd_si128(_mm_cmpge_pd(dataVector,smallest))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>=output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>=output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

#endif

#if __AVX__

inline void findKbiggest256(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(__m256)/sizeof(float);

	__m256 smallest=_mm256_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m256 dataVector=_mm256_loadu_ps(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castps_si256(_mm256_cmp_ps(dataVector,smallest,_CMP_GT_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
			{
					addValueB(data, N, k, output, j);
				}
			}
			smallest=_mm256_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, i);
		}
	}
}

inline void findKbiggest256(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m256)/sizeof(float);

	__m256 smallest=_mm256_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m256 dataVector=_mm256_loadu_ps(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castps_si256(_mm256_cmp_ps(dataVector,smallest,_CMP_GT_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, j);
				}
			}
			smallest=_mm256_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKbiggest256(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m256)/sizeof(float);

	__m256 smallest=_mm256_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m256 dataVector=_mm256_loadu_ps(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castps_si256(_mm256_cmp_ps(dataVector,smallest,_CMP_GE_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>=output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm256_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>=output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

inline void findKbiggest256(const double* data,const unsigned int N,const unsigned short k, double* restrict output){

	unsigned char ratio=sizeof(__m256)/sizeof(double);

	__m256d smallest=_mm256_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m256d dataVector=_mm256_loadu_pd(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castpd_si256(_mm256_cmp_pd(dataVector,smallest,_CMP_GT_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
			{
					addValueB(data, N, k, output, j);
				}
			}
			smallest=_mm256_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, i);
		}
	}
}

inline void findKbiggest256(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m256)/sizeof(double);

	__m256d smallest=_mm256_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m256d dataVector=_mm256_loadu_pd(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castpd_si256(_mm256_cmp_pd(dataVector,smallest,_CMP_GT_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, j);
				}
			}
			smallest=_mm256_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKbiggest256(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m256)/sizeof(double);

	__m256d smallest=_mm256_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m256d dataVector=_mm256_loadu_pd(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castpd_si256(_mm256_cmp_pd(dataVector,smallest,_CMP_GE_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>=output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm256_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>=output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}


#endif

#if __AVX512F__

inline void findKbiggest512(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(__m512)/sizeof(float);

	__m512 smallest=_mm512_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m512 dataVector=_mm512_loadu_ps(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_ps_mask(dataVector,smallest,_CMP_GT_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
			{
					addValueB(data, N, k, output, j);
				}
			}
			smallest=_mm512_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, i);
		}
	}
}

inline void findKbiggest512(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m512)/sizeof(float);

	__m512 smallest=_mm512_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m512 dataVector=_mm512_loadu_ps(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_ps_mask(dataVector,smallest,_CMP_GT_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, j);
				}
			}
			smallest=_mm512_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKbiggest512(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m512)/sizeof(float);

	__m512 smallest=_mm512_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m512 dataVector=_mm512_loadu_ps(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_ps_mask(dataVector,smallest,_CMP_GE_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>=output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm512_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>=output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

inline void findKbiggest512(const double* data,const unsigned int N,const unsigned short k, double* restrict output){

	unsigned char ratio=sizeof(__m512)/sizeof(double);

	__m512d smallest=_mm512_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m512d dataVector=_mm512_loadu_pd(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_pd_mask(dataVector,smallest,_CMP_GT_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
			{
					addValueB(data, N, k, output, j);
				}
			}
			smallest=_mm512_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, i);
		}
	}
}

inline void findKbiggest512(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m512)/sizeof(double);

	__m512d smallest=_mm512_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m512d dataVector=_mm512_loadu_pd(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_pd_mask(dataVector,smallest,_CMP_GT_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, j);
				}
			}
			smallest=_mm512_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, i);
		}
	}
}
template<typename urgT>
inline void findKbiggest512(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m512)/sizeof(double);

	__m512d smallest=_mm512_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( N/ratio)*ratio; i+=ratio)
	{
		__m512d dataVector=_mm512_loadu_pd(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_pd_mask(dataVector,smallest,_CMP_GE_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>=output[k-1]) //then change
			{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm512_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>=output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

#endif


template<typename T>
inline void findKBiggest(const T* data,const unsigned int N,const unsigned short k, T* restrict output){
	std::fill(output,output+k,-INFINITY);

#if __AVX512F__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX512F))
	#endif
	{
		findKbiggest512(data, N, k, output);
		return;
	}
#endif

#if __AVX__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX))
	#endif
	{
		findKbiggest256(data, N, k, output);
		return;
	}
#endif

#if __SSE4_1__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_SSE4_1))
	#endif
	{
		findKbiggest128(data, N, k, output);
		return;
	}
#endif

// #if G2S_ARM64
// 	{
// 		if(std::is_same<T, float>::value){
// 			findKbiggestARM(data, N, k, output);
// 			return;
// 		}
// 	}
// #endif

	findKbiggest<T>(data, N, k, output);
	return;

}

template<typename T>
inline void findKBiggest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue){

	std::fill(output,output+k,-INFINITY);
	std::fill(positionValue,positionValue+k,UINT_MAX);
#if __AVX512F__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX512F))
	#endif
	{
		findKbiggest512(data, N, k, output, positionValue);
		return;
	}
#endif

#if __AVX__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX))
	#endif
	{
		findKbiggest256(data, N, k, output, positionValue);
		return;
	}
#endif

#if __SSE4_1__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_SSE4_1))
	#endif
	{
		findKbiggest128(data, N, k, output, positionValue);
		return;
	}
#endif

// #if G2S_ARM64
// 	{
// 		if(std::is_same<T, float>::value){
// 			findKbiggestARM(data, N, k, output, positionValue);
// 			return;
// 		}
// 	}
// #endif

	findKbiggest<T>(data, N, k, output, positionValue);
	return;

}

template<typename T, typename urgT>
inline void findKBiggest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, urgT generator){

	std::fill(output,output+k,-INFINITY);
	std::fill(positionValue,positionValue+k,UINT_MAX);
#if __AVX512F__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX512F))
	#endif
	{
		findKbiggest512(data, N, k, output, positionValue, generator);
		return;
	}
#endif

#if __AVX__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX))
	#endif
	{
		findKbiggest256(data, N, k, output, positionValue, generator);
		return;
	}
#endif

#if __SSE4_1__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_SSE4_1))
	#endif
	{
		findKbiggest128(data, N, k, output, positionValue, generator);
		return;
	}
#endif

// #if G2S_ARM64
// 	{
// 		if(std::is_same<T, float>::value){
// 			findKbiggestARM(data, N, k, output, positionValue, generator);
// 			return;
// 		}
// 	}
// #endif

	findKbiggest<T>(data, N, k, output, positionValue, generator);
	return;

}

}
#endif
