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
#if __arm64__
	#include <arm_neon.h>
#else
	#include <immintrin.h>
#endif
#include <limits>
#include <chrono>
#include <algorithm>
#include <numeric>

#define restrict
namespace fKst {

template<typename T>
	inline __attribute__((always_inline)) void addValueB(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int i){
		short position=k-2;
		while ((data[i]>output[position]) && (position>-1) ){
			output[position+1]=output[position];
			position--;
		}
		output[position+1]=data[i];
	}

template<typename T>
	inline __attribute__((always_inline)) void addValueB(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, unsigned int i){
		short position=k-2;
		while ((data[i]>output[position]) && (position>-1) ){
			output[position+1]=output[position]; 
			positionValue[position+1]=positionValue[position]; 
			position--;
		}
		output[position+1]=data[i];
		positionValue[position+1]=i;
	}

template<typename T, typename urgT>
	inline __attribute__((always_inline)) void addValueB(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, urgT generator, unsigned &cpt, unsigned int i){
		short position=k-2;
		short positionLikeLast=k-2;
		while ((output[k-1]==output[positionLikeLast]) && (position>-1) ){
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

			while ((data[i]>output[position]) && (position>-1) ){
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
	inline void findKbigest(const T* data,const unsigned int N,const unsigned short k, T* restrict output){

		for (unsigned int i = 0; i < N; ++i)
		{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, i);
		}
	}

}

template<typename T>
inline void findKbigest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue){

	for (unsigned int i = 0; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, i);
		}
	}

}

template<typename T,typename urgT>
inline void findKbigest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	for (unsigned int i = 0; i < N; ++i)
	{
		if(data[i]>=output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}

}


#if __arm64__
template <int len>
inline void findKbigestARM(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(float32x4_t)/sizeof(float);

	float32x4_t smallest=vld1q_dup_f32(output+k-1);
	for (unsigned int i = 0; i < ( (N-len)/ratio)*ratio; i+=ratio)
	{
		float32x4_t dataVector[len];
		for(int j = 0; j < len; j++){
			dataVector[j] = vld1q_f32(data+i+j);
		}

		int mask = vminvq_u32(vreinterpretq_u32_f32(vcgtq_f32(dataVector[0],smallest)));
		for(int j = 1; j < len; j++){
			mask &= vminvq_u32(vreinterpretq_u32_f32(vcgtq_f32(dataVector[j],smallest)));
		}

		if(mask == 0)
		{
			for (unsigned int j = i; j < i+len; ++j)
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
template <int len>
inline void findKbigestARM(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(float32x4_t)/sizeof(float);

	float32x4_t smallest=vld1q_dup_f32(output+k-1);
	for (unsigned int i = 0; i < ( (N-len)/ratio)*ratio; i+=ratio)
	{
		float32x4_t dataVector[len];
		for(int j = 0; j < len; j++){
			dataVector[j] = vld1q_f32(data+i+j);
		}

		int mask = vminvq_u32(vreinterpretq_u32_f32(vcgtq_f32(dataVector[0],smallest)));
		for(int j = 1; j < len; j++){
			mask &= vminvq_u32(vreinterpretq_u32_f32(vcgtq_f32(dataVector[j],smallest)));
		}

		if(mask == 0)
		{
			for (unsigned int j = i; j < i+len; ++j)
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
template <int len, typename urgT>
inline void findKbigestARM(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(float32x4_t)/sizeof(float);

	float32x4_t smallest=vld1q_dup_f32(output+k-1);
	for (unsigned int i = 0; i < ( (N-len)/ratio)*ratio; i+=ratio)
	{
		float32x4_t dataVector[len];
		for(int j = 0; j < len; j++){
			dataVector[j] = vld1q_f32(data+i+j);
		}

		int mask = vminvq_u32(vreinterpretq_u32_f32(vcgeq_f32(dataVector[0],smallest)));
		for(int j = 1; j < len; j++){
			mask &= vminvq_u32(vreinterpretq_u32_f32(vcgeq_f32(dataVector[j],smallest)));
		}

		if(mask == 0)
		{
			for (unsigned int j = i; j < i+len; ++j)
			{
                if(data[j]>output[k-1]) //then change
                {
                	addValueB(data, N, k, output, positionValue, generator, cpt, j);
                }
            }
            smallest=vld1q_dup_f32(output+k-1);
        }
        
    }

    for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
    {
        if(data[i]>output[k-1]) //then change
        {
        	addValueB(data, N, k, output, positionValue, generator, cpt, i);
        }
    }
}

#endif

#if __SSE4_1__
inline void findKbigest128(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(__m128)/sizeof(float);

	__m128 smallest=_mm_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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

inline void findKbigest128(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m128)/sizeof(float);

	__m128 smallest=_mm_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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
inline void findKbigest128(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m128)/sizeof(float);

	__m128 smallest=_mm_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m128 dataVector=_mm_loadu_ps(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castps_si128(_mm_cmpge_ps(dataVector,smallest))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
				{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

inline void findKbigest128(const double* data,const unsigned int N,const unsigned short k, double* restrict output){

	unsigned char ratio=sizeof(__m128)/sizeof(double);

	__m128d smallest=_mm_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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

inline void findKbigest128(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m128)/sizeof(double);

	__m128d smallest=_mm_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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
inline void findKbigest128(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m128)/sizeof(double);

	__m128d smallest=_mm_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m128d dataVector=_mm_loadu_pd(data+i);

		if(!_mm_testc_si128(_mm_set1_epi8(0),_mm_castpd_si128(_mm_cmpge_pd(dataVector,smallest))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
				{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

#endif

#if __AVX__

inline void findKbigest256(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(__m256)/sizeof(float);

	__m256 smallest=_mm256_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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

inline void findKbigest256(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m256)/sizeof(float);

	__m256 smallest=_mm256_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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
inline void findKbigest256(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m256)/sizeof(float);

	__m256 smallest=_mm256_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m256 dataVector=_mm256_loadu_ps(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castps_si256(_mm256_cmp_ps(dataVector,smallest,_CMP_GE_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
				{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm256_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

inline void findKbigest256(const double* data,const unsigned int N,const unsigned short k, double* restrict output){

	unsigned char ratio=sizeof(__m256)/sizeof(double);

	__m256d smallest=_mm256_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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

inline void findKbigest256(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m256)/sizeof(double);

	__m256d smallest=_mm256_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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
inline void findKbigest256(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m256)/sizeof(double);

	__m256d smallest=_mm256_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m256d dataVector=_mm256_loadu_pd(data+i);
		if(!_mm256_testc_si256(_mm256_set1_epi8(0),_mm256_castpd_si256(_mm256_cmp_pd(dataVector,smallest,_CMP_GE_OQ))))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
				{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm256_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}


#endif

#if __AVX512F__

inline void findKbigest512(const float* data,const unsigned int N,const unsigned short k, float* restrict output){

	unsigned char ratio=sizeof(__m512)/sizeof(float);

	__m512 smallest=_mm512_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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

inline void findKbigest512(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m512)/sizeof(float);

	__m512 smallest=_mm512_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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
inline void findKbigest512(const float* data,const unsigned int N,const unsigned short k, float* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m512)/sizeof(float);

	__m512 smallest=_mm512_set1_ps(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m512 dataVector=_mm512_loadu_ps(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_ps_mask(dataVector,smallest,_CMP_GE_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
				{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm512_set1_ps(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

inline void findKbigest512(const double* data,const unsigned int N,const unsigned short k, double* restrict output){

	unsigned char ratio=sizeof(__m512)/sizeof(double);

	__m512d smallest=_mm512_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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

inline void findKbigest512(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue){

	unsigned char ratio=sizeof(__m512)/sizeof(double);

	__m512d smallest=_mm512_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
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
inline void findKbigest512(const double* data,const unsigned int N,const unsigned short k, double* restrict output, unsigned int* restrict positionValue, urgT generator){
	unsigned cpt=0;
	unsigned char ratio=sizeof(__m512)/sizeof(double);

	__m512d smallest=_mm512_set1_pd(output[k-1]);
	for (unsigned int i = 0; i < ( (N-1)/ratio)*ratio; i+=ratio)
	{
		__m512d dataVector=_mm512_loadu_pd(data+i);	
		if(!_mm512_kortestz(_mm512_int2mask(0),_mm512_cmp_pd_mask(dataVector,smallest,_CMP_GE_OQ)))
		{
			for (unsigned int j = i; j < i+ratio; ++j)
			{
				if(data[j]>output[k-1]) //then change
				{
					addValueB(data, N, k, output, positionValue, generator, cpt, j);
				}
			}
			smallest=_mm512_set1_pd(output[k-1]);
		}
		
	}

	for (unsigned int i = ( N/ratio)*ratio; i < N; ++i)
	{
		if(data[i]>output[k-1]) //then change
		{
			addValueB(data, N, k, output, positionValue, generator, cpt, i);
		}
	}
}

#endif


template<typename T>
inline void findKBigest(const T* data,const unsigned int N,const unsigned short k, T* restrict output){
	std::fill(output,output+k,-INFINITY);

#if __AVX512F__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX512F))
	#endif
	{
		findKBigest512(data, N, k, output);
		return;
	}
#endif

#if __AVX__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX))
	#endif
	{
		findKbigest256(data, N, k, output);
		return;
	}
#endif

#if __SSE4_1__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_SSE4_1))
	#endif
	{
		findKbigest128(data, N, k, output);
		return;
	}
#endif

#if __arm64__
	{
		if(std::is_same<T, float>::value){
			findKbigestARM<16>(data, N, k, output);
			return;
		}
	}
#endif

	findKbigest<T>(data, N, k, output);
	return;

}

template<typename T>
inline void findKBigest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue){

	std::fill(output,output+k,-INFINITY);
	std::fill(positionValue,positionValue+k,UINT_MAX);
#if __AVX512F__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX512F))
	#endif
	{
		findKbigest512(data, N, k, output, positionValue);
		return;
	}
#endif

#if __AVX__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX))
	#endif
	{
		findKbigest256(data, N, k, output, positionValue);
		return;
	}
#endif

#if __SSE4_1__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_SSE4_1))
	#endif
	{
		findKbigest128(data, N, k, output, positionValue);
		return;
	}
#endif

#if __arm64__
	{
		if(std::is_same<T, float>::value){
			findKbigestARM<16>(data, N, k, output, positionValue);
			return;
		}
	}
#endif

	findKbigest<T>(data, N, k, output, positionValue);
	return;

}

template<typename T, typename urgT>
inline void findKBigest(const T* data,const unsigned int N,const unsigned short k, T* restrict output, unsigned int* restrict positionValue, urgT generator){

	std::fill(output,output+k,-INFINITY);
	std::fill(positionValue,positionValue+k,UINT_MAX);
#if __AVX512F__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX512F))
	#endif
	{
		findKbigest512(data, N, k, output, positionValue, generator);
		return;
	}
#endif

#if __AVX__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_AVX))
	#endif
	{
		findKbigest256(data, N, k, output, positionValue, generator);
		return;
	}
#endif

#if __SSE4_1__
	#if __INTEL_COMPILER
	if(_may_i_use_cpu_feature(_FEATURE_SSE4_1))
	#endif
	{
		findKbigest128(data, N, k, output, positionValue, generator);
		return;
	}
#endif

#if __arm64__
	{
		if(std::is_same<T, float>::value){
			findKbigestARM<16>(data, N, k, output, positionValue, generator);
			return;
		}
	}
#endif

	findKbigest<T>(data, N, k, output, positionValue, generator);
	return;

}

}
#endif
