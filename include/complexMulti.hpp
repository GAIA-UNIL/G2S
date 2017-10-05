/*
 * Mathieu Gravey
 * Copyright (C) 2017 Mathieu Gravey
 * 
 * This program is protected software: you can not redistribute, use, and/or modify it
 * without the explicit accord from the author : Mathieu Gravey, gravey.mathieu@gmail.com
 *
 */
#ifndef COMPLEX_MULI_HPP
#define COMPLEX_MULI_HPP

#include <iostream>
#include <cmath>
#include <immintrin.h>
#include <limits>
#include <chrono>
#include <algorithm>

#	ifndef _mm512_moveldup_pd
#       define _mm512_moveldup_pd(a) _mm512_permute_pd(a, 0)
#   endif
#   ifndef _mm512_movehdup_pd
#       define _mm512_movehdup_pd(a) _mm512_permute_pd(a, 255)
#   endif

#	ifndef _mm256_moveldup_pd
#       define _mm256_moveldup_pd(a) _mm256_permute_pd(a, 0)
#   endif
#   ifndef _mm256_movehdup_pd
#       define _mm256_movehdup_pd(a) _mm256_permute_pd(a, 15)
#   endif

#   ifndef _mm_moveldup_pd
#       define _mm_moveldup_pd(a) _mm_permute_pd(a, 0)
#   endif
#   ifndef _mm_movehdup_pd
#       define _mm_movehdup_pd(a) _mm_permute_pd(a, 3)
#   endif



#define restrict
namespace  g2s{

	template<typename T>
	inline void complexAddAlphaxCxD_32(T* dst, const T* C, const T* D, const T Alpha, const unsigned int size){

		#pragma omp simd
		for (int i = 0; i < size; i++)
		{
			dst[2*i+0]+= Alpha * ( C[2*i+0]*D[2*i+0] - C[2*i+1]*D[2*i+1] );
			dst[2*i+1]+= Alpha * ( C[2*i+0]*D[2*i+1] + C[2*i+1]*D[2*i+0] );
		}
	}
	
#if __SSE3__
	inline void complexAddAlphaxCxD_128(float* dst,  const float* C, const float* D, const float Alpha, const unsigned int size){


		__m128 alphaVect=_mm_set1_ps(Alpha);

		for (int i = 0; i < 2*size/4; i++)
		{

			__m128 C_vec=_mm_loadu_ps(C+i*4);
			__m128 D_vec=_mm_loadu_ps(D+i*4);

			__m128 CD=_mm_mul_ps(_mm_moveldup_ps(D_vec),C_vec);
			__m128 CCD=_mm_mul_ps(_mm_movehdup_ps(D_vec),C_vec);
			CCD=_mm_shuffle_ps( CCD, CCD, _MM_SHUFFLE(2,3,0,1));

			__m128 valCD=_mm_addsub_ps(CD,CCD);
		
		#if __FMA__
			__m128 val=_mm_fmadd_ps (valCD,alphaVect,_mm_loadu_ps(dst+i*4));//a*b+c
		#else
			__m128 val=_mm_add_ps (_mm_mul_ps(valCD,alphaVect),_mm_loadu_ps(dst+i*4));
		#endif
	
			_mm_storeu_ps(dst+i*4,val);
		}

		for (int i = (size/4)*4; i < size; i++)
		{
			dst[2*i+0]+= Alpha * ( C[2*i+0]*D[2*i+0] - C[2*i+1]*D[2*i+1] );
			dst[2*i+1]+= Alpha * ( C[2*i+0]*D[2*i+1] + C[2*i+1]*D[2*i+0] );
		}
	}

	
	inline void complexAddAlphaxCxD_128(double* dst,  const double* C, const double* D, const double Alpha, const unsigned int size){


		__m128d alphaVect=_mm_set1_pd(Alpha);

		for (int i = 0; i < 2*size/2; i++)
		{

			__m128d C_vec=_mm_loadu_pd(C+i*2);
			__m128d D_vec=_mm_loadu_pd(D+i*2);

			__m128d CD=_mm_mul_pd(_mm_moveldup_pd(D_vec),C_vec);
			__m128d CCD=_mm_mul_pd(_mm_movehdup_pd(D_vec),C_vec);
			CCD=_mm_shuffle_pd( CCD, CCD, 1);

			__m128d valCD=_mm_addsub_pd(CD,CCD);
		
		#if __FMA__
			__m128d val=_mm_fmadd_pd (valCD,alphaVect,_mm_loadu_pd(dst+i*2));//a*b+c
		#else
			__m128d val=_mm_add_pd (_mm_mul_pd(valCD,alphaVect),_mm_loadu_pd(dst+i*2));
		#endif
	
			_mm_storeu_pd(dst+i*2,val);
		}

		for (int i = (size/2)*2; i < size; i++)
		{
			dst[2*i+0]+= Alpha * ( C[2*i+0]*D[2*i+0] - C[2*i+1]*D[2*i+1] );
			dst[2*i+1]+= Alpha * ( C[2*i+0]*D[2*i+1] + C[2*i+1]*D[2*i+0] );
		}
	}

	
	
#endif

#if __AVX__
	inline void complexAddAlphaxCxD_256(float* dst, const float* C, const float* D, const float Alpha, const unsigned int size){


		__m256 alphaVect=_mm256_set1_ps(Alpha);

		for (int i = 0; i < 2*size/8; i++)
		{
			
			__m256 C_vec=_mm256_loadu_ps(C+i*8);
			__m256 D_vec=_mm256_loadu_ps(D+i*8);

			__m256 CD=_mm256_mul_ps(_mm256_moveldup_ps(D_vec),C_vec);
			__m256 CCD=_mm256_mul_ps(_mm256_movehdup_ps(D_vec),C_vec);
			CCD=_mm256_shuffle_ps( CCD, CCD, _MM_SHUFFLE(2,3,0,1));

			
			__m256 valCD=_mm256_addsub_ps(CD,CCD);
		#if __FMA__
			__m256 val=_mm256_fmadd_ps (valCD,alphaVect,_mm256_loadu_ps(dst+i*8));//a*b+c
		#else
			__m256 val=_mm256_add_ps (_mm256_mul_ps(valCD,alphaVect),_mm256_loadu_ps(dst+i*8));
		#endif
			_mm256_storeu_ps(dst+i*8,val);
		}

		for (int i = (size/8)*8; i < size; i++)
		{
			dst[2*i+0]+= Alpha * ( C[2*i+0]*D[2*i+0] - C[2*i+1]*D[2*i+1] );
			dst[2*i+1]+= Alpha * ( C[2*i+0]*D[2*i+1] + C[2*i+1]*D[2*i+0] );
		}
	}


	inline void complexAddAlphaxCxD_256(double* dst, const double* C, const double* D, const double Alpha, const unsigned int size){


		__m256d alphaVect=_mm256_set1_pd(Alpha);

		for (int i = 0; i < 2*size/4; i++)
		{
			
			__m256d C_vec=_mm256_loadu_pd(C+i*4);
			__m256d D_vec=_mm256_loadu_pd(D+i*4);

			__m256d CD=_mm256_mul_pd(_mm256_moveldup_pd(D_vec),C_vec);
			__m256d CCD=_mm256_mul_pd(_mm256_movehdup_pd(D_vec),C_vec);
			CCD=_mm256_shuffle_pd( CCD, CCD, 5);

			
			__m256d valCD=_mm256_addsub_pd(CD,CCD);
		#if __FMA__
			__m256d val=_mm256_fmadd_pd (valCD,alphaVect,_mm256_loadu_pd(dst+i*4));//a*b+c
		#else
			__m256d val=_mm256_add_pd (_mm256_mul_pd(valCD,alphaVect),_mm256_loadu_pd(dst+i*4));
		#endif
			_mm256_storeu_pd(dst+i*4,val);
		}

		for (int i = (size/4)*4; i < size; i++)
		{
			dst[2*i+0]+= Alpha * ( C[2*i+0]*D[2*i+0] - C[2*i+1]*D[2*i+1] );
			dst[2*i+1]+= Alpha * ( C[2*i+0]*D[2*i+1] + C[2*i+1]*D[2*i+0] );
		}
	}

	
#endif

#if __AVX512F__
	inline void complexAddAlphaxCxD_512(float* dst, const float* C, const float* D, const float Alpha, const unsigned int size){


		__m512 alphaVect=_mm512_set1_ps(Alpha);
		__m512 onesVect=_mm512_set1_ps(1.f);

		for (int i = 0; i < 2*size/16; i++)
		{
			
			__m512 C_vec=_mm512_loadu_ps(C+i*16);
			__m512 D_vec=_mm512_loadu_ps(D+i*16);

			__m512 CD=_mm512_mul_ps(_mm512_moveldup_ps(D_vec),C_vec);
			__m512 CCD=_mm512_mul_ps(_mm512_movehdup_ps(D_vec),C_vec);
			CCD=_mm512_shuffle_ps( CCD, CCD, _MM_SHUFFLE(2,3,0,1));

			
			__m512 valCD=_mm512_fmaddsub_ps (CD, onesVect, CCD);
			
		#if __FMA__
			__m512 val=_mm512_fmadd_ps (valCD,alphaVect,_mm512_loadu_ps(dst+i*16));//a*b+c
		#else
			__m512 val=_mm512_add_ps (_mm512_mul_ps(valCD,alphaVect),_mm512_loadu_ps(dst+i*16));
		#endif
	
			_mm512_storeu_ps(dst+i*16,val);
		}

		for (int i = (size/16)*16; i < size; i++)
		{
			dst[2*i+0]+= Alpha * ( C[2*i+0]*D[2*i+0] - C[2*i+1]*D[2*i+1] );
			dst[2*i+1]+= Alpha * ( C[2*i+0]*D[2*i+1] + C[2*i+1]*D[2*i+0] );
		}
	}

	inline void complexAddAlphaxCxD_512(double* dst, const double* C, const double* D, const double Alpha, const unsigned int size){


		__m512d alphaVect=_mm512_set1_pd(Alpha);
		__m512d onesVect=_mm512_set1_pd(1.f);

		for (int i = 0; i < 2*size/8; i++)
		{
			
			__m512d C_vec=_mm512_loadu_pd(C+i*8);
			__m512d D_vec=_mm512_loadu_pd(D+i*8);

			__m512d CD=_mm512_mul_pd(_mm512_moveldup_pd(D_vec),C_vec);
			__m512d CCD=_mm512_mul_pd(_mm512_movehdup_pd(D_vec),C_vec);
			CCD=_mm512_shuffle_pd( CCD, CCD, 85);

			
			__m512d valCD=_mm512_fmaddsub_pd(CD, onesVect, CCD);
			
		#if __FMA__
			__m512d val=_mm512_fmadd_pd(valCD,alphaVect,_mm512_loadu_pd(dst+i*8));//a*b+c
		#else
			__m512d val=_mm512_add_pd(_mm512_mul_pd(valCD,alphaVect),_mm512_loadu_pd(dst+i*8));
		#endif
	
			_mm512_storeu_pd(dst+i*8,val);
		}

		for (int i = (size/8)*8; i < size; i++)
		{
			dst[2*i+0]+= Alpha * ( C[2*i+0]*D[2*i+0] - C[2*i+1]*D[2*i+1] );
			dst[2*i+1]+= Alpha * ( C[2*i+0]*D[2*i+1] + C[2*i+1]*D[2*i+0] );
		}
	}

#endif
template<typename T>
	inline void complexAddAlphaxCxD(T* dst, const T* C, const T* D, const T Alpha, const unsigned int size){

	#if __AVX512F__
		complexAddAlphaxCxD_512(dst, C, D, Alpha, size);
		return;
	#endif

	#if __AVX__
		complexAddAlphaxCxD_256(dst, C, D, Alpha, size);
		return;
	#endif

	#if __SSE3__
		complexAddAlphaxCxD_128(dst, C, D, Alpha, size);
		return;
	#endif

		complexAddAlphaxCxD_32(dst, C, D, Alpha, size);
		return;	

	}
}

#endif