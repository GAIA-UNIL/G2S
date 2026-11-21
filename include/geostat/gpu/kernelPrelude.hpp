/*
 * G2S - geostat: the thin macro layer that lets OpenCL and CUDA compile the
 * same kernel source (gpu/geostatKernels.inc).
 * Copyright (C) 2018-2026, Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL (University of Lausanne)
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

#ifndef G2S_GEOSTAT_GPU_KERNEL_PRELUDE_HPP
#define G2S_GEOSTAT_GPU_KERNEL_PRELUDE_HPP

#if defined(__CUDACC__) || defined(__CUDA__)

// ---------------------------------------------------------------- CUDA
// Included by src/geostatCuda.cu before the kernels; REAL is double (the
// kriging kernels run on CUDA in double only).
#include <cstdint>
typedef uint32_t g2s_u32;
typedef int32_t g2s_i32;
typedef unsigned long long g2s_u64;
typedef long long g2s_i64;
#define REAL double
#define G2S_KERNEL extern "C" __global__
#define G2S_DEVICE __device__ inline
#define G2S_GLOBAL
#define G2S_LOCAL_ARRAY(T, name, n) __shared__ T name[n]
#define G2S_BARRIER() __syncthreads()
#define G2S_CUDA_COMPONENT(v, d) ((unsigned long long)((d) == 0 ? (v).x : ((d) == 1 ? (v).y : (v).z)))
#define G2S_GROUP_ID(d) G2S_CUDA_COMPONENT(blockIdx, d)
#define G2S_LOCAL_ID(d) G2S_CUDA_COMPONENT(threadIdx, d)
#define G2S_LOCAL_SIZE(d) G2S_CUDA_COMPONENT(blockDim, d)
#define G2S_GLOBAL_ID(d) (G2S_GROUP_ID(d) * G2S_LOCAL_SIZE(d) + G2S_LOCAL_ID(d))
#define G2S_MULHI64(a, b) __umul64hi((a), (b))
#define G2S_FMAF(a, b, c) __fmaf_rn((a), (b), (c))
#define G2S_FMUL_RN(a, b) __fmul_rn((a), (b))
#define G2S_FADD_RN(a, b) __fadd_rn((a), (b))
#define G2S_SQRTF(x) __fsqrt_rn(x)
#define G2S_AS_FLOAT(x) __uint_as_float(x)
#define G2S_AS_UINT(x) __float_as_uint(x)
#define G2S_U64(c) c##ull

#else

// ---------------------------------------------------------------- OpenCL
// Prepended to the kernel string by the OpenCL back end. Built with
// -DG2S_FP64 on devices with cl_khr_fp64 (REAL = double), otherwise REAL is
// float (then the kriging kernels are only used by the tests, never for
// results: see gpu/backend.hpp). FP_CONTRACT OFF: a * b + c is never fused
// unless written with G2S_FMAF, so that the noise generator matches the CPU
// bit for bit (geostat/random.hpp avoids FMAs for the same reason).
namespace g2s {
namespace geostat {
namespace gpu {
static const char* const kOpenClPrelude =
	"#pragma OPENCL FP_CONTRACT OFF\n"
	"#ifdef G2S_FP64\n"
	"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
	"#define REAL double\n"
	"#else\n"
	"#define REAL float\n"
	"#endif\n"
	"typedef uint g2s_u32;\n"
	"typedef int g2s_i32;\n"
	"typedef ulong g2s_u64;\n"
	"typedef long g2s_i64;\n"
	"#define G2S_KERNEL __kernel\n"
	"#define G2S_DEVICE inline\n"
	"#define G2S_GLOBAL __global\n"
	"#define G2S_LOCAL_ARRAY(T, name, n) __local T name[n]\n"
	"#define G2S_BARRIER() barrier(CLK_LOCAL_MEM_FENCE)\n"
	"#define G2S_GROUP_ID(d) get_group_id(d)\n"
	"#define G2S_LOCAL_ID(d) get_local_id(d)\n"
	"#define G2S_LOCAL_SIZE(d) get_local_size(d)\n"
	"#define G2S_GLOBAL_ID(d) get_global_id(d)\n"
	"#define G2S_MULHI64(a, b) mul_hi((ulong)(a), (ulong)(b))\n"
	"#define G2S_FMAF(a, b, c) fma((float)(a), (float)(b), (float)(c))\n"
	"#define G2S_FMUL_RN(a, b) ((a) * (b))\n"
	"#define G2S_FADD_RN(a, b) ((a) + (b))\n"
	"#define G2S_SQRTF(x) sqrt(x)\n"
	"#define G2S_AS_FLOAT(x) as_float(x)\n"
	"#define G2S_AS_UINT(x) as_uint(x)\n"
	"#define G2S_U64(c) c##UL\n";
} // namespace gpu
} // namespace geostat
} // namespace g2s

#endif

#endif // G2S_GEOSTAT_GPU_KERNEL_PRELUDE_HPP
