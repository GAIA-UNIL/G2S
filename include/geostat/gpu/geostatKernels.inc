/*
 * G2S - geostat: device kernels shared by the OpenCL and CUDA back ends.
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

// One source for both back ends. The whole file is the argument of
// G2S_GPU_SOURCE(...):
//   * OpenCL (gpu/openclBackend, src/geostatOpenCL.cpp) defines it as
//     #__VA_ARGS__: the code becomes a string, compiled at run time after the
//     OpenCL prelude (gpu/kernelPrelude.hpp, kOpenClPrelude);
//   * CUDA (src/geostatCuda.cu) defines it as __VA_ARGS__ after the CUDA
//     prelude: the same code is compiled by nvcc.
// Rules that follow from this: no preprocessor directive and no comment that
// must survive inside the parentheses (comments are stripped, which is fine);
// only the prelude macros for qualifiers, types, thread indices and the few
// intrinsics that differ:
//   G2S_KERNEL            kernel entry point (__kernel / extern "C" __global__)
//   G2S_DEVICE            device function (inline / __device__ inline)
//   G2S_GLOBAL            pointer to device memory (__global / nothing)
//   G2S_LOCAL_ARRAY(T, name, n)   work-group shared array
//   G2S_BARRIER()         work-group barrier on shared memory
//   G2S_GROUP_ID(d), G2S_LOCAL_ID(d), G2S_GLOBAL_ID(d), G2S_LOCAL_SIZE(d)
//   g2s_u32, g2s_i32, g2s_u64, g2s_i64, REAL (float or double, kriging only)
//   G2S_MULHI64(a, b)     high 64 bits of a 64 x 64 bit product
//   G2S_FMAF(a, b, c)     fused float multiply-add
//   G2S_FMUL_RN, G2S_FADD_RN  float multiply / add never contracted into an FMA
//   G2S_SQRTF             correctly rounded float square root
//   G2S_AS_FLOAT, G2S_AS_UINT  bit casts
//   G2S_U64(c)            64-bit unsigned integer constant
//
// Index arithmetic that may exceed 2^32 uses g2s_u64.

G2S_GPU_SOURCE(

G2S_DEVICE g2s_u32 g2sMix32(g2s_u32 h) {
	h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
	return h;
}

G2S_DEVICE void g2sTwoSum(float* hi, float* lo, float x) {
	float s = G2S_FADD_RN(*hi, x);
	float bp = G2S_FADD_RN(s, -*hi);
	float e = G2S_FADD_RN(G2S_FADD_RN(*hi, -G2S_FADD_RN(s, -bp)), G2S_FADD_RN(x, -bp));
	*hi = s;
	*lo = G2S_FADD_RN(*lo, e);
}

G2S_DEVICE float g2sSinTurnsQuarter(float a) {
	float x = a * a;
	float p = G2S_FMAF(39.53417324387686f, x, -76.54941360513733f);
	p = G2S_FMAF(p, x, 81.60098593229426f);
	p = G2S_FMAF(p, x, -41.34165469233216f);
	p = G2S_FMAF(p, x, 6.283185158278849f);
	return a * p;
}

G2S_DEVICE float g2sCosFixed(g2s_u64 phase) {
	g2s_i32 y = (g2s_i32)(g2s_u32)(phase >> 32);
	g2s_u32 magnitude = y < 0 ? (g2s_u32)0 - (g2s_u32)y : (g2s_u32)y;
	g2s_i32 a = (g2s_i32)(0x40000000u - magnitude);
	return g2sSinTurnsQuarter((float)a * 2.3283064365386963e-10f);
}

G2S_DEVICE void g2sFixedMulAdd(g2s_u32* whole, g2s_u64* fraction, g2s_u64 i, g2s_u32 stepWhole, g2s_u64 stepFraction) {
	g2s_u64 low = i * stepFraction;
	g2s_u64 high = G2S_MULHI64(i, stepFraction);
	g2s_u64 f = *fraction + low;
	g2s_u32 carry = f < low ? 1u : 0u;
	*whole = *whole + (g2s_u32)i * stepWhole + (g2s_u32)high + carry;
	*fraction = f;
}

G2S_DEVICE g2s_u64 g2sFixedRound32(g2s_u32 whole, g2s_u64 fraction) {
	return ((g2s_u64)whole << 32) + (fraction >> 32) + ((fraction >> 31) & 1u);
}

G2S_KERNEL void tbRowTable(G2S_GLOBAL const g2s_u32* lineWhole, G2S_GLOBAL const g2s_u64* lineFraction,
                           g2s_u32 lineCount, g2s_u32 linePitch, g2s_u64 size1, g2s_u64 firstRow, g2s_u32 rowCount, g2s_u32 rowPitch,
                           g2s_u32 spectral, G2S_GLOBAL float* rowTrig, G2S_GLOBAL g2s_u64* rowPhase) {
	g2s_u32 r = (g2s_u32)G2S_GLOBAL_ID(0);
	g2s_u32 l = (g2s_u32)G2S_GLOBAL_ID(1);
	if (r >= rowPitch || l >= linePitch) return;
	if (r >= rowCount || l >= lineCount) {
		if (spectral) { rowTrig[(g2s_u64)(2 * l) * rowPitch + r] = 0.0f; rowTrig[(g2s_u64)(2 * l + 1) * rowPitch + r] = 0.0f; }
		else rowPhase[(g2s_u64)l * rowPitch + r] = 0;
		return;
	}
	g2s_u64 row = firstRow + r;
	g2s_u64 i1 = row % size1, i2 = row / size1;
	g2s_u32 whole = lineWhole[l];
	g2s_u64 fraction = lineFraction[l];
	g2sFixedMulAdd(&whole, &fraction, i1, lineWhole[2 * linePitch + l], lineFraction[2 * linePitch + l]);
	g2sFixedMulAdd(&whole, &fraction, i2, lineWhole[3 * linePitch + l], lineFraction[3 * linePitch + l]);
	if (spectral) {
		rowTrig[(g2s_u64)(2 * l) * rowPitch + r] = g2sCosFixed(fraction);
		rowTrig[(g2s_u64)(2 * l + 1) * rowPitch + r] = -g2sCosFixed(fraction - G2S_U64(0x4000000000000000));
	} else {
		rowPhase[(g2s_u64)l * rowPitch + r] = g2sFixedRound32(whole, fraction);
	}
}

G2S_KERNEL void tbColumnTable(G2S_GLOBAL const g2s_u32* lineWhole, G2S_GLOBAL const g2s_u64* lineFraction,
                              g2s_u32 lineCount, g2s_u32 linePitch, g2s_u32 columnCount, g2s_u32 columnPitch,
                              g2s_u32 spectral, G2S_GLOBAL float* columnTrig, G2S_GLOBAL g2s_u64* columnPhase) {
	g2s_u32 x = (g2s_u32)G2S_GLOBAL_ID(0);
	g2s_u32 l = (g2s_u32)G2S_GLOBAL_ID(1);
	if (x >= columnPitch || l >= linePitch) return;
	if (x >= columnCount || l >= lineCount) {
		if (spectral) { columnTrig[(g2s_u64)(2 * l) * columnPitch + x] = 0.0f; columnTrig[(g2s_u64)(2 * l + 1) * columnPitch + x] = 0.0f; }
		else columnPhase[(g2s_u64)l * columnPitch + x] = 0;
		return;
	}
	g2s_u32 whole = 0;
	g2s_u64 fraction = 0;
	g2sFixedMulAdd(&whole, &fraction, (g2s_u64)x, lineWhole[linePitch + l], lineFraction[linePitch + l]);
	if (spectral) {
		columnTrig[(g2s_u64)(2 * l) * columnPitch + x] = g2sCosFixed(fraction);
		columnTrig[(g2s_u64)(2 * l + 1) * columnPitch + x] = g2sCosFixed(fraction - G2S_U64(0x4000000000000000));
	} else {
		columnPhase[(g2s_u64)l * columnPitch + x] = g2sFixedRound32(whole, fraction);
	}
}

G2S_KERNEL void tbSpectralProduct(G2S_GLOBAL const float* rowTrig, G2S_GLOBAL const float* columnTrig,
                                  g2s_u32 depth, g2s_u32 rowPitch, g2s_u32 columnPitch,
                                  float amplitude, g2s_u32 accumulate, G2S_GLOBAL float* out) {
	G2S_LOCAL_ARRAY(float, tileA, 16 * 128);
	G2S_LOCAL_ARRAY(float, tileB, 16 * 64);
	g2s_u32 tx = (g2s_u32)G2S_LOCAL_ID(0), ty = (g2s_u32)G2S_LOCAL_ID(1);
	g2s_u32 x0 = (g2s_u32)G2S_GROUP_ID(0) * 64, r0 = (g2s_u32)G2S_GROUP_ID(1) * 128;
	g2s_u32 thread = ty * 16 + tx;
	float partial[32], total[32];
	for (int i = 0; i < 32; ++i) { partial[i] = 0.0f; total[i] = 0.0f; }
	for (g2s_u32 k0 = 0; k0 < depth; k0 += 16) {
		for (int j = 0; j < 8; ++j) {
			g2s_u32 e = thread + 256 * j;
			tileA[e] = rowTrig[(g2s_u64)(k0 + e / 128) * rowPitch + r0 + e % 128];
		}
		for (int j = 0; j < 4; ++j) {
			g2s_u32 e = thread + 256 * j;
			tileB[e] = columnTrig[(g2s_u64)(k0 + e / 64) * columnPitch + x0 + e % 64];
		}
		G2S_BARRIER();
		for (int kk = 0; kk < 16; ++kk) {
			float a[8], b[4];
			for (int i = 0; i < 8; ++i) a[i] = tileA[kk * 128 + ty + 16 * i];
			for (int j = 0; j < 4; ++j) b[j] = tileB[kk * 64 + tx + 16 * j];
			for (int i = 0; i < 8; ++i)
				for (int j = 0; j < 4; ++j) partial[i * 4 + j] = G2S_FMAF(a[i], b[j], partial[i * 4 + j]);
		}
		G2S_BARRIER();
		if (((k0 + 16) % 128) == 0 || k0 + 16 >= depth) {
			for (int i = 0; i < 32; ++i) { total[i] += partial[i]; partial[i] = 0.0f; }
		}
	}
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 4; ++j) {
			g2s_u64 index = (g2s_u64)(r0 + ty + 16 * i) * columnPitch + x0 + tx + 16 * j;
			float value = amplitude * total[i * 4 + j];
			out[index] = accumulate ? out[index] + value : value;
		}
}

G2S_KERNEL void tbDilutionProduct(G2S_GLOBAL const g2s_u64* rowPhase, G2S_GLOBAL const g2s_u64* columnPhase,
                                  G2S_GLOBAL const g2s_u32* signKey, g2s_u32 lineCount, g2s_u32 linePitch,
                                  g2s_u32 rowPitch, g2s_u32 columnPitch, g2s_u32 cubic,
                                  float amplitude, g2s_u32 accumulate, G2S_GLOBAL float* out) {
	G2S_LOCAL_ARRAY(g2s_u64, tileA, 16 * 64);
	G2S_LOCAL_ARRAY(g2s_u64, tileB, 16 * 64);
	G2S_LOCAL_ARRAY(g2s_u32, tileKey, 32);
	g2s_u32 tx = (g2s_u32)G2S_LOCAL_ID(0), ty = (g2s_u32)G2S_LOCAL_ID(1);
	g2s_u32 x0 = (g2s_u32)G2S_GROUP_ID(0) * 64, r0 = (g2s_u32)G2S_GROUP_ID(1) * 64;
	g2s_u32 thread = ty * 16 + tx;
	const float factor = cubic ? 7.24568837309471540f : 3.46410161513775459f;
	float partial[16], hi[16], lo[16];
	for (int i = 0; i < 16; ++i) { partial[i] = 0.0f; hi[i] = 0.0f; lo[i] = 0.0f; }
	for (g2s_u32 l0 = 0; l0 < lineCount; l0 += 16) {
		for (int j = 0; j < 4; ++j) {
			g2s_u32 e = thread + 256 * j;
			g2s_u32 kk = e / 64, c = e % 64;
			tileA[e] = rowPhase[(g2s_u64)(l0 + kk) * rowPitch + r0 + c];
			tileB[e] = columnPhase[(g2s_u64)(l0 + kk) * columnPitch + x0 + c];
		}
		if (thread < 16) { tileKey[thread] = signKey[l0 + thread]; tileKey[16 + thread] = signKey[linePitch + l0 + thread]; }
		G2S_BARRIER();
		g2s_u32 lines = lineCount - l0 < 16 ? lineCount - l0 : 16;
		for (g2s_u32 kk = 0; kk < lines; ++kk) {
			g2s_u64 a[4], b[4];
			for (int i = 0; i < 4; ++i) { a[i] = tileA[kk * 64 + ty + 16 * i]; b[i] = tileB[kk * 64 + tx + 16 * i]; }
			g2s_u32 multiplier = tileKey[kk], offset = tileKey[16 + kk];
			for (int i = 0; i < 4; ++i)
				for (int j = 0; j < 4; ++j) {
					g2s_u64 t = a[i] + b[j];
					g2s_u32 segment = (g2s_u32)(t >> 32);
					float s = (float)(g2s_i32)((g2s_u32)t - 0x80000000u) * 2.3283064365386963e-10f;
					g2s_u32 sign = g2sMix32(segment * multiplier + offset) & 0x80000000u;
					float g = cubic ? s * G2S_FMAF(-4.0f * factor, s * s, factor) : factor * s;
					partial[i * 4 + j] += G2S_AS_FLOAT(G2S_AS_UINT(g) ^ sign);
				}
		}
		G2S_BARRIER();
		if (((l0 + 16) % 64) == 0 || l0 + 16 >= lineCount) {
			for (int i = 0; i < 16; ++i) { g2sTwoSum(&hi[i], &lo[i], partial[i]); partial[i] = 0.0f; }
		}
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) {
			g2s_u64 index = (g2s_u64)(r0 + ty + 16 * i) * columnPitch + x0 + tx + 16 * j;
			float value = amplitude * (hi[i * 4 + j] + lo[i * 4 + j]);
			out[index] = accumulate ? out[index] + value : value;
		}
}

G2S_DEVICE void g2sPhilox(g2s_u32* c, g2s_u32 k0, g2s_u32 k1) {
	for (int round = 0; round < 10; ++round) {
		if (round > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
		g2s_u64 p0 = (g2s_u64)0xD2511F53u * c[0];
		g2s_u64 p1 = (g2s_u64)0xCD9E8D57u * c[2];
		g2s_u32 n0 = (g2s_u32)(p1 >> 32) ^ c[1] ^ k0;
		g2s_u32 n2 = (g2s_u32)(p0 >> 32) ^ c[3] ^ k1;
		c[0] = n0; c[1] = (g2s_u32)p1; c[2] = n2; c[3] = (g2s_u32)p0;
	}
}

G2S_DEVICE float g2sMulAdd(float a, float b, float c) { return G2S_FADD_RN(G2S_FMUL_RN(a, b), c); }

G2S_DEVICE float g2sFastLog(float x) {
	g2s_u32 bits = G2S_AS_UINT(x);
	g2s_i32 e = (g2s_i32)(bits >> 23) - 126;
	float m = G2S_AS_FLOAT((bits & 0x007FFFFFu) | 0x3F000000u);
	int low = m < 0.707106781186547524f;
	e -= low ? 1 : 0;
	m = low ? G2S_FADD_RN(G2S_FADD_RN(m, m), -1.0f) : G2S_FADD_RN(m, -1.0f);
	float z = G2S_FMUL_RN(m, m);
	float y = 7.0376836292e-2f;
	y = g2sMulAdd(y, m, -1.1514610310e-1f);
	y = g2sMulAdd(y, m, 1.1676998740e-1f);
	y = g2sMulAdd(y, m, -1.2420140846e-1f);
	y = g2sMulAdd(y, m, 1.4249322787e-1f);
	y = g2sMulAdd(y, m, -1.6668057665e-1f);
	y = g2sMulAdd(y, m, 2.0000714765e-1f);
	y = g2sMulAdd(y, m, -2.4999993993e-1f);
	y = g2sMulAdd(y, m, 3.3333331174e-1f);
	float mz = G2S_FMUL_RN(m, z);
	y = G2S_FMUL_RN(y, mz);
	float fe = (float)e;
	y = g2sMulAdd(-2.12194440e-4f, fe, y);
	y = g2sMulAdd(-0.5f, z, y);
	float sum = G2S_FADD_RN(m, y);
	return g2sMulAdd(0.693359375f, fe, sum);
}

G2S_DEVICE void g2sBoxMuller(g2s_u32 k1, g2s_u32 k2, float* first, float* second) {
	float u1 = g2sMulAdd((float)k1, 2.3283064365386963e-10f, 1.1641532182693481e-10f);
	float radius = G2S_SQRTF(G2S_FMUL_RN(-2.0f, g2sFastLog(u1)));
	g2s_u32 shifted = k2 + (1u << 29);
	g2s_u32 quadrant = shifted >> 30;
	float a = G2S_FMUL_RN((float)((g2s_i32)(shifted & 0x3FFFFFFFu) - (g2s_i32)(1u << 29)), 1.46291807926715968e-9f);
	float z = G2S_FMUL_RN(a, a);
	float s = g2sMulAdd(-1.9515295891e-4f, z, 8.3321608736e-3f);
	s = g2sMulAdd(s, z, -1.6666654611e-1f);
	float za = G2S_FMUL_RN(z, a);
	s = g2sMulAdd(s, za, a);
	float c = g2sMulAdd(2.443315711809948e-5f, z, -1.388731625493765e-3f);
	c = g2sMulAdd(c, z, 4.166664568298827e-2f);
	float zz = G2S_FMUL_RN(z, z);
	c = g2sMulAdd(c, zz, 1.0f);
	c = g2sMulAdd(-0.5f, z, c);
	float cq = (quadrant & 1u) ? s : c;
	float sq = (quadrant & 1u) ? c : s;
	float cosine = (quadrant == 1u || quadrant == 2u) ? -cq : cq;
	float sine = (quadrant >= 2u) ? -sq : sq;
	*first = G2S_FMUL_RN(radius, cosine);
	*second = G2S_FMUL_RN(radius, sine);
}

G2S_DEVICE void g2sFastNormalBlock(g2s_u64 block, g2s_u32 realisation, g2s_u32 stream, g2s_u32 k0, g2s_u32 k1, float* out) {
	g2s_u32 c[4];
	c[0] = (g2s_u32)block; c[1] = (g2s_u32)(block >> 32); c[2] = realisation; c[3] = stream;
	g2sPhilox(c, k0, k1);
	g2sBoxMuller(c[0], c[1], &out[0], &out[1]);
	g2sBoxMuller(c[2], c[3], &out[2], &out[3]);
}

G2S_KERNEL void fftWhiteNoise(g2s_u32 k0, g2s_u32 k1, g2s_u32 realisation, g2s_u32 stream,
                              g2s_u64 n0, g2s_u64 rowPitch, g2s_u64 cells, G2S_GLOBAL float* padded) {
	g2s_u64 block = G2S_GLOBAL_ID(0);
	if (block * 4 >= cells) return;
	float normals[4];
	g2sFastNormalBlock(block, realisation, stream, k0, k1, normals);
	for (int lane = 0; lane < 4; ++lane) {
		g2s_u64 flat = block * 4 + lane;
		if (flat >= cells) break;
		g2s_u64 row = flat / n0, x = flat - row * n0;
		padded[row * rowPitch + x] = normals[lane];
	}
}

G2S_KERNEL void fftMultiplyAmplitude(G2S_GLOBAL const float* amplitude, float scale, g2s_u64 halfCount, G2S_GLOBAL float* spectrum) {
	g2s_u64 i = G2S_GLOBAL_ID(0);
	if (i >= halfCount) return;
	float a = G2S_FMUL_RN(scale, amplitude[i]);
	spectrum[2 * i] = G2S_FMUL_RN(spectrum[2 * i], a);
	spectrum[2 * i + 1] = G2S_FMUL_RN(spectrum[2 * i + 1], a);
}

G2S_KERNEL void fftSpectralNoise(g2s_u32 k0, g2s_u32 k1, g2s_u32 realisation, g2s_u32 stream,
                                 g2s_u32 ndim, g2s_u64 e0, g2s_u64 e1, g2s_u64 e2, g2s_u64 halfRow, g2s_u64 rows,
                                 G2S_GLOBAL const float* amplitude, float scale, G2S_GLOBAL float* spectrum) {
	g2s_u64 i = G2S_GLOBAL_ID(0);
	if (i >= halfRow * rows) return;
	g2s_u64 row = i / halfRow, kx = i - row * halfRow;
	g2s_u64 flat = row * e0 + kx;
	float normals[4];
	g2sFastNormalBlock((2 * flat) >> 2, realisation, stream, k0, k1, normals);
	g2s_u32 lane = (g2s_u32)((2 * flat) & 3u);
	float re = normals[lane], im = normals[lane + 1];
	const float halfScale = 0.70710678118654752f;
	int selfPlane = (kx == 0) || ((e0 % 2) == 0 && kx == e0 / 2);
	float sre, sim;
	if (!selfPlane) {
		sre = re * halfScale; sim = im * halfScale;
	} else {
		g2s_u64 c1 = ndim > 1 ? row % e1 : 0;
		g2s_u64 c2 = ndim > 2 ? (row / e1) % e2 : 0;
		g2s_u64 p1 = ndim > 1 ? (e1 - c1) % e1 : 0;
		g2s_u64 p2 = ndim > 2 ? (e2 - c2) % e2 : 0;
		g2s_u64 partnerRow = p1 + e1 * p2;
		g2s_u64 partnerFlat = partnerRow * e0 + kx;
		if (partnerFlat == flat) { sre = re; sim = 0.0f; }
		else if (flat < partnerFlat) { sre = re * halfScale; sim = im * halfScale; }
		else {
			float pn[4];
			g2sFastNormalBlock((2 * partnerFlat) >> 2, realisation, stream, k0, k1, pn);
			g2s_u32 pl = (g2s_u32)((2 * partnerFlat) & 3u);
			sre = pn[pl] * halfScale; sim = -pn[pl + 1] * halfScale;
		}
	}
	float a = scale * amplitude[i];
	spectrum[2 * i] = a * sre;
	spectrum[2 * i + 1] = a * sim;
}

G2S_KERNEL void fftCrop(G2S_GLOBAL const float* padded, g2s_u32 ndim, g2s_u64 g0, g2s_u64 g1, g2s_u64 g2,
                        g2s_u64 e1, g2s_u64 rowPitch, float mean, G2S_GLOBAL float* out) {
	g2s_u64 i = G2S_GLOBAL_ID(0);
	if (i >= g0 * g1 * g2) return;
	g2s_u64 outRow = i / g0, x = i - outRow * g0;
	g2s_u64 c1 = outRow % g1, c2 = outRow / g1;
	g2s_u64 row = c1 + e1 * c2;
	out[i] = mean + padded[row * rowPitch + x];
}

G2S_KERNEL void krigingGather(G2S_GLOBAL const REAL* table, G2S_GLOBAL const g2s_u64* dataBase, G2S_GLOBAL const g2s_u64* targetOffset,
                              g2s_u32 n, g2s_u32 m, g2s_u32 pitch, G2S_GLOBAL REAL* y) {
	g2s_u32 t = (g2s_u32)G2S_GLOBAL_ID(0);
	g2s_u32 i = (g2s_u32)G2S_GLOBAL_ID(1);
	if (t >= pitch) return;
	y[(g2s_u64)i * pitch + t] = (t < m && i < n) ? table[dataBase[i] + targetOffset[t]] : (REAL)0;
}

G2S_KERNEL void krigingGemmSubtract(G2S_GLOBAL const REAL* l, g2s_u32 ldl, G2S_GLOBAL REAL* y, g2s_u32 pitch,
                                    g2s_u32 rowBegin, g2s_u32 depth) {
	G2S_LOCAL_ARRAY(REAL, tileA, 16 * 64);
	G2S_LOCAL_ARRAY(REAL, tileB, 16 * 64);
	g2s_u32 tx = (g2s_u32)G2S_LOCAL_ID(0), ty = (g2s_u32)G2S_LOCAL_ID(1);
	g2s_u32 c0 = (g2s_u32)G2S_GROUP_ID(0) * 64, r0 = rowBegin + (g2s_u32)G2S_GROUP_ID(1) * 64;
	g2s_u32 thread = ty * 16 + tx;
	REAL acc[16];
	for (int i = 0; i < 16; ++i) acc[i] = (REAL)0;
	for (g2s_u32 k0 = 0; k0 < depth; k0 += 16) {
		for (int j = 0; j < 4; ++j) {
			g2s_u32 e = thread + 256 * j;
			g2s_u32 rr = e / 16, kk = e % 16;
			tileA[kk * 64 + rr] = l[(g2s_u64)(r0 + rr) * ldl + k0 + kk];
			g2s_u32 kb = e / 64, cc = e % 64;
			tileB[e] = y[(g2s_u64)(k0 + kb) * pitch + c0 + cc];
		}
		G2S_BARRIER();
		for (int kk = 0; kk < 16; ++kk) {
			REAL a[4], b[4];
			for (int i = 0; i < 4; ++i) { a[i] = tileA[kk * 64 + ty + 16 * i]; b[i] = tileB[kk * 64 + tx + 16 * i]; }
			for (int i = 0; i < 4; ++i)
				for (int j = 0; j < 4; ++j) acc[i * 4 + j] += a[i] * b[j];
		}
		G2S_BARRIER();
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) {
			g2s_u64 index = (g2s_u64)(r0 + ty + 16 * i) * pitch + c0 + tx + 16 * j;
			y[index] -= acc[i * 4 + j];
		}
}

G2S_KERNEL void krigingDiagonalSolve(G2S_GLOBAL const REAL* l, g2s_u32 ldl, G2S_GLOBAL REAL* y, g2s_u32 pitch,
                                     g2s_u32 rowBegin, g2s_u32 rowEnd) {
	g2s_u32 t = (g2s_u32)G2S_GLOBAL_ID(0);
	if (t >= pitch) return;
	for (g2s_u32 i = rowBegin; i < rowEnd; ++i) {
		G2S_GLOBAL const REAL* li = l + (g2s_u64)i * ldl;
		REAL v = y[(g2s_u64)i * pitch + t];
		for (g2s_u32 k = rowBegin; k < i; ++k) v -= li[k] * y[(g2s_u64)k * pitch + t];
		y[(g2s_u64)i * pitch + t] = v / li[i];
	}
}

G2S_KERNEL void krigingReduce(G2S_GLOBAL const REAL* y, G2S_GLOBAL const REAL* w, g2s_u32 n, g2s_u32 m, g2s_u32 pitch,
                              G2S_GLOBAL REAL* sumSquares, G2S_GLOBAL REAL* dotW) {
	g2s_u32 t = (g2s_u32)G2S_GLOBAL_ID(0);
	if (t >= m) return;
	REAL s = (REAL)0, q = (REAL)0;
	for (g2s_u32 i = 0; i < n; ++i) {
		REAL v = y[(g2s_u64)i * pitch + t];
		s += v * v;
		q += w[i] * v;
	}
	sumSquares[t] = s;
	dotW[t] = q;
}

)
