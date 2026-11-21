/*
 * G2S - geostat: OpenCL back end (src/geostatOpenCL.cpp).
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

#ifndef G2S_GEOSTAT_GPU_OPENCL_DEVICE_HPP
#define G2S_GEOSTAT_GPU_OPENCL_DEVICE_HPP

#include <string>
#include <vector>

#include "geostat/gpu/backend.hpp"

namespace g2s {
namespace geostat {
namespace gpu {

// True when this build has the OpenCL back end (WITH_OPENCL).
bool openclAvailable();
// Names of the OpenCL GPU devices of every platform, in the order of the
// -W_GPU index; empty without OpenCL or without a device.
std::vector<std::string> openclDeviceNames();
// The OpenCL GPU device `index` (-W_GPU index), or nullptr with a message.
Device* openOpenClDevice(int index, std::string& message);
// Tests only: the kriging kernels in float on a device without fp64, to check
// the kernel code itself (never used for results; nullptr when the device has
// fp64, use Device::kriging then).
KrigingKernel* openclKrigingFloatForTests(Device* device, std::string& message);

} // namespace gpu
} // namespace geostat
} // namespace g2s

#endif // G2S_GEOSTAT_GPU_OPENCL_DEVICE_HPP
