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

#include <set>
#include <vector>

#ifndef SharedMemoryManager_HPP
#define SharedMemoryManager_HPP

#include "utils.hpp"

class ComputeDeviceModule;

class SharedMemoryManager
{
public:
	SharedMemoryManager(std::vector<unsigned> size);
	void addVaraible(void* memoryAdress);

	void addDevice(ComputeDeviceModule *device);
	void removeDevice(ComputeDeviceModule *device);
	~SharedMemoryManager();


	std::vector<unsigned> _srcSize;
	std::vector<unsigned> _fftSize;
	std::vector<g2s::spaceFrequenceMemoryAddress> adressSharedMemory(int memID);
	void allowNewModule(bool);


private:

	ComputeDeviceModule* findSameHostDevice(ComputeDeviceModule *device);
	void createSharedMemoryForDevice(ComputeDeviceModule *device);
	void removeSharedMemoryForDevice(ComputeDeviceModule *device);

	std::vector<void* > _srcMemoryAdress;

	std::set<ComputeDeviceModule*> _devices;
	std::vector<std::vector<g2s::spaceFrequenceMemoryAddress > > _memoryAddress;	
	
	bool _allowNewModule=true;	
};

#endif