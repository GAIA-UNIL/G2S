/*
 * G2S (c) by Mathieu Gravey (gravey.mathieu@gmail.com)
 * 
 * G2S is licensed under a
 * Creative Commons Attribution-NonCommercial 4.0 International License.
 * 
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
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