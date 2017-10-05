/*
 * Mathieu Gravey
 * Copyright (C) 2017 Mathieu Gravey
 * 
 * This program is protected software: you can not redistribute, use, and/or modify it
 * without the explicit accord from the author : Mathieu Gravey, gravey.mathieu@gmail.com
 *
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
	
	unsigned _memorySize=0;
	bool _allowNewModule=true;
	
	
};

#endif