/*
 * Mathieu Gravey
 * Copyright (C) 2017 Mathieu Gravey
 * 
 * This program is protected software: you can not redistribute, use, and/or modify it
 * without the explicit accord from the author : Mathieu Gravey, gravey.mathieu@gmail.com
 *
 */
#include <iostream>
#include <cassert>
#include "sharedMemoryManager.hpp"
#include "computeDeviceModule.hpp"
#include "computeDeviceModule.hpp"



 SharedMemoryManager::SharedMemoryManager(std::vector<unsigned> trueSize){
 	_srcSize=trueSize;
	_fftSize=std::vector<unsigned>(trueSize.begin(),trueSize.end());

	for (int j = 0; j < _fftSize.size(); ++j)
	{
		if(_fftSize[j]==1)continue;
		int currentSize=_fftSize[j];
		for (int i = currentSize; i <= (1<<(32-__builtin_clz(currentSize-1))); ++i)
		{
			int val=i;
			for (int j = 2; j < 8; ++j)
			{
				while(val%j==0) {
					val/=j;
				}
			}
			if(val%11==0){
				val/=11;
			}else{
				if(val%13==0){
					val/=13;
				}
			}
			if(val==1){
				currentSize=i;
				break;
			}
		}
		_fftSize[j]=currentSize;
	}

}

void SharedMemoryManager::addVaraible(void* memoryAdress){
	_srcMemoryAdress.push_back(memoryAdress);	
}

void SharedMemoryManager::allowNewModule(bool value){
	_allowNewModule=value;
}

void SharedMemoryManager::addDevice(ComputeDeviceModule *device){
	#pragma omp critical (updateSharedMemory)
	{
		ComputeDeviceModule* sameDevice=findSameHostDevice(device);
		if(nullptr==sameDevice){ //create shared memory
			createSharedMemoryForDevice(device);
		}else{
			device->_memoryID=sameDevice->_memoryID;
		}
		_devices.insert(device);
	}
}

void SharedMemoryManager::removeDevice(ComputeDeviceModule *device){
	#pragma omp critical (updateSharedMemory)
	{
		_devices.erase(device);
		if(nullptr==findSameHostDevice(device)){ //remove shared memory
			removeSharedMemoryForDevice(device);
		}
	}

}

SharedMemoryManager::~SharedMemoryManager(){
	while(!_devices.empty()) {
		removeDevice(*(_devices.begin()));
	}
}

ComputeDeviceModule* SharedMemoryManager::findSameHostDevice(ComputeDeviceModule *device){

	for (auto i = _devices.begin(); i != _devices.end(); ++i)
	{
		if( ((*i)->_memoryID==device->_memoryID) || (((*i)->_deviceType==device->_deviceType) && ((*i)->_deviceID==device->_deviceID) )){
			//fprintf(stderr, "same divice memId =%d\n",(*i)->_memoryID);
			return (*i);
		}
	}
	return nullptr;
}

void SharedMemoryManager::createSharedMemoryForDevice(ComputeDeviceModule *device){
	assert(_allowNewModule);
	device->_memoryID=_memoryAddress.size();
	_memoryAddress.push_back(device->allocAndInitSharedMemory( _srcMemoryAdress, _srcSize, _fftSize));
	
}

void SharedMemoryManager::removeSharedMemoryForDevice(ComputeDeviceModule *device){
	unsigned deviceMemId=device->_memoryID;

	if(deviceMemId >= _memoryAddress.size())return;
	if(0!=deviceMemId){
		_memoryAddress[deviceMemId]=device->freeSharedMemory( _memoryAddress[deviceMemId]);
	}
	device->_memoryID=0;
}

std::vector<g2s::spaceFrequenceMemoryAddress> SharedMemoryManager::adressSharedMemory(int memID){
	return _memoryAddress[memID];
}
