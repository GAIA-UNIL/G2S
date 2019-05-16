/* Copyright © 2019 Mathieu Gravey <gravey.mathieu@gmail.com>
 * This program is free software. It comes without any warranty, to
 * the extent permitted by applicable law. You can redistribute it
 * and/or modify it under the terms of the Do What The Fuck You Want
 * To Public License, Version 2, as published by Sam Hocevar. See
 * http://www.wtfpl.net/ for more details. */

/*
	original file at https://github.com/mgravey/MexInterrupt
*/

#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>
#include <mex.h>

class std::shared_ptr<struct InterruptStateData> const & utGetInterruptStateData(void);

class utInterruptState
{
	
public:
	static bool IsInterruptPending();	
	static bool IsInterruptPending(std::shared_ptr<struct InterruptStateData> const& state);
};

class mexInterrupt
{
public:
	inline static std::future<void> startInterruptCheck(std::atomic<bool> &done){
		
#if (!defined(MATLAB_COMPILE_VERSION) || (MATLAB_COMPILE_VERSION>=0x2018b))
		if (runtimeMatlabVersion()>=0x2018b)
		{
			auto interruptState=utGetInterruptStateData();
			return std::async(std::launch::async, mexInterrupt::testIfInterupted, std::ref(done), interruptState);
		}else
#endif
		{
			return std::async(std::launch::async, mexInterrupt::testIfInteruptedOld, std::ref(done));
		}
		
	}

	inline static long runtimeMatlabVersion(){
		mxArray* runtimeVersion;
		mxArray* argmumentToVersion=mxCreateString("-release");
		mexCallMATLAB(1,&runtimeVersion,1, &argmumentToVersion, "version");
		long matlabVersionRuntime = strtol(mxArrayToString(runtimeVersion), NULL, 16);
		mxDestroyArray(runtimeVersion);
		mxDestroyArray(argmumentToVersion);
		//printf("%x\n",matlabVersionRuntime);
		return matlabVersionRuntime;
	}

private:
	inline static void testIfInterupted(std::atomic<bool> &done, std::shared_ptr<InterruptStateData> const &interruptState)
	{
		while (!done){
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
			if(utInterruptState::IsInterruptPending(interruptState)){
				done=true;
			}
		}
	}

	inline static void testIfInteruptedOld(std::atomic<bool> &done)
	{
		while (!done){
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
			if(utInterruptState::IsInterruptPending()){
				done=true;
			}
		}
	}
};
