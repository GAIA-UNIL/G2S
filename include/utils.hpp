/*
 * Mathieu Gravey
 * Copyright (C) 2017 Mathieu Gravey
 * 
 * This program is protected software: you can not redistribute, use, and/or modify it
 * without the explicit accord from the author : Mathieu Gravey, gravey.mathieu@gmail.com
 *
 */
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>
 
#if _OPENMP
	#include <omp.h>
#endif

#ifndef G2S_UTILS
#define G2S_UTILS

namespace g2s {

	enum DistanceType{
		KERNEL,
		EUCLIDIEN,
		MANAHTTAN
	};

	enum KernelType{
		UNIFORM,
		TRIANGULAR,
		EXPONENTIAL,
		EPANECHNIKOV,
		QUARTIC,
		TRIWEIGHT,
		TRICUBE,
		GAUSSIAN,
		COSINE,
		LOGISTIC,
		SIGMOID,
		SILVERMAN
	};

	#define clamp( x,  a,  b) ( (x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))

	template <typename T>
	struct positionVector2D{
	  T x,y;
	 // unsigned int sourceIndex;
	};

	template <typename T>
	struct positionVector3D{
	  T x,y,z;
	  //unsigned int sourceIndex;
	};

	struct spaceFrequenceMemoryAddress{
		void* space,*fft;
	};

	inline unsigned long rdtscp(int *chip, int *core)
	{
	#ifdef XEON
		unsigned a, d, c;
		__asm__ volatile("rdtscp" : "=a" (a), "=d" (d), "=c" (c));
		*chip = (c & 0xFFF000)>>12;
		*core = c & 0xFFF;
		#ifdef _OPENMP
			//*core=omp_get_thread_num();
		#else
			*core=0;
		#endif
		return ((unsigned long)a) | (((unsigned long)d) << 32);
	#else
		*chip=0;
		#ifdef _OPENMP
			*core=omp_get_thread_num();
		#else
			*core=0;
		#endif
		return 0;
	#endif
	}

	inline   void max_rdtscp(int *nb_chip, int *nb_core)
	{
		int maxCores=0;
		int maxNodes=0;
		#ifdef _OPENMP
			#pragma omp parallel default(none) reduction(max:maxCores,maxNodes)
			{
				rdtscp(&maxNodes,&maxCores);
			}
		#endif
		*nb_chip=maxNodes+1;
		*nb_core=maxCores+1;
	}

	inline std::multimap<std::string, std::string> argumentReader(int argc, char const *argv[]){
		std::multimap<std::string, std::string> arg;
		int argic=0;
		while(argic<argc)
		{
			if (argv[argic][0]=='-')
			{
				bool minOne=false;
				int name=argic;
				argic++;
				while((argic<argc)&&(argv[argic][0]!='-'))
				{
					arg.insert(std::pair<std::string,std::string>(std::string(argv[name]),std::string(argv[argic])));
					argic++;
					minOne=true;
				}
				if(!minOne)arg.insert(std::pair<std::string,std::string>(std::string(argv[name]),std::string()));
			}
			else{
				argic++;
			}
		}
		return arg;
	}

	inline int file_exist (char *filename)
	{
	  struct stat   buffer;
	  return (stat (filename, &buffer) == 0);
	}

} 

#endif