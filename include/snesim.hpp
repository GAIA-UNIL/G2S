#ifndef SNESIM_HPP
#define SNESIM_HPP

#include "DataImage.hpp"
#include "utils.hpp"


template <size_t nbClass>
union snesimTreeElement {
   unsigned int node[nbClass+1];
   unsigned int count[nbClass];
   unsigned int numberOfLevel;
};

template <size_t nbClass>
void addEllement(std::vector<snesimTreeElement<nbClass>> &tree, unsigned int position, unsigned char level, std::vector<unsigned char> &signature, bool extendTree=false){
	if(level==signature.size()){
		tree[position].count[signature[0]]++;
	}else{
		if(tree[position].node[signature[level]]==0){

			#pragma omp critical (updateTree)
			{
				if(tree[position].node[signature[level]]==0){
					tree[position].node[signature[level]]=tree.size();
					tree.emplace_back();
				}
			}
		}
		addEllement(tree, tree[position].node[signature[level]],   level+1, signature, extendTree);
		if(extendTree){
			if(tree[position].node[nbClass]==0){
				#pragma omp critical (updateTree)
				{
					if(tree[position].node[nbClass]==0){
						tree[position].node[nbClass]=tree.size();
						tree.emplace_back();
					}
				}
			}
			addEllement(tree, tree[position].node[nbClass], level+1, signature, extendTree);
		}
	}
}

template <size_t nbClass>
std::vector<snesimTreeElement<nbClass>> createTree(std::vector<g2s::DataImage> &TIs, std::vector<std::vector<std::vector<int> > > &pathPositionArray, unsigned nbThreads, bool extendTree=false){
	Timer timer;
	timer.start();

	std::vector<snesimTreeElement<nbClass>> tree(pathPositionArray.size()+1);
	tree.reserve(1<<(pathPositionArray[0].size()+1)*pathPositionArray.size());
	tree[0].numberOfLevel=pathPositionArray.size();

	int *externalMemory[nbThreads];
	for (int i = 0; i < nbThreads; ++i)
	{
		externalMemory[i]=new int[pathPositionArray[0][0].size()];
	}
	#pragma omp parallel num_threads(nbThreads) default(none) firstprivate(pathPositionArray, externalMemory, __stderrp, extendTree) shared(TIs,tree)
	{
		//#pragma omp single
		for (int pathIndex = 0; pathIndex < pathPositionArray.size(); ++pathIndex)
		{
			std::vector<unsigned char> signature(pathPositionArray[0].size());
			for (int imageIndex = 0; imageIndex < TIs.size(); ++imageIndex)
			{
				#pragma omp for 
				for (int pixelIndex = 0; pixelIndex < TIs[imageIndex].dataSize(); ++pixelIndex)
				{
					unsigned moduleID=0;
					#if _OPENMP
						moduleID=omp_get_thread_num();
					#endif
					// if(pixelIndex<100 || pixelIndex%1000==0)
					// 	fprintf(stderr, "%d\n", pixelIndex);
					for (int inPathIndex = 0; inPathIndex < pathPositionArray[pathIndex].size(); ++inPathIndex)
					{
						unsigned location;
						signature[inPathIndex]=(TIs[imageIndex].indexWithDelta(location, pixelIndex, pathPositionArray[pathIndex][inPathIndex], externalMemory[moduleID]) ? TIs[imageIndex]._data[location] : nbClass);
						addEllement(tree,pathIndex+1,1,signature,extendTree);
					}
				}
			}
		}
	}
	for (int i = 0; i < nbThreads; ++i)
	{
		delete[] externalMemory[i];
	}
	
	
	return tree;
}

template <size_t nbClass>
void snesimSimulation(FILE *logFile,g2s::DataImage &di, std::vector<snesimTreeElement<nbClass>> tree, bool extendTree,
  std::vector<std::vector<std::vector<int> > > &pathPositionArray, g2s::DataImage &path, unsigned* solvingPath, unsigned numberOfPointToSimulate,
 float* seedAray, std::vector<unsigned> numberNeighbor, unsigned maxN, g2s::DataImage *nii, unsigned nbThreads=1, bool circularSim=false){
	int displayRatio=std::max(numberOfPointToSimulate/100,1u);
	unsigned* posterioryPath=(unsigned*)malloc( sizeof(unsigned) * di.dataSize()/di._nbVariable);
	memset(posterioryPath,255,sizeof(unsigned) * di.dataSize()/di._nbVariable);
	for (unsigned int i = 0; i < numberOfPointToSimulate; ++i)
	{
		posterioryPath[solvingPath[i]]=i;
	}
	int* externalMemory4IndexComputation[nbThreads];
	for (int i = 0; i < nbThreads; ++i)
	{
		externalMemory4IndexComputation[i]=new int[di._dims.size()];
	}

	std::vector<unsigned char> signature(maxN);

	//#pragma omp parallel for num_threads(nbThreads) schedule(monotonic:dynamic,1) default(none) 
	//firstprivate(kernelAutoSelection,forceSimulation, kvi, nii, kii, displayRatio, circularSim, fullStationary, numberOfVariable,categoriesValues,numberOfPointToSimulate, \
		posterioryPath, solvingPath, seedAray, numberNeighbor, importDataIndex, logFile, ii, externalMemory4IndexComputation) shared( pathPositionArray, di, samplingModule, TIs, kernels)
	for (unsigned int indexPath = 0; indexPath < numberOfPointToSimulate; ++indexPath){

		unsigned currentCell=solvingPath[indexPath];
		float localSeed=seedAray[indexPath];
		bool withOnlyData=true;
		unsigned level=unsigned(floor(path._data[indexPath]));

		for (unsigned int i = 0; i < di._nbVariable; ++i)
		{
			withOnlyData&=!std::isnan(di._data[currentCell*di._nbVariable+i]);
		}

		if(withOnlyData) continue;

		unsigned moduleID=0;
		#if _OPENMP
			moduleID=omp_get_thread_num();
		#endif

		if(nii){
			numberNeighbor.clear();
			for (int i = 0; i < nii->_nbVariable; ++i)
			{
				numberNeighbor.push_back(int(nii->_data[currentCell*nii->_nbVariable+i]));
			}
		}


		{
			unsigned int count[nbClass]={0};
			if (!extendTree){
				for (int inPathIndex = 1; inPathIndex < pathPositionArray[level].size(); ++inPathIndex)
				{
					unsigned dataIndex;
					if(di.indexWithDelta(dataIndex, currentCell, pathPositionArray[level][inPathIndex], externalMemory4IndexComputation[moduleID])){
						float val;
						#pragma omp atomic read
						val=di._data[dataIndex];
						while (std::isnan(val)){
							std::this_thread::sleep_for(std::chrono::microseconds(250));
							#pragma omp atomic read
							val=di._data[dataIndex];
						}
						signature[inPathIndex]=(unsigned char)(val);
					}else{
						signature[inPathIndex]=nbClass;
					}
				}
				//searchHist(count,tree,signature);
			}
			else{
				unsigned positionInTree=level+1;
				for (int inPathIndex = 1; inPathIndex < pathPositionArray[level].size(); ++inPathIndex)
				{
					unsigned dataIndex;
					if((di.indexWithDelta(dataIndex, currentCell, pathPositionArray[level][inPathIndex], externalMemory4IndexComputation[moduleID])||circularSim) && (posterioryPath[dataIndex]<=indexPath)){
						float val;
						#pragma omp atomic read
						val=di._data[dataIndex];
						while (std::isnan(val)){
							std::this_thread::sleep_for(std::chrono::microseconds(250));
							#pragma omp atomic read
							val=di._data[dataIndex];
						}
						positionInTree=tree[positionInTree].node[(unsigned char)(val)];
					}else{
						positionInTree=tree[positionInTree].node[nbClass];
					}
				}
				for (int i = 0; i < nbClass; ++i)
				{
					count[i]=tree[positionInTree].count[i];
				}
			}

			// sampling

			unsigned int sumBin=0;
			for (int i = 0; i < nbClass; ++i)
			{
				sumBin+=count[i];
			}
			float sample=localSeed*sumBin;
			unsigned char simVal;
			sumBin=0;
			for (unsigned char i = 0; i < nbClass; ++i)
			{
				if (sumBin<sample)
				{
					simVal=i;
				}
				sumBin+=count[i];
			}

			#pragma omp atomic write
			di._data[currentCell]=simVal;
			if(indexPath%(displayRatio)==0)
				fprintf(logFile, "progress : %.2f%%\n",float(indexPath)/numberOfPointToSimulate*100);
		}
	}
}

#endif // SNESIM_HPP