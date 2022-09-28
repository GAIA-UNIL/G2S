#ifndef SNESIM_HPP
#define SNESIM_HPP

#include "DataImage.hpp"
#include "utils.hpp"
#include <fstream>


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
	memset((void*)(tree.data()),0,1<<(pathPositionArray[0].size()+1)*pathPositionArray.size()*sizeof(snesimTreeElement<2>));

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
void searchHist(unsigned int *count, std::vector<snesimTreeElement<nbClass>> &tree,std::vector<unsigned char> &signature, unsigned positionInTree, int inPathIndex, int depth, bool all=false){
	if(inPathIndex==depth){ // the end of the recursion 
		// fprintf(stderr, "%d, loc %d %d\n", positionInTree, tree[positionInTree].count[0],tree[positionInTree].count[1] );
		for (int i = 0; i < nbClass; ++i)
		{
			count[i]+=tree[positionInTree].count[i];
		}
	}
	else{
		if(signature[inPathIndex]==nbClass || all) {
			for (int i = 0; i < nbClass; ++i)
			{
				if(tree[positionInTree].node[i]!=0)
				searchHist(count, tree, signature, tree[positionInTree].node[i], inPathIndex+1, depth,all);
			}
		}
		else {
			if(tree[positionInTree].node[signature[inPathIndex]]==0){
				searchHist(count, tree, signature, positionInTree, inPathIndex, depth,true);	
			}
			searchHist(count, tree, signature, tree[positionInTree].node[signature[inPathIndex]], inPathIndex+1, depth,false);
		}
	}
}


template <size_t nbClass>
void snesimSimulation(FILE *logFile,g2s::DataImage &di, std::vector<snesimTreeElement<nbClass>> &tree, bool extendTree,
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
		unsigned level=unsigned(floor(path._data[currentCell]));
		// fprintf(stderr, "%d, %f\n",level, path._data[currentCell]);
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
					if((di.indexWithDelta(dataIndex, currentCell, pathPositionArray[level][inPathIndex], externalMemory4IndexComputation[moduleID])||circularSim) && (posterioryPath[dataIndex]<=indexPath)){
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
				// for (int i = 0; i < pathPositionArray[level].size(); ++i)
				// {
				// 	fprintf(stderr, "%d,%d;\t", pathPositionArray[level][i][0],pathPositionArray[level][i][1]);
				// }
				// fprintf(stderr, "\n");
				// for (int i = 0; i < pathPositionArray[level].size(); ++i)
				// {
				// 	fprintf(stderr, "%d, \t", signature[i]);
				// }
				// fprintf(stderr, "\n");
				searchHist(count,tree,signature,level+1, 1, pathPositionArray[level].size());
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
			//fprintf(stderr, "%d %d\n", count[0],count[1]);
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

template <size_t nbClass>
bool loadTree(std::vector<snesimTreeElement<nbClass>> &trees,std::vector<std::vector<std::vector<int> > > &pathPositionArray, std::vector<std::string> sourceFileNameVector, bool extendTree){
	sourceFileNameVector.push_back(std::to_string(pathPositionArray.size()));
	sourceFileNameVector.push_back(std::to_string(pathPositionArray[0].size()));
	std::string saveFilename= std::accumulate(std::begin(sourceFileNameVector), std::end(sourceFileNameVector), std::string(),
                                [](std::string &ss, std::string &s)
                                {
                                    return ss.empty() ? s : ss + "_" + s;
                                });
	saveFilename=std::string("./data/")+saveFilename+std::string((extendTree?".etree":".tree"));
	std::ifstream treeFile=std::ifstream(saveFilename, std::ios::in | std::ios::binary);
	if (treeFile)
	{
	    treeFile.seekg (0, std::ios::end);
		long unsigned length = treeFile.tellg();
		fprintf(stderr, "%d\n", length);
		treeFile.seekg (0, std::ios::beg);
		trees.resize(length/sizeof(snesimTreeElement<nbClass>));
		treeFile.read((char*)trees.data(),trees.size()*sizeof(snesimTreeElement<nbClass>));
	    return true;
	}else{
		return false;
	}
}

template <size_t nbClass>
void saveTree(std::vector<snesimTreeElement<nbClass>> &trees,std::vector<std::vector<std::vector<int> > > &pathPositionArray, std::vector<std::string> sourceFileNameVector, bool extendTree){
	sourceFileNameVector.push_back(std::to_string(pathPositionArray.size()));
	sourceFileNameVector.push_back(std::to_string(pathPositionArray[0].size()));
	std::string saveFilename= std::accumulate(std::begin(sourceFileNameVector), std::end(sourceFileNameVector), std::string(),
                                [](std::string &ss, std::string &s)
                                {
                                    return ss.empty() ? s : ss + "_" + s;
                                });
	saveFilename=std::string("./data/")+saveFilename+std::string((extendTree?".etree":".tree"));
    std::ofstream treeFile (saveFilename, std::ios::out | std::ios::binary);
    treeFile.write ((const char*)trees.data(), trees.size()*sizeof(snesimTreeElement<nbClass>));
}

#endif // SNESIM_HPP