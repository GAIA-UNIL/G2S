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

#include "DataImage.hpp"
#include "picosha2.h"
#include <vector>
#if defined(_WIN32)
  #include <io.h>
  #include <process.h>
#else
  #include <unistd.h>
  #include <sys/stat.h>
  #include <sys/time.h>
#endif
#include <algorithm>
#include <atomic>
#include <cstdint>
#ifndef _WIN32
	#include "zlib.h"
#endif
#if defined(__APPLE__)
	#include <CommonCrypto/CommonDigest.h>
#endif
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
	#include <thread>
	#define G2S_PARALLEL_GZIP 1
#endif

namespace {
const size_t MaxRawPayloadSize = g2s::DataImage::MaxSerializedBytes;

#ifndef _WIN32
bool readCompressedFile(const char* filename, std::vector<char>& output){
	gzFile dataFile=gzopen(filename,"rb");
	if(!dataFile) return false;

	output.clear();
	char buffer[16384];
	bool ok=true;
	while(true){
		int bytesRead=gzread(dataFile, buffer, sizeof(buffer));
		if(bytesRead > 0){
			if(output.size()+size_t(bytesRead) > MaxRawPayloadSize){
				ok=false;
				break;
			}
			output.insert(output.end(), buffer, buffer+bytesRead);
			continue;
		}
		if(bytesRead == 0) break;
		ok=false;
		break;
	}
	gzclose(dataFile);
	return ok;
}
#endif

bool readPlainFile(const char* filename, std::vector<char>& output){
	FILE* dataFile=fopen(filename,"rb");
	if(!dataFile) return false;
	if(fseek(dataFile, 0L, SEEK_END) != 0){
		fclose(dataFile);
		return false;
	}
	long fileSize=ftell(dataFile);
	if(fileSize < 0 || size_t(fileSize) > MaxRawPayloadSize){
		fclose(dataFile);
		return false;
	}
	rewind(dataFile);
	output.resize(size_t(fileSize));
	if(!output.empty() && fread(output.data(), 1, output.size(), dataFile) != output.size()){
		fclose(dataFile);
		output.clear();
		return false;
	}
	fclose(dataFile);
	return true;
}

char* copyValidatedRaw(const std::vector<char>& data, size_t* outSize){
	if(!g2s::DataImage::validateSerializedData(data.data(), data.size())) return nullptr;
	char* ptr=(char*)malloc(data.size());
	if(!ptr) return nullptr;
	memcpy(ptr, data.data(), data.size());
	if(outSize) *outSize=data.size();
	return ptr;
}
}

#ifndef _WIN32
namespace {
// A name unique to this process and call, next to `path`, for atomic replacement.
void temporaryName(const char* path, char* out, size_t size){
	static std::atomic<unsigned> counter(0);
	snprintf(out, size, "%s.tmp.%ld.%u", path, (long)getpid(), counter++);
}

bool isSymbolicLink(const char* path){
	struct stat info;
	return lstat(path, &info) == 0 && S_ISLNK(info.st_mode);
}
}
#endif

namespace {
// SHA-256 of the serialized image (its content address). CommonCrypto on Apple
// platforms (hardware SHA, about 10x picosha2); picosha2 elsewhere. Same digest.
void sha256(const char* data, size_t size, unsigned char* digest){
	#if defined(__APPLE__)
	CC_SHA256_CTX context;
	CC_SHA256_Init(&context);
	const size_t step=size_t(1)<<30; // CC_LONG is 32 bits
	for(size_t offset=0; offset<size; offset+=step){
		const size_t n=std::min(step, size-offset);
		CC_SHA256_Update(&context, data+offset, CC_LONG(n));
	}
	CC_SHA256_Final(digest, &context);
	#else
	std::vector<unsigned char> hash(32);
	picosha2::hash256((const unsigned char*)data, ((const unsigned char*)data)+size, hash.begin(), hash.end());
	memcpy(digest, hash.data(), 32);
	#endif
}

#ifndef _WIN32
// True when the content file `path` already exists; its access and
// modification times are then set to now, as rewriting it would (the server
// removes data files by age), so that it can be reused instead of rewritten.
bool reuseContentFile(const char* path){
	struct stat info;
	if(stat(path, &info)!=0 || !S_ISREG(info.st_mode)) return false;
	return utimes(path, nullptr)==0;
}
#endif

#ifdef G2S_PARALLEL_GZIP
// Large images are compressed by chunks on several threads into ONE standard
// gzip member (the pigz method): each chunk is raw-deflated at the default
// level with the previous 32 KiB of input as its dictionary and ends with a
// sync flush (the last one with a final block), so the concatenation of the
// chunks is a single valid deflate stream; the CRC-32 of the chunks are
// combined. Any gzip reader (zlib gzread, gzip, Python) reads it, the content
// is the same and the compressed size within 0.1% of a serial gzopen("wb").
const size_t ParallelGzipMinimum=size_t(8)<<20;
const size_t ParallelGzipChunk=size_t(4)<<20;

bool writeGzipParallel(const char* path, const char* data, size_t size){
	const size_t chunks=(size+ParallelGzipChunk-1)/ParallelGzipChunk;
	std::vector<std::vector<unsigned char> > compressed(chunks);
	std::vector<uLong> crcs(chunks, 0);
	std::vector<char> ok(chunks, 0);
	std::atomic<size_t> next(0);
	auto work=[&](){
		for(size_t c=next++; c<chunks; c=next++){
			const size_t begin=c*ParallelGzipChunk, length=std::min(ParallelGzipChunk, size-begin);
			const bool last=(c+1==chunks);
			z_stream stream;
			memset(&stream, 0, sizeof(stream));
			if(deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY)!=Z_OK) continue;
			if(begin>0){
				const size_t window=std::min<size_t>(32768, begin);
				deflateSetDictionary(&stream, (const Bytef*)(data+begin-window), uInt(window));
			}
			std::vector<unsigned char>& out=compressed[c];
			out.resize(deflateBound(&stream, uLong(length))+64);
			stream.next_in=(Bytef*)(data+begin);
			stream.avail_in=uInt(length);
			stream.next_out=out.data();
			stream.avail_out=uInt(out.size());
			const int status=deflate(&stream, last ? Z_FINISH : Z_SYNC_FLUSH);
			const bool done=last ? status==Z_STREAM_END : (status==Z_OK && stream.avail_in==0 && stream.avail_out>0);
			out.resize(out.size()-stream.avail_out);
			deflateEnd(&stream);
			crcs[c]=crc32(crc32(0L, Z_NULL, 0), (const Bytef*)(data+begin), uInt(length));
			ok[c]=done;
		}
	};
	const unsigned threads=unsigned(std::max<size_t>(1, std::min<size_t>(chunks, std::max(1u, std::thread::hardware_concurrency()))));
	std::vector<std::thread> pool;
	for(unsigned t=1; t<threads; ++t) pool.emplace_back(work);
	work();
	for(std::thread& thread : pool) thread.join();
	uLong crc=crc32(0L, Z_NULL, 0);
	for(size_t c=0; c<chunks; ++c){
		if(!ok[c]) return false;
		const size_t length=std::min(ParallelGzipChunk, size-c*ParallelGzipChunk);
		crc=crc32_combine(crc, crcs[c], z_off_t(length));
	}
	FILE* file=fopen(path, "wb");
	if(!file) return false;
	// gzip header as zlib's gzopen writes it: no name, no time, OS = Unix.
	const unsigned char header[10]={0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3};
	bool written=fwrite(header, 1, sizeof(header), file)==sizeof(header);
	for(size_t c=0; c<chunks && written; ++c)
		written=fwrite(compressed[c].data(), 1, compressed[c].size(), file)==compressed[c].size();
	unsigned char trailer[8];
	const uint32_t isize=uint32_t(size);
	for(int i=0; i<4; ++i){ trailer[i]=(unsigned char)(crc>>(8*i)); trailer[4+i]=(unsigned char)(isize>>(8*i)); }
	written=written && fwrite(trailer, 1, sizeof(trailer), file)==sizeof(trailer);
	return (fclose(file)==0) && written;
}
#endif
}

// Links the name of an image (fullFilename, e.g. /tmp/G2S/data/im_1_<id>.bgrid.gz)
// to its content-addressed file (outputFullFilename, named by the SHA-256 of
// the content). Writing a name again replaces its link atomically: the new
// link is made under a temporary name and renamed over the old one, so a
// reader sees the old image or the new one, never a missing or stale mix, and
// concurrent writers of the same name leave one of their links. Only links
// are replaced: a regular file under that name (a content-addressed file) is
// left as it is. The link of the other compression (.bgrid vs .bgrid.gz),
// which loadRawData would otherwise prefer or fall back to, is removed.
void createLink(char* outputFullFilename, char* fullFilename){
	#ifndef _WIN32
	if(strcmp(outputFullFilename, fullFilename)==0) return; // the name is the content hash itself
	struct stat info;
	if(lstat(fullFilename, &info)==0 && !S_ISLNK(info.st_mode)) return;
	char temporary[4096];
	temporaryName(fullFilename, temporary, sizeof(temporary));
	if(symlink(outputFullFilename, temporary)!=0) return;
	if(rename(temporary, fullFilename)!=0){
		unlink(temporary);
		return;
	}
	char other[4096];
	const size_t length=strlen(fullFilename);
	if(length>3 && strcmp(fullFilename+length-3, ".gz")==0) snprintf(other, sizeof(other), "%.*s", int(length-3), fullFilename);
	else snprintf(other, sizeof(other), "%s.gz", fullFilename);
	if(isSymbolicLink(other)) unlink(other);
	#endif
}


char* loadRawData(const char * hash, size_t* outSize){
	char filename[4096];
	if(outSize) *outSize=0;
	std::vector<char> data;

	//fprintf(stderr, "look For File %s \n",hash);

	#ifndef _WIN32
	snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid.gz",hash);
	if(g2s::file_exist(filename) && readCompressedFile(filename, data)){
		char* raw=copyValidatedRaw(data, outSize);
		if(raw) return raw;
	}
	#endif
	snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid",hash);
	//fprintf(stderr, "%s\n",filename );
	if(g2s::file_exist(filename) && readPlainFile(filename, data)){
		char* raw=copyValidatedRaw(data, outSize);
		if(raw) return raw;
	}

	fprintf(stderr, "file not found or invalid\n");
	return nullptr;
}

char* writeRawData(char* data, bool compresed){
	if(!data) return nullptr;
	size_t fullSize=*((size_t*)data);
	if(!g2s::DataImage::validateSerializedData(data, fullSize)) return nullptr;
	std::vector<unsigned char> hash(32);
	sha256(data, fullSize, hash.data());

	char* hashInHexa=(char*)malloc(65);
	memset(hashInHexa,0,65);
	for (int i = 0; i < 32; ++i)
	{
		snprintf(hashInHexa+2*i,65-2*i,"%02x",hash.data()[i]);
	}
	//fprintf(stderr, "%s\n", hashInHexa);

	char filename[4096];

	// The content file is written under a temporary name and renamed, so that
	// a reader (or a link to it) never sees a partly written file, also when
	// another job is writing the same content at the same time.
	#ifndef _WIN32
	char temporary[4096];
	if(compresed) {
		snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid.gz",hashInHexa);
		// The name is the hash of the content: an existing file already holds it.
		if(reuseContentFile(filename)) return hashInHexa;
		temporaryName(filename, temporary, sizeof(temporary));
		#ifdef G2S_PARALLEL_GZIP
		if(fullSize>=ParallelGzipMinimum) {
			if(writeGzipParallel(temporary, data, fullSize)) {
				if(rename(temporary, filename)!=0) unlink(temporary);
			} else {
				unlink(temporary);
			}
			return hashInHexa;
		}
		#endif
		gzFile dataFile=gzopen(temporary,"wb");
		if(dataFile) {
			const bool written=gzwrite(dataFile, data, unsigned(fullSize))==int(fullSize);
			if(gzclose(dataFile)==Z_OK && written) {
				if(rename(temporary, filename)!=0) unlink(temporary);
			} else {
				unlink(temporary);
			}
		}
	}
	else {
		snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid",hashInHexa);
		if(reuseContentFile(filename)) return hashInHexa;
		temporaryName(filename, temporary, sizeof(temporary));
		FILE* dataFile=fopen(temporary,"wb");
		if(dataFile) {
			const bool written=fwrite(data, 1, fullSize, dataFile)==fullSize;
			if(fclose(dataFile)==0 && written) {
				if(rename(temporary, filename)!=0) unlink(temporary);
			} else {
				unlink(temporary);
			}
		}
	}
	#else
	{
		snprintf(filename,4096,"/tmp/G2S/data/%s.bgrid",hashInHexa);

		FILE* dataFile=fopen(filename,"wb");
		if(dataFile) {
			fwrite (data, 1, fullSize, dataFile);
			fclose(dataFile);
		}
	}
	#endif
	return hashInHexa;
}
