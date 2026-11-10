/*
 * Browser-only OpenMP microtask dispatcher for QS's large captured contexts.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifdef __EMSCRIPTEN_PTHREADS__

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {

using Microtask = void (*)(int*, int*, ...);

template<size_t... Indices>
void invokeMicrotask(Microtask function,
	int* globalThreadId,
	int* localThreadId,
	void** arguments,
	std::index_sequence<Indices...>){
	using ExactMicrotask = void (*)(
		int*,
		int*,
		decltype((void)Indices, static_cast<void*>(nullptr))...
	);
	reinterpret_cast<ExactMicrotask>(function)(
		globalThreadId,
		localThreadId,
		arguments[Indices]...
	);
}

} // namespace

extern "C" int __kmp_invoke_microtask(
	Microtask function,
	int globalThreadId,
	int localThreadId,
	int argumentCount,
	void* arguments[]){
	int global=globalThreadId;
	int local=localThreadId;

#define G2S_INVOKE_CASE(count) \
	case count: invokeMicrotask(function,&global,&local,arguments,std::make_index_sequence<count>()); break
	switch(argumentCount){
		G2S_INVOKE_CASE(0);
		G2S_INVOKE_CASE(1);
		G2S_INVOKE_CASE(2);
		G2S_INVOKE_CASE(3);
		G2S_INVOKE_CASE(4);
		G2S_INVOKE_CASE(5);
		G2S_INVOKE_CASE(6);
		G2S_INVOKE_CASE(7);
		G2S_INVOKE_CASE(8);
		G2S_INVOKE_CASE(9);
		G2S_INVOKE_CASE(10);
		G2S_INVOKE_CASE(11);
		G2S_INVOKE_CASE(12);
		G2S_INVOKE_CASE(13);
		G2S_INVOKE_CASE(14);
		G2S_INVOKE_CASE(15);
		G2S_INVOKE_CASE(16);
		G2S_INVOKE_CASE(17);
		G2S_INVOKE_CASE(18);
		G2S_INVOKE_CASE(19);
		G2S_INVOKE_CASE(20);
		G2S_INVOKE_CASE(21);
		G2S_INVOKE_CASE(22);
		G2S_INVOKE_CASE(23);
		G2S_INVOKE_CASE(24);
		G2S_INVOKE_CASE(25);
		G2S_INVOKE_CASE(26);
		G2S_INVOKE_CASE(27);
		G2S_INVOKE_CASE(28);
		G2S_INVOKE_CASE(29);
		G2S_INVOKE_CASE(30);
		G2S_INVOKE_CASE(31);
		G2S_INVOKE_CASE(32);
		G2S_INVOKE_CASE(33);
		G2S_INVOKE_CASE(34);
		G2S_INVOKE_CASE(35);
		G2S_INVOKE_CASE(36);
		G2S_INVOKE_CASE(37);
		G2S_INVOKE_CASE(38);
		G2S_INVOKE_CASE(39);
		G2S_INVOKE_CASE(40);
		default:
			std::fprintf(stderr,"G2S OpenMP microtask has too many arguments: %d\n",argumentCount);
			std::abort();
	}
#undef G2S_INVOKE_CASE

	return 1;
}

#endif
