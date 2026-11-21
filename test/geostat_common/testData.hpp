// Names and job ids of what the geostat test suites write, unique to the
// process, so that the suites can run at the same time as each other (and as
// another run of themselves), and the clean-up of what they wrote.
//
//   testName("sgs_di")  "t<pid>_sgs_di": an image name in /tmp/G2S/data, or a
//                       local file name (weight files, logs)
//   testId(k)           job id k (0 <= k < 1000) of this process; the
//                       executables write their outputs as im_<i>_<id>
//
// The first call of either registers removeTestData() with atexit: it removes
// the names of this process from /tmp/G2S/data and the current directory and
// the outputs im_<i>_<id> of its job ids. The content-addressed files these
// names link to (/tmp/G2S/data/<sha256>.bgrid.gz) are kept on purpose: they
// are shared by every writer of the same content (another run of the same
// test writes identical images, the server may serve them), so removing one
// could break a writer that is still running. Their number does not grow
// from run to run (the tests are deterministic), and the G2S server removes
// old data files.

#ifndef G2S_GEOSTAT_TEST_DATA_HPP
#define G2S_GEOSTAT_TEST_DATA_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace geostat_test {

inline unsigned processTag() { return unsigned(getpid()); }

// First job id of this process: pid * 1000 (pids are below 4194304 on Linux and 100000 on macOS).
inline unsigned firstTestId() { return (processTag() % 4000000u) * 1000u; }

inline std::string namePrefix() { return "t" + std::to_string(processTag()) + "_"; }

inline bool startsWith(const char* text, const std::string& prefix) { return std::strncmp(text, prefix.c_str(), prefix.size()) == 0; }

// True when `entry` is im_<i>_<id>[.<suffix>] with id one of this process.
inline bool isOwnOutput(const char* entry) {
	unsigned output = 0, id = 0;
	int consumed = 0;
	if (std::sscanf(entry, "im_%u_%u%n", &output, &id, &consumed) != 2) return false;
	if (entry[consumed] != '\0' && entry[consumed] != '.') return false;
	return id >= firstTestId() && id < firstTestId() + 1000u;
}

inline void removeMatching(const char* directory, bool outputs) {
	DIR* dir = opendir(directory);
	if (!dir) return;
	const std::string prefix = namePrefix();
	std::string path;
	while (dirent* entry = readdir(dir)) {
		if (!startsWith(entry->d_name, prefix) && !(outputs && isOwnOutput(entry->d_name))) continue;
		path = std::string(directory) + "/" + entry->d_name;
		struct stat info;
		if (lstat(path.c_str(), &info) == 0 && (S_ISLNK(info.st_mode) || S_ISREG(info.st_mode))) unlink(path.c_str());
	}
	closedir(dir);
}

inline void removeTestData() {
	removeMatching("/tmp/G2S/data", true);
	removeMatching(".", false);
}

inline void registerCleanup() {
	static bool registered = false;
	if (registered) return;
	registered = true;
	mkdir("/tmp/G2S", 0770);
	mkdir("/tmp/G2S/data", 0770);
	std::atexit(removeTestData);
}

inline std::string testName(const std::string& stem) {
	registerCleanup();
	return namePrefix() + stem;
}

inline unsigned testId(unsigned k) {
	registerCleanup();
	if (k >= 1000) { std::fprintf(stderr, "testId(%u): k must be below 1000\n", k); std::abort(); }
	return firstTestId() + k;
}

inline std::string testIdText(unsigned k) { return std::to_string(testId(k)); }

} // namespace geostat_test

#endif // G2S_GEOSTAT_TEST_DATA_HPP
