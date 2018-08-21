/*
 * G2S (c) by Mathieu Gravey (gravey.mathieu@gmail.com)
 * 
 * G2S is licensed under a
 * Creative Commons Attribution-NonCommercial 4.0 International License.
 * 
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
 */

#ifndef CVT_ZMQ2WS_HPP
#define CVT_ZMQ2WS_HPP
#include <atomic>
#include <string>
#define ASIO_STANDALONE

void cvtServer(char* from, char* to, std::atomic<bool> &serverRun, std::atomic<bool> &done);


#endif // CVT_ZMQ2WS_HPP