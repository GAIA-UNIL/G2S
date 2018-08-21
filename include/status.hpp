/*
 * G2S (c) by Mathieu Gravey (gravey.mathieu@gmail.com)
 * 
 * G2S is licensed under a
 * Creative Commons Attribution-NonCommercial 4.0 International License.
 * 
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
 */

#ifndef STATUS_HPP
#define STATUS_HPP

#include <iostream>

int lookForStatus(void* data, size_t dataSize);
int lookForDuration(void* data, size_t dataSize);

#endif // STATUS_HPP