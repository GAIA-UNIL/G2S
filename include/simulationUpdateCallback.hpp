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

#ifndef SIMULATION_UPDATE_CALLBACK_HPP
#define SIMULATION_UPDATE_CALLBACK_HPP

#include "pathIndexType.hpp"

enum class g2s_simulation_update_kind : unsigned char {
	Vector = 'V',
	Full = 'F'
};

using g2s_simulation_update_callback_t = void (*)(g2s_simulation_update_kind kind,
	g2s_path_index_t localIndex, unsigned variableIndex, void* userData);

#endif // SIMULATION_UPDATE_CALLBACK_HPP
