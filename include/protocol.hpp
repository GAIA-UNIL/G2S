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

enum taskType{
    JOB=1,
    UPLOAD=2,
    DOWNLOAD=3,
    EXIST=4,
    PROGESSION=5,
    DURATION=6,
    KILL=7,
    UPLOAD_JSON=8,
    DOWNLOAD_JSON=9,
    SHUTDOWN=10,
    SERVER_STATUS=11,
    JOB_STATUS=12,
    DOWNLOAD_TEXT=13
};

struct infoContainer{
    int version;
    taskType task;
};
