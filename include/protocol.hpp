/*
 * G2S (c) by Mathieu Gravey (gravey.mathieu@gmail.com)
 * 
 * G2S is licensed under a
 * Creative Commons Attribution-NonCommercial 4.0 International License.
 * 
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
 */

enum taskType{
    JOB=1,
    UPLOAD=2,
    DOWNLOAD=3,
    EXIST=4,
    STATUS=5,
    DURATION=6,
    KILL=7,
    SHUTDOWN=10
};

struct infoContainer{
    int version;
    taskType task;
};
