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
