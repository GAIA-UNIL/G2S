enum taskType{
    JOB=1,
    UPLOAD=2,
    DOWNLOAD=3,
    EXIST=4,
    STATUS=5,
    KILL=6,
    SHUTDOWN=10
};

struct infoContainer{
    int version;
    taskType task;
};
