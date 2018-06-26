#include <direct.h> 
#include "glog/logging.h"

int main(int argc, char const *argv[])
{
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_prefix = true;
    FLAGS_log_dir = std::string(getcwd(NULL,0))+"/../log";

    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "SSVO test glog";
    LOG(INFO) << "Log dir:" << FLAGS_log_dir;
    LOG(INFO) << "It is an info.";
    LOG(WARNING) << "It is a warning.";
    LOG(ERROR) << "It is an error.";

    return 0;
}