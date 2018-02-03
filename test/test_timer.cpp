#include "time_tracing.hpp"
#include <thread>

using ssvo::TimeTracing;
int main()
{

    TimeTracing::TraceNames traces;
    traces.push_back("one");
    traces.push_back("two");
    traces.push_back("three");
    TimeTracing::TraceNames logs;
    logs.push_back("status");
    logs.push_back("type");

    TimeTracing timetracing("test_timer", "/tmp", traces, logs);
    timetracing.startTimer("one");
    std::this_thread::sleep_for(std::chrono::microseconds(10000));
    timetracing.stopTimer("one");

    timetracing.startTimer("two");
    std::this_thread::sleep_for(std::chrono::microseconds(20000));
    timetracing.stopTimer("two");

    timetracing.log("status", 3);

    timetracing.writeToFile();


    return 0;
}

