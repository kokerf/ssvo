#include "time_tracing.hpp"
#include <thread>
#include <iomanip>

using ssvo::TimeTracing;
int main()
{

    const std::string t1 = "onefafafafsafasfafafaf";
    const std::string t2 = "twodfafadfaf";
    const std::string t3 = "t";
    const std::string t4 = "t44fhakfjhalsfjlasjflajlfnbeufhiohtbaikfnlkakjflafj";

    TimeTracing::TraceNames traces;
    traces.push_back(t1);
    traces.push_back(t2);
    traces.push_back(t3);
    traces.push_back(t4);
    TimeTracing::TraceNames logs;
    logs.push_back("status");
    logs.push_back("type");

    TimeTracing timetracing("test_timer", "/tmp", traces, logs);
    timetracing.startTimer(t1);
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
    timetracing.stopTimer(t1);

    timetracing.startTimer(t2);
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
    timetracing.stopTimer(t2);

    timetracing.startTimer(t3);
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
    timetracing.stopTimer(t3);

    const int N = 10000;
    ssvo::MillisecondTimer timer;
    timer.start();
    for(int i = 0 ; i < N; i++)
    {
        timetracing.startTimer(t4);
        timetracing.stopTimer(t4);
    }
    const double duration = timer.stop();

    timetracing.log("status", 3);

    std::cout << std::fixed << std::setprecision(9)
              << timetracing.getTimer(t1) * 1000 << ", "
              << timetracing.getTimer(t2) * 1000 << ", "
              << timetracing.getTimer(t3) * 1000 << std::endl;

    std::cout << "dur " << duration / N << std::endl;

    timetracing.writeToFile();


    return 0;
}

