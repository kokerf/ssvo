#ifndef _SSVO_TIME_TRACING_HPP_
#define _SSVO_TIME_TRACING_HPP_

#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <list>
#include <unordered_map>

namespace ssvo
{

template<typename T>
class Timer
{
public:

    inline void start()
    {
        start_ = std::chrono::steady_clock::now();
    }

    inline double stop()
    {
        duration_ = std::chrono::steady_clock::now() - start_;
        return duration_.count();
    }

    inline void reset()
    {
        duration_ = std::chrono::duration<double, T>(0.0);
    }

    inline double duration() const
    {
        return duration_.count();
    }

private:

    std::chrono::steady_clock::time_point start_;
    std::chrono::duration<double, T> duration_;
};

typedef Timer<std::milli> MillisecondTimer;
typedef Timer<std::ratio<1, 1>> SecondTimer;

class TimeTracing
{
public:

    typedef std::list<std::string> TraceNames;

    typedef std::shared_ptr<TimeTracing> Ptr;

    TimeTracing(const std::string file_name, const std::string file_path,
                const TraceNames &trace_names, const TraceNames &log_names) :
        file_name_(file_path), trace_names_(trace_names), log_names_(log_names)
    {
        size_t found = file_name_.find_last_of("/\\");
        if(found + 1 != file_name_.size())
            file_name_ += file_name_.substr(found, 1);
        file_name_ += file_name + ".csv";

        ofs_.open(file_name_.c_str());
        if(!ofs_.is_open())
            throw std::runtime_error("Could not open tracefile: " + file_name_);

        init();

        traceHeader();
    }

    ~TimeTracing()
    {
        ofs_.flush();
        ofs_.close();
    }

    inline void startTimer(const std::string &name)
    {
        auto t = timers_.find(name);
        if(t == timers_.end())
        {
            throw std::runtime_error("startTimer: Timer(" + name + ") not registered");
        }
        t->second.start();
    }

    inline void stopTimer(const std::string &name)
    {
        auto t = timers_.find(name);
        if(t == timers_.end())
        {
            throw std::runtime_error("stopTimer: Timer(" + name + ") not registered");
        }
        t->second.stop();
    }

    inline double getTimer(const std::string &name)
    {
        auto t = timers_.find(name);
        if(t == timers_.end())
        {
            throw std::runtime_error("getTimer: Timer(" + name + ") not registered");
        }
        return t->second.duration();
    }

    inline void log(const std::string &name, const double value)
    {
        auto log = logs_.find(name);
        if(log == logs_.end())
        {
            throw std::runtime_error("log: log(" + name + ") not registered");
        }
        log->second = value;
    }

    inline void writeToFile()
    {
        bool first_value = true;
        ofs_.precision(15);
        ofs_.setf(std::ios::fixed, std::ios::floatfield);

        for(auto it = trace_names_.begin(); it != trace_names_.end(); ++it)
        {
            if(first_value)
            {
                ofs_ << timers_[*it].duration();
                first_value = false;
            }
            else
                ofs_ << "," << timers_[*it].duration();
        }
        for(auto it = log_names_.begin(); it != log_names_.end(); ++it)
        {
            if(first_value)
            {
                ofs_ << logs_[*it];
                first_value = false;
            }
            else
                ofs_ << "," << logs_[*it];
        }
        ofs_ << "\n";

        reset();
    }

private:

    void init()
    {
        for(const std::string &trace : trace_names_)
        {
            timers_.emplace(trace, SecondTimer());
        }

        for(const std::string &log : log_names_)
        {
            logs_.emplace(log, -1.0);
        }
    }

    void reset()
    {
        for(auto it = timers_.begin(); it != timers_.end(); ++it)
           it->second.reset();

        for(auto it = logs_.begin(); it != logs_.end(); ++it)
            it->second = -1;

    }

    void traceHeader()
    {
        bool first_value = true;
        for(auto it = trace_names_.begin(); it != trace_names_.end(); ++it)
        {
            if(first_value)
            {
                ofs_ << *it;
                first_value = false;
            }
            else
                ofs_ << "," << *it;
        }
        for(auto it = log_names_.begin(); it != log_names_.end(); ++it)
        {
            if(first_value)
            {
                ofs_ << *it;
                first_value = false;
            }
            else
                ofs_ << "," << *it;
        }
        ofs_ << "\n";
    }

private:

    std::unordered_map<std::string, SecondTimer> timers_;
    std::unordered_map<std::string, double> logs_;
    std::string file_name_;
    TraceNames trace_names_;
    TraceNames log_names_;

    std::ofstream ofs_;
};

//! TimeTrace for ssvo
extern TimeTracing::Ptr sysTrace;
extern TimeTracing::Ptr dfltTrace;
extern TimeTracing::Ptr mapTrace;

}

#endif //_SSVO_TIME_TRACING_HPP_
