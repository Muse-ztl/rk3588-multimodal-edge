// 板端快照：RSS、进程 CPU 时间、温度、NPU/CPU 频率。
// 读取失败返回 0，bench 不因 sysfs 缺文件而失败。
#pragma once

#include <cstdio>
#include <cstring>

#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace mm {

struct SysSnapshot {
  double rss_mb = 0.0;
  double hwm_mb = 0.0;
  double cpu_ms = 0.0;
  double temp_c = 0.0;
  double npu_mhz = 0.0;
  double cpu_mhz = 0.0;
};

inline double read_first_number(const char* path) {
  FILE* fp = std::fopen(path, "r");
  if (fp == nullptr) return 0.0;
  double v = 0.0;
  if (std::fscanf(fp, "%lf", &v) != 1) v = 0.0;
  std::fclose(fp);
  return v;
}

inline void read_mem(double* rss_mb, double* hwm_mb) {
  *rss_mb = 0.0;
  *hwm_mb = 0.0;
  FILE* fp = std::fopen("/proc/self/status", "r");
  if (fp == nullptr) return;
  char line[256];
  while (std::fgets(line, sizeof(line), fp) != nullptr) {
    double kb = 0.0;
    if (std::sscanf(line, "VmRSS: %lf", &kb) == 1) *rss_mb = kb / 1024.0;
    if (std::sscanf(line, "VmHWM: %lf", &kb) == 1) *hwm_mb = kb / 1024.0;
  }
  std::fclose(fp);
}

inline double process_cpu_ms() {
#if defined(__linux__)
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
  return (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000.0 +
         (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1000.0;
#else
  return 0.0;
#endif
}

inline SysSnapshot sys_snapshot() {
  SysSnapshot s;
  read_mem(&s.rss_mb, &s.hwm_mb);
  s.cpu_ms = process_cpu_ms();
  s.temp_c = read_first_number("/sys/class/thermal/thermal_zone0/temp") / 1000.0;
  s.npu_mhz = read_first_number("/sys/class/devfreq/fdab0000.npu/cur_freq") / 1e6;
  s.cpu_mhz = read_first_number("/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq") / 1000.0;
  return s;
}

}  // namespace mm
