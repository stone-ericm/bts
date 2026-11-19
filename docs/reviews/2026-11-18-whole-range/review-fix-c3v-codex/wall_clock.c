/* Review-only wall-clock shift for the Python process on macOS. */
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

static long long offset_seconds(void) {
    const char *value = getenv("C3V_WALL_CLOCK_OFFSET");
    return value ? strtoll(value, NULL, 10) : 0;
}

static int shifted_clock_gettime(clockid_t clock, struct timespec *value) {
    int status = clock_gettime(clock, value);
    if (status == 0 && clock == CLOCK_REALTIME) {
        value->tv_sec += offset_seconds();
    }
    return status;
}

static int shifted_gettimeofday(struct timeval *value, void *zone) {
    int status = gettimeofday(value, zone);
    if (status == 0 && value) {
        value->tv_sec += offset_seconds();
    }
    return status;
}

static time_t shifted_time(time_t *target) {
    time_t value = time(NULL);
    if (value != (time_t)-1) {
        value += offset_seconds();
    }
    if (target) {
        *target = value;
    }
    return value;
}

__attribute__((used, section("__DATA,__interpose")))
static const struct { const void *replacement, *original; } functions[] = {
    {(const void *)shifted_clock_gettime, (const void *)clock_gettime},
    {(const void *)shifted_gettimeofday, (const void *)gettimeofday},
    {(const void *)shifted_time, (const void *)time},
};
