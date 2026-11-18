#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

static time_t offset(void) {
    const char *value = getenv("BTS_BATCH_CLOCK_OFFSET");
    return value ? (time_t)strtoll(value, NULL, 10) : 0;
}

static int batch_clock_gettime(clockid_t clock_id, struct timespec *value) {
    int result = clock_gettime(clock_id, value);
    if (!result && clock_id == CLOCK_REALTIME)
        value->tv_sec += offset();
    return result;
}

static int batch_gettimeofday(struct timeval *value, void *zone) {
    int result = gettimeofday(value, zone);
    if (!result && value)
        value->tv_sec += offset();
    return result;
}

static time_t batch_time(time_t *value) {
    time_t result = time(NULL) + offset();
    if (value)
        *value = result;
    return result;
}

#define INTERPOSE(replacement, original) \
    __attribute__((used, section("__DATA,__interpose"))) \
    static const struct { const void *replacement; const void *original; } \
    redirect_##original = { (const void *)replacement, (const void *)original }

INTERPOSE(batch_clock_gettime, clock_gettime);
INTERPOSE(batch_gettimeofday, gettimeofday);
INTERPOSE(batch_time, time);
