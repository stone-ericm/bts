#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

static time_t offset(void) {
    const char *value = getenv("AUDIT_READY_CLOCK_OFFSET");
    return value ? (time_t)strtoll(value, NULL, 10) : 0;
}

static int audit_clock_gettime(clockid_t clock_id, struct timespec *value) {
    int result = clock_gettime(clock_id, value);
    if (!result && clock_id == CLOCK_REALTIME)
        value->tv_sec += offset();
    return result;
}

static int audit_gettimeofday(struct timeval *value, void *zone) {
    int result = gettimeofday(value, zone);
    if (!result && value)
        value->tv_sec += offset();
    return result;
}

static time_t audit_time(time_t *value) {
    time_t result = time(NULL) + offset();
    if (value)
        *value = result;
    return result;
}

#define INTERPOSE(replacement, original) \
    __attribute__((used, section("__DATA,__interpose"))) \
    static const struct { const void *replacement; const void *original; } \
    redirect_##original = { (const void *)replacement, (const void *)original }

INTERPOSE(audit_clock_gettime, clock_gettime);
INTERPOSE(audit_gettimeofday, gettimeofday);
INTERPOSE(audit_time, time);
