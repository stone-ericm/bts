/* macOS DYLD_INSERT_LIBRARIES wall-clock interposer for 2027/2028 tests.
 * Shifts CLOCK_REALTIME, gettimeofday, and time() by BTS_CLOCK_OFFSET_SEC
 * while leaving CLOCK_MONOTONIC and kernel timers untouched.
 */
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define DYLD_INTERPOSE(_replacement, _replacee) \
    __attribute__((used)) static struct { \
        const void *replacement; \
        const void *replacee; \
    } _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = { \
        (const void *)(unsigned long)&_replacement, \
        (const void *)(unsigned long)&_replacee \
    };

static long get_offset_sec(void) {
    static int initialized = 0;
    static long cached = 0;
    if (!initialized) {
        const char *s = getenv("BTS_CLOCK_OFFSET_SEC");
        if (s && *s) {
            cached = strtol(s, NULL, 10);
        }
        initialized = 1;
    }
    return cached;
}

static int my_gettimeofday(struct timeval *tv, void *tzp) {
    int rc = gettimeofday(tv, tzp);
    if (rc == 0 && tv != NULL) {
        tv->tv_sec += get_offset_sec();
    }
    return rc;
}

static int my_clock_gettime(clockid_t clk_id, struct timespec *tp) {
    int rc = clock_gettime(clk_id, tp);
    if (rc == 0 && tp != NULL && clk_id == CLOCK_REALTIME) {
        tp->tv_sec += get_offset_sec();
    }
    return rc;
}

static time_t my_time(time_t *tloc) {
    time_t t = time(NULL);
    if (t != (time_t)-1) {
        t += (time_t)get_offset_sec();
    }
    if (tloc != NULL) {
        *tloc = t;
    }
    return t;
}

DYLD_INTERPOSE(my_gettimeofday, gettimeofday)
DYLD_INTERPOSE(my_clock_gettime, clock_gettime)
DYLD_INTERPOSE(my_time, time)
