/* Native macOS wall-clock interposer for CS6b-2 2027/2028 clock verification.
 * Shifts CLOCK_REALTIME / gettimeofday / time() to CS6B2_TARGET_EPOCH while
 * leaving CLOCK_MONOTONIC, CLOCK_UPTIME_RAW, and setitimer completely untouched.
 */

#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

static time_t g_delta_sec = 0;

__attribute__((constructor))
static void init_clock_offset(void) {
    const char *env = getenv("CS6B2_TARGET_EPOCH");
    if (!env || !*env) {
        g_delta_sec = 0;
        return;
    }
    long long target = strtoll(env, NULL, 10);
    struct timeval tv;
    g_delta_sec = 0;
    if (gettimeofday(&tv, NULL) == 0 && target > 0) {
        g_delta_sec = (time_t)(target - (long long)tv.tv_sec);
    }
}

static int my_gettimeofday(struct timeval *tv, void *tz) {
    int rc = gettimeofday(tv, tz);
    if (rc == 0 && tv != NULL) {
        tv->tv_sec += g_delta_sec;
    }
    return rc;
}

static time_t my_time(time_t *tloc) {
    time_t now = time(NULL) + g_delta_sec;
    if (tloc != NULL) {
        *tloc = now;
    }
    return now;
}

static int my_clock_gettime(clockid_t clk_id, struct timespec *tp) {
    int rc = clock_gettime(clk_id, tp);
    if (rc == 0 && tp != NULL && clk_id == CLOCK_REALTIME) {
        tp->tv_sec += g_delta_sec;
    }
    return rc;
}

static uint64_t my_clock_gettime_nsec_np(clockid_t clk_id) {
    uint64_t ns = clock_gettime_nsec_np(clk_id);
    if (ns != 0 && clk_id == CLOCK_REALTIME) {
        ns = (uint64_t)((int64_t)ns + (int64_t)g_delta_sec * 1000000000LL);
    }
    return ns;
}

typedef struct interpose_s {
    const void *new_func;
    const void *orig_func;
} interpose_t;

__attribute__((used)) static const interpose_t interpose_table[]
__attribute__((section("__DATA,__interpose"))) = {
    { (const void *)my_gettimeofday, (const void *)gettimeofday },
    { (const void *)my_time, (const void *)time },
    { (const void *)my_clock_gettime, (const void *)clock_gettime },
    { (const void *)my_clock_gettime_nsec_np, (const void *)clock_gettime_nsec_np },
};
