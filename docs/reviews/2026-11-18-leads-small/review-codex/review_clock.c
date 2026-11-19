/* Process-local realtime shift for macOS. Monotonic and CPU clocks retain
 * their native values. Absolute times from the filesystem remain real. */
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

static pthread_once_t initialized = PTHREAD_ONCE_INIT;
static time_t offset;

static void initialize_offset(void) {
    const char *target = getenv("BTS_REVIEW_EPOCH");
    if (target != NULL) {
        struct timeval actual;
        if (gettimeofday(&actual, NULL) == 0) {
            offset = (time_t)strtoll(target, NULL, 10) - actual.tv_sec;
        }
    }
}

static time_t review_offset(void) {
    pthread_once(&initialized, initialize_offset);
    return offset;
}

static int review_clock_gettime(clockid_t clock, struct timespec *value) {
    int status = clock_gettime(clock, value);
    if (status == 0 && clock == CLOCK_REALTIME) {
        value->tv_sec += review_offset();
    }
    return status;
}

static uint64_t review_clock_gettime_nsec_np(clockid_t clock) {
    uint64_t value = clock_gettime_nsec_np(clock);
    if (value != 0 && clock == CLOCK_REALTIME) {
        value += review_offset() * INT64_C(1000000000);
    }
    return value;
}

static int review_gettimeofday(struct timeval *value, void *zone) {
    int status = gettimeofday(value, zone);
    if (status == 0 && value != NULL) {
        value->tv_sec += review_offset();
    }
    return status;
}

static time_t review_time(time_t *output) {
    time_t value = time(NULL);
    if (value != (time_t)-1) {
        value += review_offset();
    }
    if (output != NULL) {
        *output = value;
    }
    return value;
}

#define INTERPOSE(new, old) \
    __attribute__((used, section("__DATA,__interpose"))) \
    static const struct { const void *new_fn; const void *old_fn; } pair_##old = { \
        (const void *)(uintptr_t)&new, (const void *)(uintptr_t)&old \
    }

INTERPOSE(review_clock_gettime, clock_gettime);
INTERPOSE(review_clock_gettime_nsec_np, clock_gettime_nsec_np);
INTERPOSE(review_gettimeofday, gettimeofday);
INTERPOSE(review_time, time);
