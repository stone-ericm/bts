#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

static time_t offset(void) {
    const char *value = getenv("WR_C3_WALL_OFFSET");
    return value ? (time_t)strtoll(value, NULL, 10) : 0;
}

static time_t review_time(time_t *out) {
    time_t value = time(NULL) + offset();
    if (out) *out = value;
    return value;
}

static int review_gettimeofday(struct timeval *tv, void *zone) {
    int result = gettimeofday(tv, zone);
    if (!result && tv) tv->tv_sec += offset();
    return result;
}

static int review_clock_gettime(clockid_t clock_id, struct timespec *tp) {
    int result = clock_gettime(clock_id, tp);
    if (!result && clock_id == CLOCK_REALTIME) tp->tv_sec += offset();
    return result;
}

static uint64_t review_clock_gettime_nsec_np(clockid_t clock_id) {
    uint64_t result = clock_gettime_nsec_np(clock_id);
    if (clock_id == CLOCK_REALTIME) result += offset() * 1000000000LL;
    return result;
}

#define INTERPOSE(new, old) \
    __attribute__((used)) static const struct { const void *replacement; const void *original; } \
        binding_##old __attribute__((section("__DATA,__interpose"))) = { (const void *)new, (const void *)old }

INTERPOSE(review_time, time);
INTERPOSE(review_gettimeofday, gettimeofday);
INTERPOSE(review_clock_gettime, clock_gettime);
INTERPOSE(review_clock_gettime_nsec_np, clock_gettime_nsec_np);
