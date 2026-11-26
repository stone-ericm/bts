#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define INTERPOSE(replacement, original)                                      \
    __attribute__((used, section("__DATA,__interpose")))                       \
    static const struct { const void *replacement; const void *original; }     \
    interpose_##original = { (const void *)(uintptr_t)&replacement,           \
                             (const void *)(uintptr_t)&original }

static long wall_offset(void) {
    const char *value = getenv("CS4CR2_WALL_OFFSET");
    return value ? strtol(value, NULL, 10) : 0;
}

static int review_clock_gettime(clockid_t clock, struct timespec *value) {
    int result = clock_gettime(clock, value);
    if (result == 0 && clock == CLOCK_REALTIME) {
        value->tv_sec += wall_offset();
    }
    return result;
}

static int review_gettimeofday(struct timeval *restrict value, void *restrict zone) {
    int result = gettimeofday(value, zone);
    if (result == 0 && value) {
        value->tv_sec += wall_offset();
    }
    return result;
}

INTERPOSE(review_clock_gettime, clock_gettime);
INTERPOSE(review_gettimeofday, gettimeofday);
