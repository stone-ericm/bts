/* Set a process wall clock while leaving interval and timeout clocks alone. */
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

static time_t offset;

__attribute__((constructor)) static void initialize_clock(void) {
    const char *epoch = getenv("CS6B2_WALL_EPOCH");
    if (epoch != NULL)
        offset = (time_t)strtoll(epoch, NULL, 10) - time(NULL);
}

static int review_clock_gettime(clockid_t clock, struct timespec *value) {
    int result = clock_gettime(clock, value);
    if (result == 0 && clock == CLOCK_REALTIME)
        value->tv_sec += offset;
    return result;
}

static int review_gettimeofday(struct timeval *value, void *zone) {
    int result = gettimeofday(value, zone);
    if (result == 0 && value != NULL)
        value->tv_sec += offset;
    return result;
}

static time_t review_time(time_t *value) {
    time_t result = time(NULL) + offset;
    if (value != NULL)
        *value = result;
    return result;
}

#define INTERPOSE(replacement, target) \
    __attribute__((used)) static const struct { const void *new_fn; const void *old_fn; } \
    interpose_##target __attribute__((section("__DATA,__interpose"))) = { \
        (const void *)&replacement, (const void *)&target \
    }

INTERPOSE(review_clock_gettime, clock_gettime);
INTERPOSE(review_gettimeofday, gettimeofday);
INTERPOSE(review_time, time);
