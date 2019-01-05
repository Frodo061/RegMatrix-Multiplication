#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <mm_malloc.h>
#include <vector>
#include <string.h>
#include <papi.h>

using namespace std;

#define TIME_RESOLUTION 1000000	// time measuring resolution (us)

vector<long long unsigned> *time_measurement = new vector<long long unsigned>();
long long unsigned initial_time;
double clearcache [30000000];
timeval t;
long long **values;
int *events;
int numEvents;
int eventSet = PAPI_NULL;

void clearCache (void) {
	for (unsigned i = 0; i < 30000000; ++i)
		clearcache[i] = i;
}

void utils_start_timer(void) {
	gettimeofday(&t, NULL);
	initial_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;
}

void utils_stop_timer(void) {
	gettimeofday(&t, NULL);
	long long unsigned final_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;
	time_measurement->push_back(final_time - initial_time);
}

int utils_init_matrices(float **a, float **b, float **c, int N) {
    int i;
    const int total_elements = N * N;
    *a = (float*) _mm_malloc(N*N*sizeof(float), 32);
    *b = (float*) _mm_malloc(N*N*sizeof(float), 32);
    *c = (float*) _mm_malloc(N*N*sizeof(float), 32);
    for (i = 0; i < total_elements; i++)
    {
        (*a)[i] = ((float) rand()) / ((float) RAND_MAX);
        (*b)[i] = 1;
        (*c)[i] = 0;
    }
    return 1;
}

void utils_setup_papi(int repetitions, const char *type) {
    if (!strcmp(type, "time"))
    {
        return;
    }
    else if (!strcmp(type, "l1mr"))
    {
        numEvents = 2;
        events = (int *)malloc(numEvents * sizeof(int));
        events[0] = PAPI_L1_DCM;
        events[1] = PAPI_LD_INS;
    }
    else if (!strcmp(type, "l2mr"))
    {
        numEvents = 2;
        events = (int *)malloc(numEvents * sizeof(int));
        events[0] = PAPI_L3_DCR;
        events[1] = PAPI_L1_DCM;
    }
    else if (!strcmp(type, "l3mr"))
    {
        numEvents = 2;
        events = (int *)malloc(numEvents * sizeof(int));
        events[0] = PAPI_L3_TCM;
        events[1] = PAPI_L3_TCA;
    }
    else if (!strcmp(type, "flops"))
    {
        numEvents = 1;
        events = (int *)malloc(numEvents * sizeof(int));
        events[0] = PAPI_FP_OPS;
    }
    else if (!strcmp(type, "vflops"))
    {
        numEvents = 1;
        events = (int *)malloc(numEvents * sizeof(int));
        events[0] = PAPI_VEC_SP;
    }
    values = (long long **)malloc(sizeof(long long) * repetitions);
    for (int i = 0; i < repetitions; i++)
    {
        values[i] = (long long *)malloc(sizeof(long long) * numEvents);
    }
    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&eventSet);
    PAPI_add_events(eventSet, events, numEvents); /* Start the counters */
}

void utils_results(const char *type) {
    int repetitions = time_measurement->size();
    for (int i = 0; i < repetitions; i++)
    {
        if (!strcmp(type, "time"))
        {
            double tm = time_measurement->at(i) / (double)1000;
            cout << "Execution Time;" << tm << endl;
        }
        else if (!strcmp(type, "l1mr"))
        {
            cout << values[i][0] <<";"<<values[i][1] << endl;
        }
        else if (!strcmp(type, "l2mr"))
        {
            cout << values[i][0] <<";"<<values[i][1] << endl;
        }
        else if (!strcmp(type, "l3mr"))
        {
            cout << values[i][0] <<";"<<values[i][1] << endl;
        }
        else if (!strcmp(type, "flops"))
        {
            cout << values[i][0] << endl;
        }
        else if (!strcmp(type, "vflops"))
        {
            cout << values[i][0] << endl;
        }
    }
}

int utils_clean_matrices(float **a, float **b, float **c){
    if(*a!=NULL)
        free(*a);
    if(*b!=NULL)
        free(*b);
    if(*c!=NULL)
        free(*c);
    return 0;
}

void utils_start_papi(const char *type) {
    if (strcmp(type, "time"))
        PAPI_start(eventSet);        
}

void utils_stop_papi(int rep, const char *type) {
    if (strcmp(type, "time"))
        PAPI_stop(eventSet, values[rep]);
}