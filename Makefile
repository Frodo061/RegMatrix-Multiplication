BIN_NAME = main

# CXX = g++
# LD  = g++

CXX = icpc
LD  = icpc

#-fopenmp/-openmp for GNU/Intel

#CXXFLAGS = -O3 -Wall -Wextra -std=c++11 -Wno-unused-parameter -I/share/apps/papi/5.5.0/include -L/share/apps/papi/5.5.0/lib -lpapi
CXXFLAGS = -O3 -Wall -Wextra -std=c++11 -Wno-unused-parameter -qopt-report=3 -qopenmp  -I/share/apps/papi/5.5.0/include -L/share/apps/papi/5.5.0/lib -lpapi

ifeq ($(VEC),no)
	CXXFLAGS += -no-vec
endif

ifeq ($(DOT_PR_1),yes)
	CXXFLAGS += -DDOT_PR_1
endif

ifeq ($(DOT_PR_1_TR),yes)
	CXXFLAGS += -DDOT_PR_1_TR
endif

ifeq ($(DOT_PR_1_BL),yes)
	CXXFLAGS += -DDOT_PR_1_BL
endif

ifeq ($(DOT_PR_2),yes)
	CXXFLAGS += -DDOT_PR_2
endif

ifeq ($(DOT_PR_2_BL),yes)
	CXXFLAGS += -DDOT_PR_2_BL
endif

ifeq ($(DOT_PR_3),yes)
	CXXFLAGS += -DDOT_PR_3
endif

ifeq ($(DOT_PR_3_TR),yes)
	CXXFLAGS += -DDOT_PR_3_TR
endif

ifeq ($(DOT_PR_3_BL),yes)
	CXXFLAGS += -DDOT_PR_3_BL
endif

SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build
LOG_DIR = logs
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(patsubst src/%.cpp,build/%.o,$(SRC))
DEPS = $(patsubst build/%.o,build/%.d,$(OBJ))
BIN = $(BIN_NAME)

vpath %.cpp $(SRC_DIR)

.DEFAULT_GOAL = all

$(BUILD_DIR)/%.d: %.cpp
	$(CXX) -M $< -o $@ -lm $(CXXFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) -c $< -o $@ -lm $(CXXFLAGS)

$(BIN_DIR)/$(BIN_NAME): $(DEPS) $(OBJ)
	$(CXX) -o $@ $(OBJ) -lm $(CXXFLAGS)

checkdirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

all: checkdirs $(BIN_DIR)/$(BIN_NAME)

run: clean all
	./bin/main

clean:
	rm -f $(BUILD_DIR)/* $(BIN_DIR)/*
