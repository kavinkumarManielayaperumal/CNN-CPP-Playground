# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -g

# Directories
SRC_DIR := .
BUILD_DIR := build
TARGET := $(BUILD_DIR)/main

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Default target
all: $(TARGET)

# Build the target
$(TARGET): $(OBJS)
	@echo "Linking objects into $(TARGET)"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compile each source file into an object file
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $< into $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning up..."
	@if [ -d "$(BUILD_DIR)" ]; then rm -rf "$(BUILD_DIR)"; else echo "No build directory to clean."; fi
