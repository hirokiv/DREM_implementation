CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes`  -I/usr/local/include/eigen3
TARGET = adjugate_module`python3-config --extension-suffix`
SOURCES = compute_adjugate.cpp
LDFLAGS = -undefined dynamic_lookup

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
