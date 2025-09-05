CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
TARGET = argmax_benchmark
SOURCE = argmax.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) -lm

clean:
	rm -f $(TARGET)

# Quick test run
test: $(TARGET)
	./$(TARGET) 1000 100

# Performance test
perf: $(TARGET)
	./$(TARGET) 10000 1000