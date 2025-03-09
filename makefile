default:
	nvcc src/$(TARGET)/main.cu -o main.exe -lcublas
	nvprof --trace gpu main.exe
