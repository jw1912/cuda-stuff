default:
	nvcc --run src/$(TARGET).cu -o main.exe -Xptxas="-v"
