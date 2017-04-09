
CC = mpic++

CFLAGS = -Wall -O3 -pipe -fomit-frame-pointer -funroll-all-loops -s

EasyBMPTest: EasyBMP.o Practica1.o
	$(CC) $(CFLAGS) EasyBMP.o Practica1.o -o Filtros -fopenmp

EasyBMP.o: ../EasyBMP.cpp ../EasyBMP*.h
	cp ../EasyBMP*.h .
	cp ../EasyBMP.cpp .
	g++ $(CFLAGS) -c EasyBMP.cpp

Practica1.o: Practica1.cpp
	$(CC) -c Practica1.cpp -fopenmp

clean: 
	rm EasyBMP*.h
	rm EasyBMP.cpp
	rm -f *.o
	rm Filtros
	rm Media.bmp

