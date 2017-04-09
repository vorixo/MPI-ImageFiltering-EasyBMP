// Alvaro Jover Alvarez
// Jordi Amoros Moreno

// FILTRO MEDIANO + GAUSSIANO

#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <sys/time.h>
#include <omp.h>
#include "EasyBMP.h"
#include "mpi.h"

using namespace std;

// Constantes para filtro gaussiano
const double PI = 3.141592654;

// Numero maximo de hilos para la cpu actual
const int MAX_NUM_THREADS = omp_get_max_threads();

// Tamanyo de imagen
int WIDTH;
int HEIGHT;

// Struct para comparar
struct Compare {
    int valor;
    int indice;
};

// String es numero
bool is_digits(const std::string &str)
{
    return str.find_first_not_of("0123456789") == string::npos;
}

// Aplicar Gaussian Filter unidireccional
float Gaussian1D(int x, int sigma) {
    return exp(-x * x / (2 * sigma * sigma)) / sigma / sqrt(2 * PI);
}

int main(int argc, char **argv) {
    // Variable imagen

    MPI_Init(&argc, &argv);
    MPI_Status status;

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    BMP Image;
    BMP Mediana;
    BMP gausshori,gaussverti;
    BMP Media;

    if(argc >= 2 && argc <= 5){
        // argv[1] nombre de la imagen
        // argv[2] tamaño de la malla
        // argv[3] fuerza del mediano
        // argv[4] sigma (gaussiano)
        // argv[5] numero de threads

        // Input para imagen
        const char *nombreimagen = argv[1];
        // cout << nombreimagen << endl;

        // Leemos la imagen
        if (Image.ReadFromFile(nombreimagen)) {

            // Variables no relevantes
            string toInt = "";
            bool fallo = false;

            // Datos de la imagen
            WIDTH = Image.TellWidth();
            HEIGHT = Image.TellHeight();
            Mediana.SetSize(WIDTH, HEIGHT);
            Mediana.SetBitDepth(Image.TellBitDepth()); // cambiar bitdepth al de input
            gausshori.SetSize(WIDTH, HEIGHT);
            gausshori.SetBitDepth(Image.TellBitDepth());
            gaussverti.SetSize(WIDTH, HEIGHT);
            gaussverti.SetBitDepth(Image.TellBitDepth());
            Media.SetSize(WIDTH, HEIGHT);
            Media.SetBitDepth(Image.TellBitDepth());

            // Variables de dependencia del algoritmo
            int dim = 0;

            // -----------------------------------------------------
            //Pasar la entrada a int
            if(argc > 2) {
                toInt = argv[2];
                if (is_digits(toInt)) {
                    istringstream ss(toInt);
                    ss >> dim;

                    if (dim % 2 == 0) {
                        fallo = true;
                    }
                }
                // Manejo de errores (cortamos la ejecucion)
                if (fallo) {
                    cout << "Error: Introduce un numero impar." << endl;
                    getchar();
                    return 0;
                }
            } else {
                dim = 3;
            }
            // cout << dim << endl;

            // Numero de valores que contendra la malla
            int dimord = dim*dim;
            volatile int halfdimord = dimord / 2;
            int half = dim/2;

            // -----------------------------------------------------
            int fuerza = 0;

            // Pasar la entrada a int
            if (argc > 3) {
                toInt = argv[3];
                if (is_digits(toInt)) {
                    istringstream ss(toInt);
                    ss >> fuerza;
                }

                // Manejo de errores (cortamos la ejecucion)
                if (fuerza > 100 || fuerza < 1) {
                    cout << "Error: Valor " << fuerza << " fuera de rango para fuerza." << endl;
                    getchar();
                    return 0;
                }
            } else {
                fuerza = 1;
            }
            // cout << fuerza << endl;

            // ---------------------------------------------

            // SIGMA PARA GAUSSIANO
            int sigma = 1; // CAMBIAR PARA DIFERENTES VALORES DE SIGMA

            // Pasar la entrada a int
            if (argc > 4) {
               toInt = argv[4];
               if (is_digits(toInt)) {
                   istringstream ss(toInt);
                   ss >> sigma;
               }

               // Manejo de errores (cortamos la ejecucion)
               if (sigma > 50 || sigma < 1) {
                   cout << "Error: Valor " << sigma << " fuera de rango para sigma." << endl;
                   getchar();
                   return 0;
               }
           }else {
               sigma = 1;
           }
           // cout << sigma << endl;
           // ---------------------------------------------

            int hilos;

            //cout << "Numero de threads [1 - " << MAX_NUM_THREADS << "] : ";

            if (argc > 5) {
               toInt = argv[5];
               if (is_digits(toInt)) {
                   istringstream ss(toInt);
                   ss >> hilos;
               }

               if (hilos > MAX_NUM_THREADS) {
                   cout << "Error: Valor " << MAX_NUM_THREADS << " fuera de rango." << endl;
                   getchar();
                   return 0;
               }
           }else {
               hilos = 1;
           }
           // cout << hilos << endl;

            // omp_set_num_threads(hilos);

          if(world_rank == 0) {

            cout << "Nombre de la imagen: " << argv[1] << endl;
            cout << "Dimensiones imagen: " << WIDTH << " x " << HEIGHT << endl;
            cout << "Dimension de la malla: " << dim << endl;
            cout << "Fuerza del filtro [1-100] (Median Filter): " << fuerza << endl;
            cout << "Valor de sigma [1-50] (Gaussian Filter): " << sigma << endl;
            // -----------------------------------------------------
            // --  EMPEZAMOS OPERACIONES DE CPU SIN PARALELIZAR   --
            // -----------------------------------------------------

            // Variables t (tiempo monotonico para evitar doblar los ticks del procesador)
            struct timespec start, finish;
            double elapsed;

            clock_gettime(CLOCK_MONOTONIC, &start);

            /////////////////////////////
            //    FILTRO MEDIANA        //
            //////////////////////////////

            RGBApixel black, median;
  			    black.Red = 0;
  			    black.Green = 0;
  			    black.Blue = 0;

            int r, g, b, x_i, y_i, row;
            int offset = HEIGHT/(world_size-1);
            int message_array[3] = {HEIGHT,WIDTH,offset};

            MPI_Bcast(message_array, 3 , MPI_INT , 0 , MPI_COMM_WORLD);

            for (int i = 1; i < world_size; i++) {
	            row = (i-1)*offset;
	            MPI_Send(&row,1,MPI_INT,i,0,MPI_COMM_WORLD);
		        }

		        int missing = HEIGHT%(world_size-1);
		        int coord[5];

		        for(int i = 0; i < (WIDTH*HEIGHT) - (missing*WIDTH); i++) {
		            MPI_Recv(coord,5,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
		            x_i = coord[0];
		            y_i = coord[1];
		            r = coord[2];
		            g = coord[3];
		            b = coord[4];
		            if(x_i >= 0 && x_i < WIDTH && y_i >= 0 && y_i < HEIGHT) {
		                RGBApixel temp;
		                temp.Red = coord[2];
		                temp.Green = coord[3];
		                temp.Blue = coord[4];
		                Mediana.SetPixel(x_i,y_i,temp);
		            } else {
		                Mediana.SetPixel(x_i,y_i,black);
		            }
		        }

		        // Median filter aplicado
	            // Descomentar para ver
	            Mediana.WriteToFile("Mediana.bmp");
	            cout << "Mediana completa" << endl;

	            clock_gettime(CLOCK_MONOTONIC, &finish);

	            elapsed = (finish.tv_sec - start.tv_sec);
	            elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;


	            cout << "Tiempo de ejecucion para " << dim <<
	                " de tamano de malla y " << fuerza << " de fuerza en la mediana a la resolucion dada: "
	                << elapsed << " segundos" << endl;

	            cout << endl << "Pulsa intro para finalizar...";

	        }

	        else{

	        	int message_array[3];
	        	int row;
	        	// Mediana
	            unsigned char neighboursred[dimord];
	            unsigned char neighboursblue[dimord];
	            unsigned char neighboursgreen[dimord];
	            unsigned char ordenacion;

	            // Gaussiano
	            float gaussian_filter[dimord];
	            float acumuladorgauss = 0;
	            unsigned count;
	            struct Compare maximored, maximoblue, maximogreen;

	            int tmp;
	            int w, s, y, x, k, l;

				MPI_Bcast(message_array,3,MPI_INT,0,MPI_COMM_WORLD);

				int offset = message_array[2];

				MPI_Recv(&row,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);

	            // For indicativo de la fuerza del algoritmo
	            for (int i = 0; i < fuerza; i++) {

	                // Recorremos la imagen
	                /*
	                #pragma omp parallel for \
	                shared(Image, WIDTH, HEIGHT, dim, half) \
	                private(y,x,w,s,k,l,tmp,count,neighboursred, neighboursblue, \
	                neighboursgreen, maximored, maximoblue,maximogreen) \
	                schedule(static)
	                */
	                for (y = 0; y < WIDTH; y ++) {
	                    for (x = row; x < row + offset; x++) {

	                        // Guardamos los valores de la imagen en una malla de tamanyo dim*dim
	                        count = 0;

	                        for (w = 0; w < dim && (y + w - half) < WIDTH; w++) {
	                            for (s = 0; s < dim && (x + s - half) < HEIGHT; s++) {

	                                if (y + w - half >= 0 && x + s - half >= 0) {
	                                    neighboursred[count] = Image(y + w - half, x + s - half)->Red;
	                                    neighboursblue[count] = Image(y + w - half, x + s - half)->Blue;
	                                    neighboursgreen[count] = Image(y + w - half, x + s - half)->Green;
	                                    count++;
	                                }
	                            }
	                        }

	                        // Ordenamos para extraer la mediana
	                        for (k = count-1; k >= 0; --k) {
	                            maximored.valor = neighboursred[k];
	                            maximored.indice = k;

	                            maximoblue.valor = neighboursblue[k];
	                            maximoblue.indice = k;

	                            maximogreen.valor = neighboursgreen[k];
	                            maximogreen.indice = k;

	                            for (l = k-1; l >= 0; --l) {
	                                if (neighboursred[l] > maximored.valor) {
	                                    maximored.valor = neighboursred[l];
	                                    maximored.indice = l;
	                                }
	                                if (neighboursblue[l] > maximoblue.valor) {
	                                    maximoblue.valor = neighboursblue[l];
	                                    maximoblue.indice = l;
	                                }
	                                if (neighboursgreen[l] > maximogreen.valor) {
	                                    maximogreen.valor = neighboursgreen[l];
	                                    maximogreen.indice = l;
	                                }
	                            }
	                            tmp = neighboursred[k];
	                            neighboursred[k] = maximored.valor;
	                            neighboursred[maximored.indice] = tmp;

	                            tmp = neighboursblue[k];
	                            neighboursblue[k] = maximoblue.valor;
	                            neighboursblue[maximoblue.indice] = tmp;

	                            tmp = neighboursgreen[k];
	                            neighboursgreen[k] = maximogreen.valor;
	                            neighboursgreen[maximogreen.indice] = tmp;
	                        }

	                        int image_data[5] = {y, x, neighboursred[count/2], neighboursgreen[count/2], neighboursblue[count/2]};
	                        MPI_Send(image_data,5,MPI_INT,0,0,MPI_COMM_WORLD);
	                    }
	                }
	            }
	        }  // Fin del else de las esclavas
        }

        else {
            cerr << endl << "CONSEJO: Asegurate que el archivo de entrada tiene el formato correcto (BMP),"
                << " esta en el directorio correcto (../Sample) y esta escrito correctamente." << endl;
        }

        getchar();
    } else {
        cout << "Error. Uso: mpirun -np (numero nodos) --hostfile (hostfiles.txt) (nombre ejecutable) (nombre imagen)" << endl;
        cout << "Parametros opcionales (por orden): (Tamaño malla [default=3]) (Fuerza Mediano [default=1]) (Sigma Gaussiano [default=1]) (Numero threads (OpenMP) [default=1])." << endl;
    }

    MPI_Finalize();

    return 0;

}
