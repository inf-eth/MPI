#include <mpi.h>
#include <iostream>
using std::cout;
using std::endl;

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	cout << "Hello from processor " << processor_name << " with rank " << world_rank << " of " << world_size << endl;

	const int Data_Size = 8;
	int Data[Data_Size];
	const int countPerProcess = 2;
	int subData[countPerProcess];
	// Process 0: 0, 1
	// Process 1: 2, 3
	// Process 2: 4, 5
	// Process 3: 6, 7
	for (int i=0; i<countPerProcess; i++)
		subData[i] = countPerProcess*world_rank+i;


	MPI_Allgather(subData, countPerProcess, MPI_INT,\
		Data, countPerProcess, MPI_INT, MPI_COMM_WORLD);

	cout << "Process " << world_rank << " gathered: "\
		<< Data[0] << ", " << Data[1] << ", " << Data[2] << ", " << Data[3] << ", "\
		<< Data[4] << ", " << Data[5] << ", " << Data[6] << ", " << Data[7] << endl;

	MPI_Finalize();
	return 0;
}
