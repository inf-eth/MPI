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

	const int countPerProcess = 2;
	int Data[countPerProcess];
	int subData[countPerProcess];
	// Process 0: 0, 1
	// Process 1: 2, 3
	// Process 2: 4, 5
	// Process 3: 6, 7
	for (int i=0; i<countPerProcess; i++)
		subData[i] = countPerProcess*world_rank+i;


	MPI_Reduce(subData, Data, countPerProcess, MPI_INT,\
		MPI_SUM, 0, MPI_COMM_WORLD);

	for (int i=0; i<countPerProcess; i++)
	{
		if (i==0)
			cout << "Process " << world_rank << " data: ";
		cout << subData[i] << " ";
		if (i==countPerProcess-1)
			cout << endl;
	}

	if (world_rank==0)
	{
		cout << "root Data: ";
		for (int i=0; i<countPerProcess; i++)
			cout << Data[i] << " ";
		cout << endl;
	}

	MPI_Finalize();
	return 0;
}
