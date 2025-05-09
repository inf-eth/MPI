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

	//cout << "Hello from processor " << processor_name << " with rank " << world_rank << " of " << world_size << endl;

	const int Data_Size = 8;
	const int countPerProcess = 2;
	int subData[countPerProcess];

	if (world_rank==0)
	{
		int Data[Data_Size];
		for (int i=0; i<Data_Size; i++)
			Data[i] = i;
		MPI_Scatter(Data, countPerProcess, MPI_INT,\
					subData, countPerProcess, MPI_INT, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Scatter(NULL, countPerProcess, MPI_INT,\
					subData, countPerProcess, MPI_INT, 0, MPI_COMM_WORLD);
	}
	cout << processor_name << ", rank " << world_rank << " of "\
	<< world_size << " has data: "\
	<< subData[0] << ", " << subData[1] << ", " << subData[2] << endl;

	MPI_Finalize();
	return 0;
}
