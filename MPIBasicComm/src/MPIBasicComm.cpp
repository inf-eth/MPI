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

	int Size = 5*1024*1024;
	if (world_rank == 0)
	{
		int* Data = new int[Size];
		for (int i=0; i<Size; i++)
			Data[i] = i;
		MPI_Send(&Data[Size/2], Size/2, MPI_INT, 1, 0, MPI_COMM_WORLD);
		long long mySum = 0;
		for (int i=0; i<Size/2; i++)
			mySum += Data[i];
		if (Size<20)
		{
			for (int i=0; i<Size/2; i++)
				cout << Data[i] << " ";
			cout << endl;
		}
		long long theirSum;
		MPI_Recv(&theirSum, 1, MPI_LONG_LONG, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		cout << "Total Sum: " << mySum+theirSum << endl;
		delete[] Data;
	}
	else if (world_rank == 1)
	{
		int* Data = new int[Size/2];
		MPI_Recv(Data, Size/2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		long long mySum = 0;
		for (int i=0; i<Size/2; i++)
			mySum += Data[i];
		if (Size<20)
		{
			for (int i=0; i<Size/2; i++)
				cout << Data[i] << " ";
			cout << endl;
		}
		MPI_Send(&mySum, 1, MPI_LONG_LONG, 0, 1, MPI_COMM_WORLD);
		delete[] Data;
	}
	else
	{
		cout << "No work for process " << world_rank << " of " << world_size << endl;
	}

	MPI_Finalize();
	return 0;
}