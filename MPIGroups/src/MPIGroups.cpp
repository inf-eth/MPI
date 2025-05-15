#include <mpi.h>
#include <iostream>
using std::cout;
using std::endl;

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	int world_size, local_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank, local_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	MPI_Group world_group, local_group;
	MPI_Comm local_comm;

	// split global ranks into two groups, lower half ranks and upper half ranks
	// For world_size of 8
	// Ranks1[] = {0,1,2,3}; Ranks2[] = {4,5,6,7}
	const int nRanks1 = world_size/2;
	const int nRanks2 = world_size-nRanks1;
	int* Ranks1 = new int[nRanks1];
	int* Ranks2 = new int[nRanks2];
	for (int i=0; i<nRanks1; i++)
		Ranks1[i] = i;
	for (int i=0; i<nRanks2; i++)
		Ranks2[i] = i+nRanks1;

	// extract the original group handle
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	if (world_rank < world_size/2)
		MPI_Group_incl(world_group, nRanks1, Ranks1, &local_group);
	else
		MPI_Group_incl(world_group, nRanks2, Ranks2, &local_group);

	MPI_Comm_create(MPI_COMM_WORLD, local_group, &local_comm);
	MPI_Group_rank(local_group, &local_rank);
	cout << "Hello from processor " << processor_name << " with rank " << world_rank << " of " << world_size << ", local_rank: " << local_rank << endl;

	// local reduction of ranks
	int local_sum;
	MPI_Reduce(&world_rank, &local_sum, 1, MPI_INT, MPI_SUM, 0, local_comm);
	if (local_rank==0)
		cout << "rank: " << world_rank << ", world rank group sum: " << local_sum << endl;

	delete[] Ranks1;
	delete[] Ranks2;

	MPI_Finalize();
	return 0;
}
