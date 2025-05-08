#define TYPE float
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <chrono>
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

struct msClock
{
	typedef std::chrono::high_resolution_clock clock;
	std::chrono::time_point<clock> t1, t2;
	void Start() { t1 = high_resolution_clock::now(); }
	void Stop() { t2 = high_resolution_clock::now(); }
	double ElapsedTime()
	{
		duration<double, std::milli> ms_doubleC = t2-t1;
		return ms_doubleC.count();
	}
}
Clock;

void NullMat(TYPE*, int, int);
TYPE diffMat(TYPE*, TYPE*, int , int);
void initialiseMat(TYPE*, int, int);
void displayMat(TYPE*, int, int);
void matMult(TYPE*, TYPE*, TYPE*, int, int, int, int);
void matMultOMP(TYPE*, TYPE*, TYPE*, int, int, int, int);

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

	if (world_rank == 0)
	{
		// size of vectors
		int rA = 1024;
		int cA = 1024;
		int rB = cA;
		int cB = 1024;

		// allocate memory on both host and device
		TYPE* A = new TYPE[rA*cA];
		TYPE* B = new TYPE[rB*cB];
		TYPE* C = new TYPE[rA*cB];
		TYPE* mpiC = new TYPE[rA*cB];

		// initialise memory
		initialiseMat(A, rA, cA);
		initialiseMat(B, rB, cB);

		displayMat(A,rA,cA);
		displayMat(B,rB,cB);

		// Benchmarking single thread
		Clock.Start();
		matMult(C,A,B,rA,cA,rB,cB);
		Clock.Stop();
		cout << "Time taken (single): " << Clock.ElapsedTime() << " ms." << endl;
		displayMat(C,rA,cB);

		// Benchmarking OpenMP
		Clock.Start();
		matMultOMP(C,A,B,rA,cA,rB,cB);
		Clock.Stop();
		cout << "Time taken (omp): " << Clock.ElapsedTime() << " ms." << endl;
		displayMat(C,rA,cB);

		cout << "Master/root process at " << processor_name << ", world size: " << world_size << endl;

		Clock.Start();
		// Send matrix sizes and matrices A, B
		MPI_Bcast(&rA, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&cA, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&rB, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&cB, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(A, rA*cA, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(B, rB*cB, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// Workload division
		int nT = world_size-1;
		int Workload = rA/nT;
		int remWorkload = rA%nT;
		int istart;
		int iend = 0;
		MPI_Request *request = new MPI_Request[nT];

		for (int i=0; i<nT; i++)
		{
			istart = iend;
			iend = i<remWorkload ? istart+Workload+1 : istart+Workload;
			MPI_Send(&istart, 1, MPI_INT, i+1, 1, MPI_COMM_WORLD);
			MPI_Send(&iend, 1, MPI_INT, i+1, 2, MPI_COMM_WORLD);
			//MPI_Recv(mpiC+(istart*cB), (iend-istart)*cB, MPI_FLOAT, i+1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Irecv(mpiC+(istart*cB), (iend-istart)*cB, MPI_FLOAT, i+1, 3, MPI_COMM_WORLD, &request[i]);
		}
		MPI_Waitall(nT, request, MPI_STATUSES_IGNORE);
		Clock.Stop();
		cout << "Time taken (MPI): " << Clock.ElapsedTime() << " ms." << endl;

		displayMat(mpiC, rA, cB);

		cout << "Matrix diff C and mpiC: " << diffMat(C, mpiC, rA, cB) << endl;

		delete[] A;
		delete[] B;
		delete[] C;
		delete[] mpiC;
		delete[] request;
	}
	else
	{
		// size of vectors
		int rA, cA, rB, cB, istart, iend;

		// Send matrix sizes and matrices A, B, mpiC
		MPI_Bcast(&rA, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&cA, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&rB, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&cB, 1, MPI_INT, 0, MPI_COMM_WORLD);
		// allocate memory on this compute node
		TYPE* A = new TYPE[rA*cA];
		TYPE* B = new TYPE[rB*cB];
		TYPE* mpiC = new TYPE[rA*cB];
		// Receive matrices A and B
		MPI_Bcast(A, rA*cA, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(B, rB*cB, MPI_FLOAT, 0, MPI_COMM_WORLD);
		// start and end row indices
		MPI_Recv(&istart, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&iend, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		cout << processor_name << ", rank: " << world_rank << " of " << world_size << "; istart: " << istart << ", iend: " << iend << endl;

		Clock.Start();
		// Matrix multiplication
		#pragma omp parallel for
		for (int i=istart; i<iend; i++)
		{
			for (int j=0; j<cB; j++)
			{
				float Sum = 0.f;
				for (int k=0; k<cA; k++)
					Sum = Sum + A[k+cA*i]*B[j+cB*k];
				mpiC[j+cB*i] = Sum;
			}
		}
		// Send my part of matrix C back to root
		MPI_Send(mpiC+(istart*cB), (iend-istart)*cB, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
		Clock.Stop();

		cout << processor_name << ", rank: " << world_rank << " of " << world_size << "; time: " << Clock.ElapsedTime() << " ms." << endl;

		delete[] A;
		delete[] B;
		delete[] mpiC;
	}
	MPI_Finalize();
	return 0;
}

TYPE diffMat(TYPE* M1, TYPE* M2, int rM, int cM)
{
	TYPE diff = 0;
	for (int i=0; i<rM; i++)
		for (int j=0; j<cM; j++)
			diff = diff + abs(M2[j+i*cM]-M1[j+i*cM]);
	return diff;
}

void NullMat(TYPE* M, int rM, int cM)
{
	for (int i=0; i<rM; i++)
		for (int j=0; j<cM; j++)
			M[j+i*cM] = 0;
}

void initialiseMat(TYPE* M, int rM, int cM)
{
	for (int i=0; i<rM; i++)
		for (int j=0; j<cM; j++)
			M[j+i*cM] = (TYPE)i+(TYPE)j*((j%3)-1);
}

void displayMat(TYPE* M, int rM, int cM)
{
	// Don't display large matrices
	if (rM > 5 || cM > 5)
		return;
	for (int i=0; i<rM; i++)
	{
		for (int j=0; j<cM; j++)
			cout << M[j+i*cM] << " ";
		cout << endl;
	}
}

void matMult(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
	for (int i=0; i<rA; i++)
	{
		for (int j=0; j<cB; j++)
		{
			TYPE Sum = 0;
			for (int k=0; k<cA; k++)
				Sum = Sum+A[k+i*cA]*B[j+k*cB];
			C[j+i*cB] = Sum;
		}
	}
}

void matMultOMP(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
#pragma omp parallel //num_threads(16)
	{
#pragma omp for
		for (int i=0; i<rA; i++)
		{
			for (int j=0; j<cB; j++)
			{
				TYPE Sum = 0;
				for (int k=0; k<cA; k++)
					Sum = Sum+A[k+i*cA]*B[j+k*cB];
				C[j+i*cB] = Sum;
			}
		}
	}
}
