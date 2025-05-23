add_executable(MPIDistributedSum MPIDistributedSum.cpp)
set_target_properties(MPIMatMult PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
if(NOT MPI_CXX_FOUND)
message("MPI not found. please install an mpi implementation, for example:")
message("Windows: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi")
message("Linux: sudo apt install openmpi-bin openmpi-common libopenmpi-dev")
endif()
target_link_libraries(MPIDistributedSum PRIVATE MPI::MPI_CXX)

if("${MPI_CXX_INCLUDE_DIRS}" MATCHES "hpcx")
message("nvidia HPC-X found")
set(NVIDIA_HPCX TRUE)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
if(NOT MSVC)
if(${NVIDIA_HPCX})
add_custom_target(run COMMAND mpirun -n 4 -mca coll_hcoll_enable 0 ./MPIDistributedSum DEPENDS MPIDistributedSum)
else()
add_custom_target(run COMMAND mpirun -n 4 ./MPIDistributedSum DEPENDS MPIDistributedSum)
endif()
endif()

# Command: $(MSMPI_BIN)\mpiexec
# Command arguments: -n 4 "$(TargetPath)"
set_target_properties(MPIDistributedSum PROPERTIES VS_DEBUGGER_COMMAND           "$(MSMPI_BIN)mpiexec"
                                              VS_DEBUGGER_COMMAND_ARGUMENTS "-n 4 ..\\src\\$(Configuration)\\MPIDistributedSum")
											  