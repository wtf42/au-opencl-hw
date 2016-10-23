#include <CL/cl.h>
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source({ cl_string });

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices);
        size_t const block_size = 16;

        // create a message to send to kernel
        std::ifstream input("input.txt");
        size_t N, M;
        input >> N >> M;
        std::vector<float> A(N * N), B(M * M), C(N * N);
        for (size_t i = 0; i < N * N; ++i) {
            input >> A[i];
        }
        for (size_t i = 0; i < M * M; ++i) {
            input >> B[i];
        }
        size_t alignedN = (N + block_size - 1) / block_size * block_size;

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(float) * A.size());
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY,  sizeof(float) * B.size());
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * C.size());

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * A.size(), A.data());
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * B.size(), B.data());

        // load named kernel from opencl source
        queue.finish();
        cl::EnqueueArgs eargs(queue, cl::NullRange,
            cl::NDRange(alignedN, alignedN), cl::NDRange(block_size, block_size));

        //cl::Kernel kernel_gmem(program, "gpu_convolution_gmem");
        //cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&, int, int> convolution_gmem(kernel_gmem);
        //cl::Event event = convolution_gmem(eargs, dev_a, dev_b, dev_c, N, M);
        //43.4s

        cl::Kernel kernel_lmem(program, "gpu_convolution_lmem");
        cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&, int, int, cl::LocalSpaceArg&, cl::LocalSpaceArg&>
            convolution_lmem(kernel_lmem);
        size_t a_local_size = (block_size + M - 1) * (block_size * M - 1);
        size_t b_local_size = M * M;
        cl::Event event = convolution_lmem(eargs, dev_a, dev_b, dev_c, N, M,
            cl::Local(a_local_size * sizeof(float)), cl::Local(b_local_size * sizeof(float)));
        //18.8s

        event.wait();
        cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_ulong elapsed_time = end_time - start_time;

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * C.size(), &C[0]);
        std::ofstream output("output.txt");
        output << std::setprecision(3) << std::fixed;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                output << C[i * N + j] << " ";
            }
            output << std::endl;
        }

        std::cout << std::setprecision(3) << std::fixed <<
            "Total time: " << elapsed_time / 1000000.0 << " s" << std::endl;
    }
    catch (const cl::Error& e)
    {
        std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
