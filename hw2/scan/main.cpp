#include <CL/cl.h>
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stack>

size_t align(size_t n, size_t block_size) {
    return (n + block_size - 1) / block_size * block_size;
}

int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        //platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source({ cl_string });

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try {
            program.build(devices);
        } catch (const cl::Error& e) {
            if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
                for (cl::Device dev : devices) {
                    // Check the build status
                    cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                    if (status != CL_BUILD_ERROR)
                        continue;

                    // Get the build log
                    std::string name = dev.getInfo<CL_DEVICE_NAME>();
                    std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                    std::cerr << "Build log for " << name << ":" << std::endl
                        << buildlog << std::endl;
                }
            } else {
                throw e;
            }
        }
        const size_t block_size = 256;

        // create a message to send to kernel
        std::ifstream input("input.txt");
        size_t N;
        input >> N;
        size_t alignedN = align(N, block_size);

        std::vector<float> A(alignedN), B(alignedN);
        for (size_t i = 0; i < N; ++i) {
            input >> A[i];
        }

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(float) * A.size());
        cl::Buffer dev_b(context, CL_MEM_READ_WRITE, sizeof(float) * B.size());

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * A.size(), &A[0]);
        queue.finish();

        // load named kernels from opencl source
        cl::Kernel scan_kernel(program, "scan");
        cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg&, cl::LocalSpaceArg&> scan(scan_kernel);
        cl::Kernel block_sum_kernel(program, "add_block_sum");
        cl::KernelFunctor<cl::Buffer&, cl::Buffer&> block_sum(block_sum_kernel);

        struct block_sum_task {
            cl::Buffer out;
            cl::Buffer aux;
            size_t size;
        };
        std::stack<block_sum_task> st;

        cl::Buffer in = dev_a;
        cl::Buffer out = dev_b;
        size_t size = alignedN;

        clock_t start_time = clock();
        while (true) {
            size_t aux_size = align(size / block_size, block_size);
            cl::Buffer aux(context, CL_MEM_READ_WRITE, sizeof(float) * aux_size);

            scan(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(size), cl::NDRange(block_size)),
                in, out, aux,
                cl::Local(block_size * sizeof(float)),
                cl::Local(block_size * sizeof(float))
            ).wait();

            st.push({ out, aux, size });

            if (size <= block_size) {
                break;
            }

            in = out = aux;
            size = aux_size;
        }

        while (!st.empty()) {
            auto task = st.top();
            st.pop();

            block_sum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(task.size), cl::NDRange(block_size)),
                task.aux, task.out
            ).wait();
        }
        clock_t end_time = clock();

        queue.enqueueReadBuffer(dev_b, CL_TRUE, 0, sizeof(float) * B.size(), &B[0]);

        std::ofstream output("output.txt");
        output << std::setprecision(3) << std::fixed;
        for (size_t i = 0; i < N; ++i) {
            output << B[i] << " ";
        }
        output << std::endl;

        std::cout << std::setprecision(3) << std::fixed <<
            "Total time: " << (end_time - start_time + 0.0) / CLOCKS_PER_SEC << " s" << std::endl;
    } catch (const cl::Error& e) {
        std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
