#define SWAP(a, b) { __local float * tmp = a; a = b; b = tmp; }

__kernel void scan(__global float * input,
                   __global float * output,
                   __global float * aux,
                   __local float * a,
                   __local float * b)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
 
    a[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(uint s = 1; s < block_size; s <<= 1) {
        if(lid >= s) {
            b[lid] = a[lid] + a[lid - s];
        } else {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }
    output[gid] = a[lid];

    if (lid == block_size - 1) {
        aux[gid / block_size] = a[lid];
    }
}

__kernel void add_block_sum(__global float * aux, __global float * output)
{
    int gid = get_global_id(0);
    int group_id = get_group_id(0);

    if (group_id > 0) {
        output[gid] += aux[group_id - 1];
    }
}
