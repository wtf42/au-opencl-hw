__kernel void gpu_convolution_gmem(__global float * a, __global float * b,
                                   __global float * c, int n, int m)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= n || col >= n)
        return;

    float sum = 0;
	int hm = m / 2;

	for (int i = -hm; i <= hm; ++i) {
		for (int j = -hm; j <= hm; ++j) {
		    int ai = row + i, aj = col + j;
			float a_val = (ai >= 0 && ai < n && aj >= 0 && aj < n) ? a[ai * n + aj] : 0;
			sum += b[(i + hm) * m + (j + hm)] * a_val;
		}
	}
    c[row * n + col] = sum;
}

__kernel void gpu_convolution_lmem(__global float * a, __global float * b,
                                   __global float * c, int n, int m,
								   __local float * a_local, __local float * b_local)
{
    int ti = get_local_id(0);
    int tj = get_local_id(1);
	int block_rows = get_local_size(0);
	int block_cols = get_local_size(1);
	
    int row = get_global_id(0);
    int col = get_global_id(1);
	
	int cache_rows = block_rows + m - 1;
	int cache_cols = block_rows + m - 1;

	int hm = m / 2;

	for (int i = -hm; i < block_rows + hm; i += block_rows) {
		for (int j = -hm; j < block_cols + hm; j += block_cols) {
		    int ci = i + hm + ti, cj = j + hm + tj;
			if (ci < cache_rows && cj < cache_cols) {
				int ai = row + i, aj = col + j;
				float a_val = (ai >= 0 && ai < n && aj >= 0 && aj < n) ? a[ai * n + aj] : 0;
				a_local[ci * cache_cols + cj] = a_val;
			}
		}
	}

	for (int i = 0; i < m; i += block_rows) {
		for (int j = 0; j < m; j += block_cols) {
			int ci = i + ti, cj = j + tj;
			if (ci < m && cj < m) {
				b_local[ci * m + cj] = b[ci * m + cj];
			}
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

    if (row >= n || col >= n)
        return;

    float sum = 0;
    for (int i = -hm; i <= hm; ++i) {
        for (int j = -hm; j <= hm; ++j) {
			int ci = i + hm + ti, cj = j + hm + tj;
            sum += b_local[(i + hm) * m + (j + hm)] * a_local[ci * cache_cols + cj];
        }
    }
    c[row * n + col] = sum;
}