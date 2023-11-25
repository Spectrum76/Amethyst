#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "stb_image_write.h"

#include "vec3.h"

__global__ void render(vec3* fb, int max_x, int max_y)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int pixel_index = j * max_x + i;

	fb[pixel_index] = vec3(float(i) / max_x, float(j) / max_y, 0.2f);
}

int main()
{
	int image_width = 2560;
	int image_height = 1440;
	int thread_x = 8;
	int thread_y = 8;

	int num_pixels = image_width * image_height;
	size_t fb_size = num_pixels * sizeof(vec3);

	vec3* fb;
	cudaMallocManaged((void**)&fb, fb_size);

	auto start = std::chrono::high_resolution_clock::now();

	dim3 blocks(image_width / thread_x, image_height / thread_y);
	dim3 threads(thread_x, thread_y);

	render<<<blocks, threads>>>(fb, image_width, image_height);

	cudaGetLastError();
	cudaDeviceSynchronize();

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "\nTook " << duration.count() << " milliseconds" << std::endl;

	std::vector<uint8_t> image_data;

	for (int j = image_height - 1; j >= 0; --j)
	{
		for (int i = 0; i < image_width; ++i)
		{
			size_t pixel_index = j * image_width + i;
			
			int ir = static_cast<int>(255.999 * fb[pixel_index].r());
			int ig = static_cast<int>(255.999 * fb[pixel_index].g());
			int ib = static_cast<int>(255.999 * fb[pixel_index].b());

			image_data.push_back(ir);
			image_data.push_back(ig);
			image_data.push_back(ib);
		}
	}

	stbi_write_png("Amethyst.png", image_width, image_height, 3, image_data.data(), image_width * 3);

	cudaFree(fb);
}