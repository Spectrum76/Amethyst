#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "stb_image_write.h"

#include "vec3.h"
#include "ray.h"

__device__ vec3 ray_color(const ray& r)
{
	vec3 unit_direction = unit_vector(r.direction());
	auto a = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - a) * vec3(1.0f, 1.0f, 1.0f) + a * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(vec3* fb, int max_x, int max_y,
	vec3 camera_center, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 pixel00_loc)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
	auto ray_direction = pixel_center - camera_center;
	ray r(camera_center, ray_direction);

	vec3 pixel_color = ray_color(r);

	int pixel_index = j * max_x + i;

	fb[pixel_index] = pixel_color;
}

int main()
{
	int image_width = 2560;
	int image_height = 1440;
	int thread_x = 8;
	int thread_y = 8;

	// Camera
	auto focal_length = 1.0f;
	auto viewport_height = 2.0f;
	auto viewport_width = viewport_height * ((image_width) / image_height);
	auto camera_center = vec3(0, 0, 0);

	// Calculate the vectors across the horizontal and down the vertical viewport edges.
	auto viewport_u = vec3(viewport_width, 0, 0);
	auto viewport_v = vec3(0, -viewport_height, 0);

	// Calculate the horizontal and vertical delta vectors from pixel to pixel.
	auto pixel_delta_u = viewport_u / image_width;
	auto pixel_delta_v = viewport_v / image_height;

	// Calculate the location of the upper left pixel.
	auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
	auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

	int num_pixels = image_width * image_height;
	size_t fb_size = num_pixels * sizeof(vec3);

	vec3* fb;
	cudaMallocManaged((void**)&fb, fb_size);

	auto start = std::chrono::high_resolution_clock::now();

	dim3 blocks(image_width / thread_x, image_height / thread_y);
	dim3 threads(thread_x, thread_y);

	render<<<blocks, threads>>>(fb, image_width, image_height, camera_center, pixel_delta_u, pixel_delta_v, pixel00_loc);

	cudaGetLastError();
	cudaDeviceSynchronize();

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "\nTook " << duration.count() << " milliseconds" << std::endl;

	std::vector<uint8_t> image_data;

	for (int j = 0; j < image_height; ++j)
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