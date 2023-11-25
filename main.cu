#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "stb_image_write.h"
#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"

__device__ vec3 ray_color(ray& r, hittable** world)
{
	hit_record rec;
	if ((*world)->hit(r, 0.0, FLT_MAX, rec))
	{
		return 0.5f * (rec.normal + vec3(1, 1, 1));
	}

	vec3 unit_direction = unit_vector(r.direction());
	auto a = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - a) * vec3(1.0f, 1.0f, 1.0f) + a * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(uint8_t* fb, int max_x, int max_y,
	vec3 camera_center, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 pixel00_loc, hittable** world)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
	auto ray_direction = pixel_center - camera_center;
	ray r(camera_center, ray_direction);

	vec3 pixel_color = ray_color(r, world);

	int pixel_index = j * max_x * 3 + i * 3;

	fb[pixel_index + 0] = static_cast<uint8_t>(255.999 * pixel_color.r());
	fb[pixel_index + 1] = static_cast<uint8_t>(255.999 * pixel_color.g());
	fb[pixel_index + 2] = static_cast<uint8_t>(255.999 * pixel_color.b());
}

__global__ void create_world(hittable** d_list, hittable** d_world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*(d_list) = new sphere(vec3(0, 0, -1), 0.5f);
		*(d_list + 1) = new sphere(vec3(0, -100.5f, -1), 100);
		*d_world = new hittable_list(d_list, 2);
	}
}

__global__ void free_world(hittable** d_list, hittable** d_world)
{
	delete* (d_list);
	delete* (d_list + 1);
	delete* d_world;
}

int main()
{
	int image_width = 3600;
	int image_height = 1800;
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
	auto pixel_delta_u = viewport_u / (float) image_width;
	auto pixel_delta_v = viewport_v / (float) image_height;

	// Calculate the location of the upper left pixel.
	auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
	auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

	int num_pixels = image_width * image_height;
	size_t fb_size = 3 * num_pixels * sizeof(uint8_t);

	uint8_t* fb;
	cudaMallocManaged((void**)&fb, fb_size);

	hittable** d_list;
	cudaMalloc((void**)&d_list, 2 * sizeof(hittable*));

	hittable** d_world;
	cudaMalloc((void**)&d_world, sizeof(hittable*));

	create_world<<<1, 1>>>(d_list, d_world);

	cudaGetLastError();
	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();

	dim3 blocks(image_width / thread_x, image_height / thread_y);
	dim3 threads(thread_x, thread_y);

	render<<<blocks, threads>>>(fb, image_width, image_height, 
		camera_center, pixel_delta_u, pixel_delta_v, pixel00_loc, d_world);

	cudaGetLastError();
	cudaDeviceSynchronize();

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "\nTook " << duration.count() << " milliseconds" << std::endl;

	stbi_write_png("Amethyst.png", image_width, image_height, 3, fb, image_width * 3);

	cudaDeviceSynchronize();
	free_world<<<1, 1>>>(d_list, d_world);
	
	cudaGetLastError();
	cudaFree(d_list);
	cudaFree(d_world);
	cudaFree(fb);
}