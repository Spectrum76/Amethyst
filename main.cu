#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "stb_image_write.h"
#include "vec3.h"
#include "ray.h"

__device__ float hit_sphere(const vec3& center, float radius, const ray& r)
{
	vec3 oc = r.origin() - center;
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant < 0)
		return -1.0f;
	else
		return (-half_b - sqrt(discriminant)) / a;
}

__device__ vec3 ray_color(const ray& r)
{
	auto t = hit_sphere(vec3(0, 0, -1), 0.5f, r);
	if (t > 0.0)
	{
		vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
		return 0.5f * vec3(N.x() + 1, N.y() + 1, N.z() + 1);
	}

	vec3 unit_direction = unit_vector(r.direction());
	auto a = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - a) * vec3(1.0f, 1.0f, 1.0f) + a * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(uint8_t* fb, int max_x, int max_y,
	vec3 camera_center, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 pixel00_loc)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
	auto ray_direction = pixel_center - camera_center;
	ray r(camera_center, ray_direction);

	vec3 pixel_color = ray_color(r);

	int pixel_index = j * max_x * 3 + i * 3;

	fb[pixel_index + 0] = static_cast<uint8_t>(255.999 * pixel_color.r());
	fb[pixel_index + 1] = static_cast<uint8_t>(255.999 * pixel_color.g());
	fb[pixel_index + 2] = static_cast<uint8_t>(255.999 * pixel_color.b());
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
	auto pixel_delta_u = viewport_u / image_width;
	auto pixel_delta_v = viewport_v / image_height;

	// Calculate the location of the upper left pixel.
	auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
	auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

	int num_pixels = image_width * image_height;
	size_t fb_size = 3 * num_pixels * sizeof(uint8_t);

	uint8_t* fb;
	cudaMallocManaged((void**)&fb, fb_size);

	auto start = std::chrono::high_resolution_clock::now();

	dim3 blocks(image_width / thread_x, image_height / thread_y);
	dim3 threads(thread_x, thread_y);

	render <<<blocks, threads>>> (fb, image_width, image_height, camera_center, pixel_delta_u, pixel_delta_v, pixel00_loc);

	cudaGetLastError();
	cudaDeviceSynchronize();

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "\nTook " << duration.count() << " milliseconds" << std::endl;

	stbi_write_png("Amethyst.png", image_width, image_height, 3, fb, image_width * 3);

	cudaFree(fb);
}