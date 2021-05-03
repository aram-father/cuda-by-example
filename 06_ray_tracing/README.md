# 06_ray_tracing

**Author:** Wonseok Lee (aram_father@naver.com)

**Last update:** 2021-05-03(MON)

## What does it do?

Read the number of spheres, NUMBER_OF_SPHERES, to be generated from the user.
(This number should be larger than zero and less than 1025)

Show 1024 * 1024 sized pictures representing the ray tracing result of NUMBER_OF_SPHERES spheres.

It will show you two consecutive images and prompt output.
(When clicking a 'close' button on the top of the first image, the second one will be popped-up)

The first image & its measured execution time are generated using global memory.

The second image & its measured execution time are generated using constant memory.

## Usage

```bash
./06_ray_tracing 512
# Expected output
# It will show you a beautiful ray tracing result for 512 spheres and its execution time(using global memory)
# When you close the first image, it will show you another image using constant memory
```