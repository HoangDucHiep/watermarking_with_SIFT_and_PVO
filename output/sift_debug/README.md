# SIFT Debug Artifacts

This folder contains intermediate outputs of the custom SIFT pipeline.

## Step 0 - Input
- `step0_input_gray.png`: original grayscale image used for SIFT.

## Step 1 - Scale-Space (Gaussian + DoG)
- `step1_oct0_gaussian_s0.png`: Gaussian octave 0, scale 0.
- `step1_oct0_gaussian_s1.png`: Gaussian octave 0, scale 1.
- `step1_oct0_gaussian_s2.png`: Gaussian octave 0, scale 2.
- `step1_oct0_gaussian_s3.png`: Gaussian octave 0, scale 3.
- `step1_oct0_gaussian_s4.png`: Gaussian octave 0, scale 4.
- `step1_oct0_gaussian_s5.png`: Gaussian octave 0, scale 5.
- `step1_oct0_dog_s0.png`: DoG octave 0, scale 0.
- `step1_oct0_dog_s1.png`: DoG octave 0, scale 1.
- `step1_oct0_dog_s2.png`: DoG octave 0, scale 2.
- `step1_oct0_dog_s3.png`: DoG octave 0, scale 3.
- `step1_oct0_dog_s4.png`: DoG octave 0, scale 4.
- `step1_oct1_gaussian_s0.png`: Gaussian octave 1, scale 0.
- `step1_oct1_gaussian_s1.png`: Gaussian octave 1, scale 1.
- `step1_oct1_gaussian_s2.png`: Gaussian octave 1, scale 2.
- `step1_oct1_gaussian_s3.png`: Gaussian octave 1, scale 3.
- `step1_oct1_gaussian_s4.png`: Gaussian octave 1, scale 4.
- `step1_oct1_gaussian_s5.png`: Gaussian octave 1, scale 5.
- `step1_oct1_dog_s0.png`: DoG octave 1, scale 0.
- `step1_oct1_dog_s1.png`: DoG octave 1, scale 1.
- `step1_oct1_dog_s2.png`: DoG octave 1, scale 2.
- `step1_oct1_dog_s3.png`: DoG octave 1, scale 3.
- `step1_oct1_dog_s4.png`: DoG octave 1, scale 4.
- `step1_oct2_gaussian_s0.png`: Gaussian octave 2, scale 0.
- `step1_oct2_gaussian_s1.png`: Gaussian octave 2, scale 1.
- `step1_oct2_gaussian_s2.png`: Gaussian octave 2, scale 2.
- `step1_oct2_gaussian_s3.png`: Gaussian octave 2, scale 3.
- `step1_oct2_gaussian_s4.png`: Gaussian octave 2, scale 4.
- `step1_oct2_gaussian_s5.png`: Gaussian octave 2, scale 5.
- `step1_oct2_dog_s0.png`: DoG octave 2, scale 0.
- `step1_oct2_dog_s1.png`: DoG octave 2, scale 1.
- `step1_oct2_dog_s2.png`: DoG octave 2, scale 2.
- `step1_oct2_dog_s3.png`: DoG octave 2, scale 3.
- `step1_oct2_dog_s4.png`: DoG octave 2, scale 4.
- `step1_oct3_gaussian_s0.png`: Gaussian octave 3, scale 0.
- `step1_oct3_gaussian_s1.png`: Gaussian octave 3, scale 1.
- `step1_oct3_gaussian_s2.png`: Gaussian octave 3, scale 2.
- `step1_oct3_gaussian_s3.png`: Gaussian octave 3, scale 3.
- `step1_oct3_gaussian_s4.png`: Gaussian octave 3, scale 4.
- `step1_oct3_gaussian_s5.png`: Gaussian octave 3, scale 5.
- `step1_oct3_dog_s0.png`: DoG octave 3, scale 0.
- `step1_oct3_dog_s1.png`: DoG octave 3, scale 1.
- `step1_oct3_dog_s2.png`: DoG octave 3, scale 2.
- `step1_oct3_dog_s3.png`: DoG octave 3, scale 3.
- `step1_oct3_dog_s4.png`: DoG octave 3, scale 4.

## Step 2/3/4 - Extrema, Refinement, Orientation
- `step2_oct0_raw_vs_refined.png`: octave 0; yellow=raw extrema, green=refined extrema. raw=586, refined=258, oriented=282.
- `step2_oct1_raw_vs_refined.png`: octave 1; yellow=raw extrema, green=refined extrema. raw=166, refined=72, oriented=81.
- `step2_oct2_raw_vs_refined.png`: octave 2; yellow=raw extrema, green=refined extrema. raw=59, refined=30, oriented=31.
- `step2_oct3_raw_vs_refined.png`: octave 3; yellow=raw extrema, green=refined extrema. raw=9, refined=4, oriented=0.

## Step 5 - Final Keypoint Selection
- `step5_final_keypoints.png`: selected keypoints after response sort + boundary + spacing filter.

## Summary
- Raw oriented keypoints before final filtering: 394
- Final selected keypoints: 20 (requested 20)
