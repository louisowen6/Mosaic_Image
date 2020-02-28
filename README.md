# Mosaic-Image
Generate Mosaic Image using [VisualGenome Dataset](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)

Requirements: PIL, Numpy, Pandas, CV2

First, the database of Average RGB need to be created using this [script](https://github.com/louisowen6/Mosaic_Image/blob/master/build_datasets_avg_rgb.py). Then run this [script](https://github.com/louisowen6/Mosaic_Image/blob/master/create_mosaic.py) to generate the mosaic images. There are several parameters need to be tuned when generating the mosaic images:

1) pixel_batch_size: control the detail of picture, lower means more detail but takes longer time to produce. 
2) rmse_threshold: control the color similarity, try as lower as possible in the beginning. If adjust_threshold is 0 and if there is an error indicating "too lower threshold" then try to add the value slowly
3) allow_use_same_image: if true then the generator is allowed to use same picture many times. Default is False
4) adjust_threshold: value of adjusted threshold for pixels which have rmse higher then the given initial threshold. If 0 then it will not adjusted. Default is 1
5) target_PATH: PATH of the target image
6) DB_PATH: PATH of the Average RGB database created using this [script](https://github.com/louisowen6/Mosaic_Image/blob/master/build_datasets_avg_rgb.py)
7) OUTPUT_PATH: destination PATH of the mosaic image output
8) output_width: the width of output image. Height will be adjusted to maintain the aspect ratio. Default is 100


---------------------------------------------------------------------------------------------------------------------------------

This is the first try of my Mosaic Image Generator Project. There will be another improvement in the future, such as adding transformed image (RGB to BGR, Grayscale, Half-Tone, Dithering, etc) into database. 



Created by:

Louis Owen

LinkedIn: https://www.linkedin.com/in/louisowen6
