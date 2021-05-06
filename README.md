# Mosaic-Image

Code implementation of the article: "[How to Build a Mosaic Image Generator from Scratch](https://medium.datadriveninvestor.com/how-to-build-your-mosaic-image-from-scratch-using-python-123e1934e977)"


Requirements: PIL, Numpy, Pandas

![alt text](https://github.com/louisowen6/Mosaic_Image/blob/master/collage_1.JPG?raw=true)

Inputs: A set of source images, a target image

Output: A mosaic image that mimics the target image based on the set of source images

-----------------------------------------------------------------------------------------------------------------------------------------------

First, the database of Average RGB from the source images folder need to be generated. 

```bash
usage: build_datasets_avg_rgb.py [-h] --SOURCE_PATH PATH

arguments:
  -h, --help            show this help message and exit
  --SOURCE_PATH         Path to source images folder
```

Then based on the generated Average RGB dataset and the target image, final mosaic image is generated.

```bash
usage: create_mosaic.py [-h] --pixel_batch_size 1 --output_width 100

optional arguments:
  -h, --help            	show this help message and exit
  --pixel_batch_size    	control the detail of picture, lower means more detail but takes longer time to produce.
  --rmse_threshold      	control the color similarity, try as lower as possible in the beginning. If adjust_threshold is 0 and if there is an error indicating "too lower threshold" then try to add the value slowly
  --allow_use_same_image	{Y,N}. If Y then the generator is allowed to use same picture many times
  --adjust_threshold	        value of adjusted threshold for pixels which have rmse higher then the given initial threshold. If 0 then it will not adjusted
  --output_width                the width of output image. Height will be adjusted to maintain the aspect ratio
  --target_PATH	      		PATH to the target image
  --OUTPUT_PATH	      		PATH to the output image
```


Created by:

Louis Owen

LinkedIn: https://www.linkedin.com/in/louisowen6
