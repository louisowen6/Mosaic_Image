from PIL import Image
import numpy as np
import os
import pandas as pd
import cv2

def main(pixel_batch_size,rmse_threshold,target_PATH,DB_PATH,OUTPUT_PATH,allow_use_same_image):
	#Create dataframe of filenames per pixel batch size
	df,target_image_height,target_image_width,pixel_batch_size=find_filename_per_pixel_batch_size(100,pixel_batch_size,
		rmse_threshold,allow_use_same_image,
		target_PATH,DB_PATH)

	#Create list of filename per pixel batch size
	filenames=df.filename.tolist()

	#Adjust Mosaic Builder Size so it is the multiplies of pixel batch size
	size=check_mosaic_builder_size(size=50,pixel_batch_size=pixel_batch_size)
	print('')
	print('Used Mosaic Builder Size: ',size)
	print('')

	#Multiplier Constanst
	k=int(size/pixel_batch_size)

	#Iteration index
	index=0

	#Create Zeros Array for Mosaic
	img_concat=np.zeros((target_image_height*k,target_image_width*k,3))

	#Create Mosaic Picture
	for i in range(0,target_image_height*k,size):
		for j in range(0,target_image_width*k,size):
			img=Image.open('C:/Users/Louis Owen/Desktop/Mosaic_Image/imgs/VG_100K/'+filenames[0])
			img=np.array(img.resize((size,size)))
			if len(filenames)>0:
				filenames.pop(0)
			img_concat[i:i+size,j:j+size,:]=img
			print('Finish Creating Mosaic for pixel %d,%d \r'%(i+size,j+size),end='')
	
	img_concat=cv2.resize(img_concat, dsize=(int(target_image_width*k/2),int(target_image_height*k/2)), interpolation=cv2.INTER_AREA)
	output=Image.fromarray(img_concat.astype(np.uint8))
	output.save(OUTPUT_PATH)
	print('')
	print('\n Mosaic Image Saved! \n')


def find_filename_per_pixel_batch_size(resize_width,pixel_batch_size,threshold,allow_use_same_image,target_PATH,DB_PATH):
	'''
	Function to create dataframe consisting of 
	appropriate filename per pixel batch size
	'''
	#Import target image
	img=Image.open(target_PATH)
	#Resize target image
	basewidth = resize_width
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((basewidth,hsize), Image.ANTIALIAS)
	img=np.array(img)

	#Adjust Pixel Batch Size so it is the multiples of image's height and width
	pixel_batch_size=check_pixel_batch_size(pixel_batch_size,img)
	print('Used Pixel Batch Size: ',pixel_batch_size)
	print('')

	#Import Database of Average RGB
	df=pd.read_csv(DB_PATH)

	height=img.shape[0]
	width=img.shape[1]
	filename_list=[]

	print('Image Size: %dx%d \n'%(height,width))

	#looping for each image's pixel
	for i in range(0,height,pixel_batch_size):
	    for j in range(0,width,pixel_batch_size):
	        df,name=check_rmse(df,img[i:i+pixel_batch_size,j:j+pixel_batch_size,:],threshold=threshold,allow_repeated_use=allow_use_same_image)
	        filename_list.append(name)
	        print('Finish Creating Filename DataFrame for pixel %d,%d \r'%(i+pixel_batch_size,j+pixel_batch_size),end='')

	print('')

	#Convert List to Pandas DataFrame
	filename_df=pd.DataFrame(filename_list,columns=['filename'])

	#Export Filename for each pixel dataframe
	filename_df.to_csv('C:/Users/Louis Owen/Desktop/Mosaic_Image/filename_each_pixel.csv',index=False)

	return filename_df,height,width,pixel_batch_size


def check_pixel_batch_size(pixel_batch_size,img):
    '''
    Function to adjust Pixel Batch Size so it is the 
    multiples of image's height and width
    '''

    if (img.shape[0]%pixel_batch_size==0) and (img.shape[1]%pixel_batch_size==0):
        print(pixel_batch_size)
        return pixel_batch_size
    else:
        pixel_batch_size+=1
        return check_pixel_batch_size(pixel_batch_size,img)


def check_rmse(df,batch_pixel,threshold,allow_repeated_use=False):
    '''
    Function to calculate rmse between each pixel and average RGB of images in database
    Input: 
    df: Database of Average RGB
    Pixel: pixel list of RGB
    threshold: threshold for RMSE
    '''

    #Extract the average RGB from batch pixel
    pixel=[np.mean(batch_pixel[:,:,0]),np.mean(batch_pixel[:,:,1]),np.mean(batch_pixel[:,:,2])]

    #Slice database with RGB value around the threshold
    toy=df.copy()
    toy=toy[(toy.avg_r<=pixel[0]+threshold) & (toy.avg_r>=pixel[0]-threshold) & (toy.avg_g<=pixel[1]+threshold) & (toy.avg_g>=pixel[1]-threshold) & (toy.avg_b<=pixel[2]+threshold) & (toy.avg_b>=pixel[2]-threshold)][['avg_r','avg_g','avg_b']]
    it=toy.index.tolist()

    #Looping through the sliced database
    if len(toy)>0:
        for i in it:
            rmse=np.sqrt(np.mean((toy.loc[i,['avg_r','avg_g','avg_b']] - pixel)**2)) 
            if rmse<=threshold:
                filename=df.loc[i,'filename']
                if not allow_repeated_use:
                	df=df.drop(i).reset_index(drop=True)
                break
    else:
        print('')
        print('\n ----------------THRESHOLD TOO LOW---------------- \n')
    return df,filename


def check_mosaic_builder_size(size,pixel_batch_size):
    '''
    Function to adjust Mosaic Builder Size so it is the 
    multiplies of pixel batch size
    '''
    if (size%pixel_batch_size==0):
        return size
    else:
        size+=1
        return check_mosaic_builder_size(size,pixel_batch_size)


if __name__=='__main__':

	target_PATH='C:/Users/Louis Owen/Desktop/Mosaic_Image/test/test_image_3.jpg'
	DB_PATH='C:/Users/Louis Owen/Desktop/Mosaic_Image/Avg_RGB_dataset_VG.csv'
	OUTPUT_PATH='C:/Users/Louis Owen/Desktop/Mosaic_Image/output/mosaic_output_3_batch_size_2_thres_9_allow.png'

	main(pixel_batch_size=2,rmse_threshold=9,target_PATH=target_PATH,DB_PATH=DB_PATH,OUTPUT_PATH=OUTPUT_PATH,allow_use_same_image=True)

	#pixel_batch_size: control the detail of picture, lower means more detail but takes longer time to produce
	#rmse_threshold: control the color similarity, try as lower as possible in the beginning, if error then try to add the value slowly
	#allow_use_same_image: if true then the generator is allowed to use same picture many times