from PIL import Image
import numpy as np
import pandas as pd

def check_rmse(df,batch_pixel,threshold):
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
                df=df.drop(i).reset_index(drop=True)
                break
    else:
        print('Not Enough Data to build Mosaic')
    return df,filename

def check_pixel_batch_size(pixel_batch_size,img):
    '''
    Function to adjust Pixel Batch Size so it is the 
    multiples of image's height and width
    '''
    if (pixel_batch_size%img.shape[0]==0) and (pixel_batch_size%img.shape[1]==0):
        return pixel_batch_size
    else:
        pixel_batch_size+=1
        check_pixel_batch_size(pixel_batch_size,img)

def find_similar_rgb(pixel_batch_size,threshold):
    #Import target image
    img=Image.open('C:/Users/Louis Owen/Desktop/Mosaic Project/test_image.jpg')
    #Resize target image
    basewidth = 100
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img=np.array(img)

    #Adjust Pixel Batch Size so it is the multiples of image's height and width
    pixel_batch_size=check_pixel_batch_size(pixel_batch_size,img)
    print('Used Pixel Batch Size: ',pixel_batch_size)

    #Import Database of Average RGB
    df=pd.read_csv('C:/Users/Louis Owen/Desktop/Mosaic Project/Avg_RGB_dataset.csv')

    height=img.shape[0]
    width=img.shape[1]
    filename_list=[]

    #looping for each image's pixel
    for i in range(0,height,pixel_batch_size):
        for j in range(0,width,pixel_batch_size):
            df,name=check_rmse(df,img[i:i+pixel_batch_size,j:j+pixel_batch_size,:],threshold=threshold)
            filename_list.append(name)
            print('Finish for pixel %d,%d \r'%(i,j),end='')

    #Convert List to Pandas DataFrame
    filename_df=pd.DataFrame(filename_list,columns=['filename'])

    #Export Filename for each pixel dataframe
    filename_df.to_csv('C:/Users/Louis Owen/Desktop/Mosaic Project/filename_each_pixel.csv',index=False)

if __name__=='__main__':
    main(pixel_batch_size=2,threshold=15)