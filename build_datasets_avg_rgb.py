from PIL import Image
import numpy as np
import os
import pandas as pd

def build_dataframe_average_rgb():
    df=pd.DataFrame(columns=['filename','avg_r','avg_g','avg_b'])
    source='C:/Users/Louis Owen/Desktop/Mosaic Project/imgs/VG_100K/'
    _, _, filenames = next(os.walk(source))
    length=len(filenames)
    index=0
    print('')
    for filename in filenames:
        try:
            img=np.array(Image.open(source+filename))
            df=df.append({'filename':filename,'avg_r':np.mean(img[:,:,0]),'avg_g':np.mean(img[:,:,1]),'avg_b':np.mean(img[:,:,2])},ignore_index=True)
            index+=1
            print(('%.4f percents done \r')%(index*100/length),end='')
        except:
            index+=1
            print('\n Image Error')
    print('')
    df.to_csv('C:/Users/Louis Owen/Desktop/Mosaic Project/Avg_RGB_dataset_VG.csv',index=False)

if __name__=='__main__':
    build_dataframe_average_rgb()