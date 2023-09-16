import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import scipy.ndimage
import skimage

#-------------------Conversions--------------------------------------------
def type_conversion_to_int32(matrix_image):
    matrix_image=np.asarray(matrix_image, dtype=np.int32)
    
    return matrix_image

def type_conversion_to_float64(matrix_image):
    matrix_image=np.asarray(matrix_image, dtype=np.float64)
    
    return matrix_image

def float_to_uint8(Image):
    
    Image=(Image-np.min(Image))*255/(np.max(Image)-np.min(Image))
    Image=np.around(Image)
    Image=np.asarray(Image, dtype=np.uint8)
    
    return Image

def image_float16_to_float64(Image):
    
    Image=(Image-np.min(Image))*255/(np.max(Image)-np.min(Image))
    Image=np.around(Image)

    
    return Image

#-------------------------------------------------------------------------------------

def matrix_with_negative_values_to_uint8_image(matrix):
    Image2=matrix
    Image2 = Image2.clip(min=0)
    Image2=np.uint8(Image2)
    
    return Image2



def image_rgb_to_hsv(image):
    img = rgb2hsv(image)
    
    return img




def rgb_to_gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def gray_to_binary(Grayscale_image):
    thresh = threshold_otsu(Grayscale_image)
    Binary_image = Grayscale_image > thresh
    Binary_image=np.asarray(Binary_image, dtype=np.int32) 
    
    return Binary_image

#----------------------------------GAUSS---------------------------------------------------

def Gauss_smooth_mask(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def Gaussian_noise(img):
    
    Image_Gaussian_noise=skimage.util.random_noise(img, mode='gaussian')
#    Image_Gaussian_noise=np.asarray(Image_Gaussian_noise, dtype=np.uint8)
    
    return Image_Gaussian_noise

def mean_square_filter(n):
    
    filter_square=np.ones((n,n))/(n*n)
    
    return filter_square

def denoise_gaussian_image_by_mean_filter(filter_square,image):
    
    Image_filtered=scipy.ndimage.convolve(image,filter_square)
    
    return Image_filtered


def gaussian_noise_to_gaussian_smoothing(Image_Gaussian_noise,Gauss_mask):
    
    Image_filtered_by_mask_1_5=scipy.ndimage.convolve(Image_Gaussian_noise,Gauss_mask)
    Image_filtered_by_mask_1_5=(Image_filtered_by_mask_1_5-np.min(Image_filtered_by_mask_1_5))*255/(np.max(Image_filtered_by_mask_1_5)-np.min(Image_filtered_by_mask_1_5))
    Image_filtered_by_mask_1_5=np.around(Image_filtered_by_mask_1_5)
    Image_filtered_by_mask_1_5=np.asarray(Image_filtered_by_mask_1_5, dtype=np.uint8)
    
    return Image_filtered_by_mask_1_5

def image_filtering(Image,mask):
    
    Image_filtered_by_mask_1_5=scipy.ndimage.convolve(Image,mask)
    Image_filtered_by_mask_1_5=(Image_filtered_by_mask_1_5-np.min(Image_filtered_by_mask_1_5))*255/(np.max(Image_filtered_by_mask_1_5)-np.min(Image_filtered_by_mask_1_5))
    Image_filtered_by_mask_1_5=np.around(Image_filtered_by_mask_1_5)
    Image_filtered_by_mask_1_5=np.asarray(Image_filtered_by_mask_1_5, dtype=np.uint8)
    
    return Image_filtered_by_mask_1_5

#-------------------------------------MAGIC_MATRIX-----------------------------------------------------

def magic(n):
  n = int(n)
  if n < 3:
    raise ValueError("Size must be at least 3")
  if n % 2 == 1:
    p = np.arange(1, n+1)
    return n*np.mod(p[:, None] + p - (n+3)//2, n) + np.mod(p[:, None] + 2*p-2, n) + 1
  elif n % 4 == 0:
    J = np.mod(np.arange(1, n+1), 4) // 2
    K = J[:, None] == J
    M = np.arange(1, n*n+1, n)[:, None] + np.arange(n)
    M[K] = n*n + 1 - M[K]
  else:
    p = n//2
    M = magic(p)
    M = np.block([[M, M+2*p*p], [M+3*p*p, M+p*p]])
    i = np.arange(p)
    k = (n-2)//4
    j = np.concatenate((np.arange(k), np.arange(n-k+1, n)))
    M[np.ix_(np.concatenate((i, i+p)), j)] = M[np.ix_(np.concatenate((i+p, i)), j)]
    M[np.ix_([k, k+p], [0, k])] = M[np.ix_([k+p, k], [0, k])]
  return M 
#---------------------------------GITNIASEIS-------------------------------------------------
def gitniasi_4_perigramma(Image_Binary):
    cords_x=[]
    cords_y=[]
    
    for i in range(len(Image_Binary)-1):
        for j in range(len(Image_Binary[0])-1):
            if (Image_Binary[i,j]==1 and (Image_Binary[i,j-1]==0 or Image_Binary[i,j+1]==0 or Image_Binary[i-1,j]==0 or Image_Binary[i+1,j]==0)):
                cords_x.append(i)
                cords_y.append(j)
                    
    cords_x=np.asarray(cords_x, dtype=np.int32)
    cords_y=np.asarray(cords_y, dtype=np.int32)
      
    Pinakas_cords=np.transpose(np.vstack((cords_x,cords_y)))
    
    
    return Pinakas_cords

def gitniasi_8_perigramma(Image_Binary):
    new_cords_x2=[]
    new_cords_y2=[]
    for i in range(len(Image_Binary)-1):
        for j in range(len(Image_Binary)-1):
            if (Image_Binary[i,j]==1 and 
                (Image_Binary[i,j-1]==0 or
                 Image_Binary[i,j+1]==0 or
                 Image_Binary[i-1,j]==0 or 
                 Image_Binary[i+1,j]==0 or
                 Image_Binary[i-1,j-1]==0 or
                 Image_Binary[i-1,j+1]==0 or
                 Image_Binary[i+1,j-1]==0 or
                 Image_Binary[i+1,j+1]==0)):
                new_cords_x2.append(i)
                new_cords_y2.append(j)
            
    new_cords_x2=np.asarray(new_cords_x2, dtype=np.int32)
    new_cords_y2=np.asarray(new_cords_y2, dtype=np.int32)
         
    Pinakas_cords2=np.transpose(np.vstack((new_cords_x2,new_cords_y2)))
    
    
    return Pinakas_cords2

def paint_edges_gitniasi_4_8(Original_image,Pinakas_cords):
    image=Original_image
    Image_Binary=gray_to_binary(image)
    New_Binary=np.zeros((len(image),len(image[0])))

    New_Binary=np.asarray(New_Binary, dtype=np.bool)
    New_Binary=New_Binary.astype(int)
        
    New_Binary[Pinakas_cords[:,0],Pinakas_cords[:,1]]=1

    Grayscale_Image=Image_Binary*255-New_Binary*100
    Grayscale_Image=np.asarray(Grayscale_Image, dtype=np.uint8)
    
    return Grayscale_Image



#--------------------------------SALT_AND_PEPPER--------------------------------------------------
def Salt_and_Pepper_noise(image,amount_value):
    image_s_n_p=skimage.util.random_noise(image, mode='s&p', amount=amount_value,seed=None)
    image_s_n_p=(image_s_n_p-np.min(image_s_n_p))*255/(np.max(image_s_n_p)-np.min(image_s_n_p))
    
    return image_s_n_p

def denoise_Salt_and_Pepper_by_median_filter(image_s_n_p,kernel_size_value):
    
    Image_denoise=scipy.signal.medfilt2d(image_s_n_p,kernel_size=kernel_size_value)
    
    return Image_denoise


#---------------------------IMAGE_TRANSFORMS-------------------------------------------------------------

def image_to_negative(image):
    negative_img=np.invert(image)
    
    return negative_img

def power_image_transform(image,power_value):
    
    img_power_2=np.power(image,power_value)
    img_power_2=(img_power_2-np.min(img_power_2))*255/(np.max(img_power_2)-np.min(img_power_2))
    img_power_2=np.around(img_power_2)
    img_power_2=np.asarray(img_power_2, dtype=np.uint8)
    
    return img_power_2



def exponetial_transform_dark_detailing(Original_image):
    log_zero_avoidance=1

    b=255/(np.log(1+255*log_zero_avoidance))

    Image_exponetial_transform =(1/log_zero_avoidance)* (np.exp(Original_image/b)-1)
    
    Image_exponetial_transform=float_to_uint8(Image_exponetial_transform)
    
    return Image_exponetial_transform

def logarithmic_transform_light_detailing(Original_image):
    log_zero_avoidance=1

    b=255/(np.log(1+255*log_zero_avoidance))

    Image_logarithmic_transform =b*np.log(1+log_zero_avoidance*Original_image)
    
    Image_logarithmic_transform=float_to_uint8(Image_logarithmic_transform)
    
    return Image_logarithmic_transform

#-----------------------------DFT_TRANSFORM--------------------------------------

def DFT_transform_raw(image):
    Grayscale_Image_DFT=np.fft.fft2(image)
    
    return Grayscale_Image_DFT



def Centered_DFT_transform_raw(Grayscale_Image_DFT):
    Grayscale_Image_Centered_DFT=np.fft.fftshift(Grayscale_Image_DFT)
    
    return Grayscale_Image_Centered_DFT

def Inverted_DFT_transform_raw(image):
    Grayscale_Image_Inverted_DFT=np.fft.ifft2(image)
    
    return Grayscale_Image_Inverted_DFT
#-------------------------------DFT_FOR_IMSHOW-----------------------------------------
def Any_DFT_to_uint8(DFT_image):
    
    Grayscale_Image_DFT_normalized=np.log(1+np.abs(DFT_image))
    Grayscale_Image_DFT_Final=float_to_uint8(Grayscale_Image_DFT_normalized)
    
    return Grayscale_Image_DFT_Final
#----------------------------DFT_FILTRA---------------------------------------------------

def katofli_ideato_xamiloperato_filtro(image,katofli_value):
    columns=len(image)
    rows=len(image[0])

    katheta = np.linspace(-1*(int(rows/2)-1), int(rows/2),rows )
    orizontia = np.linspace(-1*(int(columns/2)-1), int(columns/2), columns)

    x,y = np.meshgrid(katheta, orizontia)

    apostasi=np.sqrt(np.power(x,2)+np.power(y,2))

    katofli=apostasi<katofli_value
    
    return katofli

def eikona_ideato_xamiloperato_filtro(katofli,Grayscale_Image_Centered_DFT):
    
    mask_circle=katofli*Grayscale_Image_Centered_DFT
    
    Filtered_image=np.fft.ifft2(mask_circle)
    Filtered_image_normalized=np.abs(Filtered_image)
    Filtered_image_final=image_float16_to_float64(Filtered_image_normalized)
    
    return Filtered_image_final
    
def Butterworth_filtering(Grayscale_Image,entasi_filtrou,basi):
    columns=len(Grayscale_Image)
    rows=len(Grayscale_Image[0])

    katheta = np.linspace(-127, 128,rows )
    orizontia = np.linspace(-127, 128, columns)

    x,y = np.meshgrid(katheta, orizontia)

    apostasi=np.sqrt(np.power(x,2)+np.power(y,2))
    n=1
    base=15
    katofli_n_1=1/(1+(np.power(apostasi,2*n)/np.power(base,2*n)))
    
    Grayscale_Image_DFT=np.fft.fft2(Grayscale_Image)
    Grayscale_Image_Centered_DFT=np.fft.fftshift(Grayscale_Image_DFT)
    
    mask_circle_n_1=katofli_n_1*Grayscale_Image_Centered_DFT
    
    Filtered_image_n_1=np.fft.ifft2(mask_circle_n_1)
    Filtered_image_normalized_n_1=np.abs(Filtered_image_n_1)
    Filtered_image_final_n_1=image_float16_to_float64(Filtered_image_normalized_n_1)
    
    return Filtered_image_final_n_1

#-------------------------------------------------------------------------------------------
def cords_finder(matrix_or_image,value,telestis_mikro_iso_megal):
    
    if(telestis_mikro_iso_megal==0):
        list_cords=([d for d in matrix_or_image ==value ])
        x_y_cords2=[]
        cords_x=[]  
        cords_y=[]
        
        for i in range(len(list_cords)):
            x_y_cords2.append([k for k, e in enumerate(list_cords[i]) if e == True ])
                    
        for i in range(len(x_y_cords2)):
            temp=len(x_y_cords2[i])
            if(temp>0):
                for k in range(temp):
                    cords_x.append(i)
                    cords_y.append(x_y_cords2[i][k])

        cords_x= np.array(cords_x)
        cords_y= np.array(cords_y)
        return cords_x,cords_y
    
    if(telestis_mikro_iso_megal==1):
        list_cords=([d for d in matrix_or_image >value ])
        x_y_cords2=[]
        cords_x=[]  
        cords_y=[]
        
        for i in range(len(list_cords)):
            x_y_cords2.append([k for k, e in enumerate(list_cords[i]) if e == True ])
                    
        for i in range(len(x_y_cords2)):
            temp=len(x_y_cords2[i])
            if(temp>0):
                for k in range(temp):
                    cords_x.append(i)
                    cords_y.append(x_y_cords2[i][k])

        cords_x= np.array(cords_x)
        cords_y= np.array(cords_y)
        return cords_x,cords_y
    
    if(telestis_mikro_iso_megal==-1):
        list_cords=([d for d in matrix_or_image <value ])
        x_y_cords2=[]
        cords_x=[]  
        cords_y=[]
        
        for i in range(len(list_cords)):
            x_y_cords2.append([k for k, e in enumerate(list_cords[i]) if e == True ])
                    
        for i in range(len(x_y_cords2)):
            temp=len(x_y_cords2[i])
            if(temp>0):
                for k in range(temp):
                    cords_x.append(i)
                    cords_y.append(x_y_cords2[i][k])

        cords_x= np.array(cords_x)
        cords_y= np.array(cords_y)
        return cords_x,cords_y


    
    
    











