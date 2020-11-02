"""
Created on Tue Oct 20 11:16:24 2020

@author: Jay Patel
"""

import cv2
import math
import glob
import sys
import xlwt
from xlwt import Workbook
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size_x, size_y=None):
    size_x = math.ceil(size_x)
    if not size_y:
        size_y = size_x
    else:
        size_y = math.ceil(size_y)
    x, y = np.mgrid[-size_x:size_x+1, -size_y:size_y+1]
    g = np.exp(-(x**2/size_x + y**2/size_y))
    GK = g / g.sum()
    plt.imshow(GK, cmap='jet')
    plt.show()
    return GK

def hexagon_kernel(size_x, size_y=None):
    size_x = math.ceil(size_x)
    if not size_y:
        size_y = size_x
    else:
        size_y = math.ceil(size_y)
    temp = np.zeros(((size_x*2)+1, (size_y)+1), dtype=np.uint8)
    col = 0
    while col <= (size_y):
        j = col
        if col > size_x:
            j = size_x
        row = (size_x-j)
        while row <= (size_x+j):
            temp[row][col] = 1
            row+=1
        col+=1
    temp_flip = np.flip(temp)
    temp_flip = np.delete(temp_flip, 0, axis=1)
    HK = np.concatenate((temp, temp_flip), axis = 1)
    plt.imshow(HK, cmap='jet')
    plt.show()
    return HK

def rectangle_kernel(size_x, size_y=None):
    size_x = math.ceil(size_x)
    if not size_y:
        size_y = size_x
    else:
        size_y = math.ceil(size_y)
    if size_x < 2 or size_y < 2:
        raise ValueError("size_x and size_y should greater than 2.")
    RK = np.zeros(((size_y), (size_x)), dtype=np.uint8)
    row, col = RK.shape
    for x in range(row):
        for y in range(col):
            RK[x][y] = 1
    plt.imshow(RK, cmap='jet')
    plt.show()
    return RK

def min_filter(image,nr,nc):
    filter_image = np.zeros((nr, nc), dtype = np.uint8)
    for i in range(nr-1):
        for j in range(nc-1):
            stemp = [image[i-1,j-1],image[i-1,j],image[i-1,j+1],
                     image[i,j-1],image[i,j],image[i,j+1],
                     image[i+1,j-1],image[i+1,j],image[i+1,j+1]]
            stemp.sort()
            small = stemp[0]
            filter_image[i,j] = small
    return filter_image

def max_filter(image,nr,nc):
    filter_image = np.zeros((nr, nc), dtype = np.uint8)
    for i in range(nr-1):
        for j in range(nc-1):
            ltemp = [image[i-1,j-1],image[i-1,j],image[i-1,j+1],
                    image[i,j-1],image[i,j],image[i,j+1],
                    image[i+1,j-1],image[i+1,j],image[i+1,j+1]]
            ltemp.sort()
            large = ltemp[-1]
            filter_image[i,j] = large
    return filter_image

def median_filter(image,nr,nc):
    filter_image = np.zeros((nr, nc), dtype = np.uint8)
    for i in range(nr-1):
        for j in range(nc-1):
            mdtemp = [image[i-1,j-1],image[i-1,j],image[i-1,j+1],
                    image[i,j-1],image[i,j],image[i,j+1],
                    image[i+1,j-1],image[i+1,j],image[i+1,j+1]]
            mdtemp.sort()
            median = mdtemp[5]
            filter_image[i,j] = median
    return filter_image

def midpoint_filter(image,nr,nc):
    filter_image = np.zeros((nr, nc), dtype = np.uint8)
    for i in range(nr-1):
        for j in range(nc-1):
            mptemp = [image[i-1,j-1],image[i-1,j],image[i-1,j+1],
                    image[i,j-1],image[i,j],image[i,j+1],
                    image[i+1,j-1],image[i+1,j],image[i+1,j+1]]
            mptemp.sort()
            small = mptemp[0]
            large = mptemp[-1]
            midpoint = int((small+large)/2)
            filter_image[i,j] = midpoint
    return filter_image

def mean_filter(image,nr,nc):
    filter_image = np.zeros((nr, nc), dtype = np.uint8)
    for i in range(nr-1):
        for j in range(nc-1):
            mptemp = [image[i-1,j-1],image[i-1,j],image[i-1,j+1],
                    image[i,j-1],image[i,j],image[i,j+1],
                    image[i+1,j-1],image[i+1,j],image[i+1,j+1]]
            mptemp.sort()
            mean = sum(mptemp)/len(mptemp)
            filter_image[i,j] = mean
    return filter_image

def ideal_hpf(image, radius):
    radius = int(radius)
    (nr, nc) = image.shape
    pad_image = np.zeros((2 * nr, 2 * nc), dtype=np.uint8)
    pad_image[0:nr, 0:nc] = image
    
    # Fourier transform of an image and its spectrum
    f = np.fft.fft2(pad_image)
    f = np.fft.fftshift(f)
    fabs = np.abs(f)
    spectrum = 255 * np.log(fabs + 1) / np.log(np.max(fabs + 1))
    spectrum = spectrum.astype(np.uint8)
    
    # Mask generation
    dx=nr ; dy=nc;
    mask_ideal = np.zeros((2 * nr, 2 * nc))
    for i in range(1, 2 * nr + 1):
        for j in range(1, 2 * nc + 1):
            dist = np.sqrt(np.power(dx - i, 2) + np.power(dy - j, 2))
            if dist >= radius:
                mask_ideal[i - 1, j - 1] = 1
    
    mask = 255 * mask_ideal
    mask = mask.astype(np.uint8)
    cut_spectrum = f * mask
    
    # Inverse Fourier transform
    back_image = np.abs(np.fft.ifft2(np.fft.fftshift(cut_spectrum)))
    if np.max(back_image) != 0:
        out_image = 255 * (back_image / np.max(back_image))
    else:
        out_image = 255 * back_image
    out_image = out_image.astype(np.uint8)        
    return out_image[0:nr, 0:nc]

def butter_hpf(image, radius, order):
    radius = int(radius)
    order = int(order)
    (nr, nc) = image.shape
    pad_image = np.zeros((2 * nr, 2 * nc), dtype=np.uint8)
    pad_image[0:nr, 0:nc] = image
    
    # Fourier transform of an image and its spectrum
    f = np.fft.fft2(pad_image)
    f = np.fft.fftshift(f)
    fabs = np.abs(f)
    spectrum = 255 * np.log(fabs + 1) / np.log(np.max(fabs + 1))
    spectrum = spectrum.astype(np.uint8)
    
    # Mask generation
    dx=nr ; dy=nc;
    mask_butter = np.zeros((2*nr,2*nc))
    for i in range (1,2*nr+1):
        for j in range(1,2*nc+1):
            dist=np.sqrt(np.power(dx-i,2)+np.power(dy-j,2))
            if dist != 0:
                mask_butter[i-1,j-1]=1/(1+((dist/radius )**(2*order)))

    mask = 255 * mask_butter
    mask = mask.astype(np.uint8)
    cut_spectrum = f * mask
    
    # Inverse Fourier transform
    back_image = np.abs(np.fft.ifft2(np.fft.fftshift(cut_spectrum)))
    if np.max(back_image) != 0:
        out_image = 255 * (back_image / np.max(back_image))
    else:
        out_image = 255 * back_image
    out_image = out_image.astype(np.uint8)        
    return out_image[0:nr, 0:nc]

def gauss_hpf(image, radius):
    radius = int(radius)
    (nr, nc) = image.shape
    pad_image = np.zeros((2 * nr, 2 * nc), dtype=np.uint8)
    pad_image[0:nr, 0:nc] = image
    
    # Fourier transform of an image and its spectrum
    f = np.fft.fft2(pad_image)
    f = np.fft.fftshift(f)
    fabs = np.abs(f)
    spectrum = 255 * np.log(fabs + 1) / np.log(np.max(fabs + 1))
    spectrum = spectrum.astype(np.uint8)
    
    # Mask generation
    dx=nr ; dy=nc;
    mask_gauss = np.zeros((2 * nr, 2 * nc))
    for i in range(1, 2 * nr + 1):
        for j in range(1, 2 * nc + 1):
            dist = np.sqrt(np.power(dx - i, 2) + np.power(dy - j, 2))
            if dist != 0:
                mask_gauss[i - 1, j - 1] = np.exp(((radius / dist) ** 2) * (-0.5))
    
    mask = 255 * mask_gauss
    mask = mask.astype(np.uint8)
    cut_spectrum = f * mask
    
    # Inverse Fourier transform
    back_image = np.abs(np.fft.ifft2(np.fft.fftshift(cut_spectrum)))
    if np.max(back_image) != 0:
        out_image = 255 * (back_image / np.max(back_image))
    else:
        out_image = 255 * back_image
    out_image = out_image.astype(np.uint8)        
    return out_image[0:nr, 0:nc]

def ideal_lpf(image, radius):
    radius = int(radius)
    (nr, nc) = image.shape
    pad_image = np.zeros((2 * nr, 2 * nc), dtype=np.uint8)
    pad_image[0:nr, 0:nc] = image
    
    # Fourier transform of an image and its spectrum
    f = np.fft.fft2(pad_image)
    f = np.fft.fftshift(f)
    fabs = np.abs(f)
    spectrum = 255 * np.log(fabs + 1) / np.log(np.max(fabs + 1))
    spectrum = spectrum.astype(np.uint8)
    
    # Mask generation
    dx=nr ; dy=nc;
    mask_ideal = np.zeros((2*nr,2*nc))
    for i in range (1,2*nr+1):
        for j in range(1,2*nc+1):
            dist=np.sqrt(np.power(dx-i,2)+np.power(dy-j,2))
            if dist<=radius:
                mask_ideal[i-1,j-1]=1
    
    mask = 255 * mask_ideal
    mask = mask.astype(np.uint8)
    cut_spectrum = f * mask
    
    # Inverse Fourier transform
    back_image = np.abs(np.fft.ifft2(np.fft.fftshift(cut_spectrum)))
    out_image = 255 * (back_image / np.max(back_image))        
    return out_image[0:nr, 0:nc]

def butter_lpf(image, radius, order):
    radius = int(radius)
    order = int(order)
    (nr, nc) = image.shape
    pad_image = np.zeros((2 * nr, 2 * nc), dtype=np.uint8)
    pad_image[0:nr, 0:nc] = image
    
    # Fourier transform of an image and its spectrum
    f = np.fft.fft2(pad_image)
    f = np.fft.fftshift(f)
    fabs = np.abs(f)
    spectrum = 255 * np.log(fabs + 1) / np.log(np.max(fabs + 1))
    spectrum = spectrum.astype(np.uint8)
    
    # Mask generation
    dx=nr ; dy=nc;
    mask_butter = np.zeros((2*nr,2*nc))
    for i in range (1,2*nr+1):
        for j in range(1,2*nc+1):
            dist=np.sqrt(np.power(dx-i,2)+np.power(dy-j,2))
            mask_butter[i-1,j-1]=1/(1+((dist/radius )**(2*order)))
    
    mask = 255 * mask_butter
    mask = mask.astype(np.uint8)
    cut_spectrum = f * mask
    
    # Inverse Fourier transform
    back_image = np.abs(np.fft.ifft2(np.fft.fftshift(cut_spectrum)))
    out_image = 255 * (back_image / np.max(back_image))        
    return out_image[0:nr, 0:nc]

def gauss_lpf(image, radius):
    radius = int(radius)
    (nr, nc) = image.shape
    pad_image = np.zeros((2 * nr, 2 * nc), dtype=np.uint8)
    pad_image[0:nr, 0:nc] = image
    
    # Fourier transform of an image and its spectrum
    f = np.fft.fft2(pad_image)
    f = np.fft.fftshift(f)
    fabs = np.abs(f)
    spectrum = 255 * np.log(fabs + 1) / np.log(np.max(fabs + 1))
    spectrum = spectrum.astype(np.uint8)
    
    # Mask generation
    dx=nr ; dy=nc;
    mask_gauss = np.zeros((2*nr,2*nc))
    for i in range (1,2*nr+1):
        for j in range(1,2*nc+1):
            dist=np.sqrt(np.power(dx-i,2)+np.power(dy-j,2))
            mask_gauss[i-1,j-1]=np.exp(((dist/radius)**2)*(-0.5) )
    
    mask = 255 * mask_gauss
    mask = mask.astype(np.uint8)
    cut_spectrum = f * mask
    
    # Inverse Fourier transform
    back_image = np.abs(np.fft.ifft2(np.fft.fftshift(cut_spectrum)))
    out_image = 255 * (back_image / np.max(back_image))        
    return out_image[0:nr, 0:nc]

def gamma_correction(Image_original, gamma=1):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    result = cv2.LUT(Image_original, lookUpTable)
    return result

def thresholdvalue(image):
    (nr, nc) = image.shape
    avg = image.mean()
    T=avg+50
    Tn=0
    
    while 1:
        m1=0
        m2=0
        c1=1
        c2=1
        for i in range(nr):
            for j in range(nc):
                if(0<=(image[i][j])<T):
                    m1+=(image[i][j])  
                    c1+=1
                else:
                    m2+=(image[i][j])
                    c2+=1
        m1=m1/c1
        m2=m2/c2
        Tn=(m1+m2)/2
        if(abs(Tn-T)<9):
            break;
        else:
            T=Tn
    return Tn

def contrasstg(image, nr, nc):
    design_ratio = (image.var())/(image.mean())
    final = np.zeros((nr,nc),dtype = 'float16')
    r1=design_ratio
    s1=design_ratio   
    r2=(design_ratio+70)
    s2=(design_ratio+170)
    #slopes
    m1=(s1-0)/(r1-0)
    m2=(s2-s1)/(r2-r1)
    m3=(256-s2)/(256-r2)
    for i in range(nr):
        for j in range(nc):
            if image[i][j]<=r1:
                final[i,j] = m1*image[i][j]
            elif image[i][j]<=r2:
                final[i,j] = m2*image[i][j]
            else:
                final[i,j] = m3*image[i][j]
    result = np.array(final,dtype=np.uint8)
    return result

def density_counter(image, tray_size):
    (nr, nc) = image.shape
    den_image = np.zeros((nr, nc), dtype = np.uint8)
    if tray_size <= 3:
        for i in range(nr-3):
            for j in range(nc-3):
                dtemp = [image[i][j],image[i][j+1],image[i][j+2],
                     image[i+1][j],image[i+1][j+1],image[i+1][j+2],
                     image[i+2][j],image[i+2][j+1],image[i+2][j+2]]
                checksum = sum(dtemp)
                if checksum >= 2040:
                    den_image[i][j]=den_image[i][j+1]=den_image[i][j+2]= \
                    den_image[i+1][j]=den_image[i+1][j+1]=den_image[i+1][j+2]= \
                    den_image[i+2][j]=den_image[i+2][j+1]=den_image[i+2][j+2]=255
    elif tray_size == 4:
        for i in range(nr-4):
            for j in range(nc-4):
                dtemp = [image[i][j],image[i][j+1],image[i][j+2],image[i][j+3],
                     image[i+1][j],image[i+1][j+1],image[i+1][j+2],image[i+1][j+3],
                     image[i+2][j],image[i+2][j+1],image[i+2][j+2],image[i+2][j+3],
                     image[i+3][j],image[i+3][j+1],image[i+3][j+2],image[i+3][j+3]]
                checksum = sum(dtemp)
                if checksum >= 3315:
                    den_image[i][j]=den_image[i][j+1]=den_image[i][j+2]=den_image[i][j+3]= \
                    den_image[i+1][j]=den_image[i+1][j+1]=den_image[i+1][j+2]=den_image[i+1][j+3]= \
                    den_image[i+2][j]=den_image[i+2][j+1]=den_image[i+2][j+2]=den_image[i+2][j+3]= \
                    den_image[i+3][j]=den_image[i+3][j+1]=den_image[i+3][j+2]=den_image[i+3][j+3]=255
    elif tray_size == 5:
        for i in range(nr-5):
            for j in range(nc-5):
                dtemp = [image[i][j],image[i][j+1],image[i][j+2],image[i][j+3],image[i][j+4],
                     image[i+1][j],image[i+1][j+1],image[i+1][j+2],image[i+1][j+3],image[i+1][j+4],
                     image[i+2][j],image[i+2][j+1],image[i+2][j+2],image[i+2][j+3],image[i+2][j+4],
                     image[i+3][j],image[i+3][j+1],image[i+3][j+2],image[i+3][j+3],image[i+3][j+4],
                     image[i+4][j],image[i+4][j+1],image[i+4][j+2],image[i+4][j+3],image[i+4][j+4]]
                checksum = sum(dtemp)
                if checksum >= 5355:
                    den_image[i][j]=den_image[i][j+1]=den_image[i][j+2]=den_image[i][j+3]=den_image[i][j+4]= \
                    den_image[i+1][j]=den_image[i+1][j+1]=den_image[i+1][j+2]=den_image[i+2][j+3]=den_image[i+1][j+4]= \
                    den_image[i+2][j]=den_image[i+2][j+1]=den_image[i+2][j+2]=den_image[i+2][j+3]=den_image[i+2][j+4]= \
                    den_image[i+3][j]=den_image[i+3][j+1]=den_image[i+3][j+2]=den_image[i+3][j+3]=den_image[i+3][j+4]= \
                    den_image[i+4][j]=den_image[i+4][j+1]=den_image[i+4][j+2]=den_image[i+4][j+3]=den_image[i+4][j+4]=255
    else:
        for i in range(nr-6):
            for j in range(nc-6):
                dtemp = [image[i][j],image[i][j+1],image[i][j+2],image[i][j+3],image[i][j+4],image[i][j+5],
                     image[i+1][j],image[i+1][j+1],image[i+1][j+2],image[i+1][j+3],image[i+1][j+4],image[i+1][j+5],
                     image[i+2][j],image[i+2][j+1],image[i+2][j+2],image[i+2][j+3],image[i+2][j+4],image[i+2][j+5],
                     image[i+3][j],image[i+3][j+1],image[i+3][j+2],image[i+3][j+3],image[i+3][j+4],image[i+3][j+5],
                     image[i+4][j],image[i+4][j+1],image[i+4][j+2],image[i+4][j+3],image[i+4][j+4],image[i+4][j+5],
                     image[i+5][j],image[i+4][j+1],image[i+5][j+2],image[i+5][j+3],image[i+5][j+4],image[i+5][j+5]]
                checksum = sum(dtemp)
                if checksum >= 5355:
                    den_image[i][j]=den_image[i][j+1]=den_image[i][j+2]=den_image[i][j+3]=den_image[i][j+4]=den_image[i][j+5]= \
                    den_image[i+1][j]=den_image[i+1][j+1]=den_image[i+1][j+2]=den_image[i+2][j+3]=den_image[i+1][j+4]=den_image[i+1][j+5]= \
                    den_image[i+2][j]=den_image[i+2][j+1]=den_image[i+2][j+2]=den_image[i+2][j+3]=den_image[i+2][j+4]=den_image[i+2][j+5]= \
                    den_image[i+3][j]=den_image[i+3][j+1]=den_image[i+3][j+2]=den_image[i+3][j+3]=den_image[i+3][j+4]=den_image[i+3][j+5]= \
                    den_image[i+4][j]=den_image[i+4][j+1]=den_image[i+4][j+2]=den_image[i+4][j+3]=den_image[i+4][j+4]=den_image[i+4][j+5]= \
                    den_image[i+5][j]=den_image[i+4][j+1]=den_image[i+5][j+2]=den_image[i+5][j+3]=den_image[i+5][j+4]=den_image[i+5][j+5]=255
    return den_image


image = cv2.imread("D:/Image processing/train/BE (8).jpg",1)
## 256x2  = 512
if image.shape[1]>512:
    ratio_fact = 512/image.shape[1]
else:
    ratio_fact = 1
image = cv2.resize(image ,None, fx=ratio_fact, fy=ratio_fact)


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(nr, nc) = gray_image.shape
th = thresholdvalue(gray_image) 
variance = gray_image.var()
average = gray_image.mean()
P = variance/average#normally > 10
D = (average/th)*100#normall > 80
A = (average/variance)*100 #normally < 10
B = ((th-average)/th)*10#opposit to D , normally < 20 
Q = D*A # normally < 300
F = ((variance-average)/variance)*100#normally >90
image_out = np.zeros((nr,nc),dtype=np.uint8)


b = math.ceil(B)
if b <= 1:
    b = 2
kernel = rectangle_kernel(math.ceil(b))
        

print("smoothing and filtering...")
if (P/10) != 0:
    p = math.ceil(P/10)
    smooth_image = cv2.blur(gray_image, (p, p))
else:
    smooth_image = gray_image
if A < 10:
    a = math.floor(A)
    analog_image = gauss_hpf(smooth_image, radius = a )
    analog_image = np.uint8(analog_image)
else:
    analog_image = smooth_image
    

print("gamma correction...")
if B>0 and B<10:
    gamma = 1 + (math.floor(B)/10)
    gamma_corrected = np.array(255*(analog_image / 255) ** gamma, dtype = 'uint8')
else:
    gamma_corrected = analog_image
thres_index, thres_gamma = cv2.threshold(gamma_corrected,th,255,cv2.THRESH_BINARY)

if (B-A) < 0:
    thres_gamma = cv2.dilate(thres_gamma, kernel)
else:
    thres_gamma = cv2.erode(thres_gamma, kernel)
    
    
print("and_res...")
ret, thres_new = cv2.threshold(smooth_image, th, 255, cv2.THRESH_BINARY)
AND_result = cv2.bitwise_and(thres_new, thres_gamma, mask=None)
read_image = median_filter(AND_result, nr, nc)
S = read_image.var()
R = read_image.mean()
G = S - variance
print("  R = ",R)
print("  G =",G)


print("dense...")
if G < 0 and R < 2:
    dense_image = digital_image = thres_gamma
elif G < 1000 and R < 3:
    thres_index, thres_fil = cv2.threshold(analog_image,th,255,cv2.THRESH_BINARY)
    dense_image = digital_image = thres_fil
elif R < 3:
    print("\nNo defect or bad Image")
    print("EXIT\n")
    sys.exit(1)
elif R > 2 and R < 8:
    dense_image = digital_image = read_image
else:
    if G < 0 or G > 5000:
        digital_image = min_filter(read_image, nr, nc)
    else :
        digital_image = read_image
        
H = digital_image.var()
print("v = ",digital_image.var())
if digital_image.var() > 2000:
    a = int(H/1000)
    dense_image = density_counter(digital_image, a)
else:
    dense_image = digital_image

# image_out = mean_filter(dense_image, nr, nc)

# =============================================================================
# inv_ratio = 1/ratio_fact
# image_out = cv2.resize(dense_image ,None, fx=inv_ratio, fy=inv_ratio)
# cv2.imwrite("RM_38_new.jpg",image_out)
# =============================================================================
titles = ['Original Image', 'Analog Image', 'Gamma corrected', 'AND result', 'Digital Image' ,'Dense Image']
images = [image, analog_image, gamma_corrected, AND_result, digital_image, dense_image]
for k in range(6):
    plt.subplot(2, 3, k + 1)
    plt.imshow(images[k], cmap='gray')
    plt.title(titles[k])
    plt.xticks([])
    plt.yticks([])
plt.show()

res = np.concatenate((digital_image, dense_image), axis =1)
cv2.imshow("densy Image", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# inpath = "D:/Image processing/train/RM/*"
# outpath = "D:/Image processing/result/" 
# 
# 
# for file in glob.glob(inpath):
#     print(file)
#     image = cv2.imread(file,1)
#     ## 256x2  = 512
#     if image.shape[1]>512:
#         ratio_fact = 512/image.shape[1]
#     else:
#         ratio_fact = 1
#     image = cv2.resize(image ,None, fx=ratio_fact, fy=ratio_fact)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     (nr, nc) = gray_image.shape
#     th = thresholdvalue(gray_image) 
#     variance = gray_image.var()
#     average = gray_image.mean()
#     
#     P = variance/average#normally > 10
#     D = (average/th)*100#normall > 80
#     A = (average/variance)*100 #normally < 10
#     B = ((th-average)/th)*10#opposit to D , normally < 20 
#     Q = D*A # normally < 300
#     F = ((variance-average)/variance)*100#normally >90
#     image_out = np.zeros((nr,nc),dtype=np.uint8)
#     
#     
#     b = math.ceil(B)
#     if b <= 1:
#         b = 2
#     kernel = rectangle_kernel(math.ceil(b))
#             
#     print("smoothing and filtering...")
#     if (P/10) != 0:
#         p = math.ceil(P/10)
#         smooth_image = cv2.blur(gray_image, (p, p))
#     else:
#         smooth_image = gray_image
#     if A < 10:
#         a = math.floor(A)
#         analog_image = gauss_hpf(smooth_image, radius = a )
#         analog_image = np.uint8(analog_image)
#     else:
#         analog_image = smooth_image
#     
#     print("gamma correction...")
#     if B>0 and B<10:
#         gamma = 1 + (math.floor(B)/10)
#         gamma_corrected = np.array(255*(analog_image / 255) ** gamma, dtype = 'uint8')
#     else:
#         gamma_corrected = analog_image
#     thres_index, thres_gamma = cv2.threshold(gamma_corrected,th,255,cv2.THRESH_BINARY)
#     
#     if (B-A) < 0:
#         thres_gamma = cv2.dilate(thres_gamma, kernel)
#     else:
#         thres_gamma = cv2.erode(thres_gamma, kernel)
#     print("and_res...")
#     ret, thres_new = cv2.threshold(smooth_image, th, 255, cv2.THRESH_BINARY)
#     AND_result = cv2.bitwise_and(thres_new, thres_gamma, mask=None)
#     read_image = median_filter(AND_result, nr, nc)
#     S = read_image.var()
#     R = read_image.mean()
#     G = S - variance
#     print("  R = ",R)
#     print("  G =",G)
#     print("dense...")
#     if G < 0 and R < 2:
#         dense_image = digital_image = thres_gamma
#     elif G < 1000 and R < 3:
#         thres_index, thres_fil = cv2.threshold(analog_image,th,255,cv2.THRESH_BINARY)
#         dense_image = digital_image = thres_fil
#     elif R < 3:
#         print("\nNo defect or bad Image")
#         print("EXIT\n")
#         sys.exit(1)
#     elif R > 2 and R < 8:
#         dense_image = digital_image = read_image
#     else:
#         if G < 0 or G > 5000:
#             digital_image = min_filter(read_image, nr, nc)
#         else :
#             digital_image = read_image
#     H = digital_image.var()
#     print("v = ",digital_image.var())
#     if digital_image.var() > 2000:
#         a = int(H/1000)
#         dense_image = density_counter(digital_image, a)
#     else:
#         dense_image = digital_image
#     
#     # image_out = mean_filter(dense_image, nr, nc)
#     
#     # =============================================================================
#     # inv_ratio = 1/ratio_fact
#     # image_out = cv2.resize(dense_image ,None, fx=inv_ratio, fy=inv_ratio)
#     # cv2.imwrite("RM_38_new.jpg",image_out)
#     # =============================================================================
#     titles = ['Original Image', 'Analog Image', 'Gamma corrected', 'AND result', 'Digital Image' ,'Dense Image']
#     images = [image, analog_image, gamma_corrected, AND_result, digital_image, dense_image]
#     for k in range(6):
#         plt.subplot(2, 3, k + 1)
#         plt.imshow(images[k], cmap='gray')
#         plt.title(titles[k])
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()
# 
# =============================================================================

