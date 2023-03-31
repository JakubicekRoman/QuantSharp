# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:49:27 2023

@author: jakubicek
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from skimage import draw
from scipy.signal import convolve2d, medfilt2d
from numpy.fft import fft2, fftshift, ifft2

import SimpleITK as sitk
import pydicom as pydic

# import napari
import warnings
# with warnings.catch_warnings():
warnings.filterwarnings("ignore", category=RuntimeWarning)
sitk.ProcessObject_SetGlobalWarningDisplay(False)

def crop_center(img, crop):
    if len(img.shape) == 3:
        x, y, z = img.shape
        (cropx, cropy, cropz) = crop
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        startz = z//2-(cropz//2)
        img = img[startx:startx+cropx,
                  starty:starty+cropy, startz:startz+cropz]
    elif len(img.shape) == 2:
        x, y = img.shape
        (cropx, cropy) = crop
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        img = img[startx:startx+cropx, starty:starty+cropy]
    return img


def crop_image(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img > tol
    ind = np.ix_(mask.any(1), mask.any(0));
    return img[np.ix_(mask.any(1), mask.any(0))], ind


def gen_interCircle(vel, rad):
    arr = np.zeros(vel)
    for i in range(rad[0], rad[1]):
        rr, cc = draw.circle_perimeter(
            vel[0]//2, vel[1]//2, radius=i, shape=arr.shape)
        arr[rr, cc] = 1

    arr = medfilt2d(arr)
    return arr


def normalize(img):
    img = img.astype(float)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def find_subdir(dirName):
    listPat = os.listdir(dirName)
    allFiles = list()
    for pat in listPat:
        pat_path = os.path.join(dirName, pat)
        if os.path.isdir(pat_path):
            listSer = os.listdir(pat_path)
            for ser in listSer:
                ser_path = os.path.join(dirName, pat, ser)
                if os.path.isdir(ser_path):
                    allFiles.append(ser_path)
    return allFiles

def nested_subfiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + nested_subfiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def QuantSharp(path_data, path_save, name_file='Results'):
        
# path_data = 'D:\\Projekty\\Prostate_MRI\\WIP_DecRec_Quality\\Data\\dirVFN'
# path_save = 'D:\\Projekty\\Prostate_MRI\\WIP_DecRec_Quality\\results'
# # path_save = os.path.normpath(path_save)
# name_file = 'results'

    
    if os.path.exists(path_data):
        print('Browsing a data folder ... ')
    else:
        print('Error: Path does not exist! Wrong Data path')
        # sys.exit()
        return
    
    if not os.path.exists( path_save ):
        print('Error: Path for saving results does not exist! Wrong Save path')
        # sys.exit()
        return
    
    
    df = pd.DataFrame()
    
    # data_list = glob.glob(path_data+'\\**\\I20', recursive=True)
    
    # data_list = glob.glob(path_data+'\\**', recursive=True)
    
    # data_list = nested_subfiles(path_data)
    data_list = find_subdir(path_data)
    
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_data + '\\' + 'S44670' + '\\S4010')
    
    if not data_list:
        print('Error: Folder of data is empty!')
        # sys.exit()
        return
    
    list_data = []
    k = 0
    for fileName in data_list:
    
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(fileName+'\\')
        
        if series_IDs:
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                fileName, series_IDs[0])
        
            ds = pydic.dcmread(series_file_names[0])
    
            df.loc[k, 'Patient Name'] = str(ds[(0x0010, 0x0010)].value)
            df.loc[k, 'Birth Date'] = ds[(0x0010, 0x0030)].value
            df.loc[k, 'Patient ID'] = ds[(0x0010, 0x0020)].value
    
            try:
                df.loc[k, 'Study date'] = ds[(0x0008, 0x0022)].value
                df.loc[k, 'Study time'] = ds[(0x0008, 0x0030)].value
                df.loc[k, 'Series time'] = ds[(0x0008, 0x0031)].value
                df.loc[k, 'Access no'] = ds[(0x0008, 0x0050)].value
        
                df.loc[k, 'instance number'] = ds[(0x0020, 0x0013)].value
                df.loc[k, 'pixel spacing'] = ds[(0x2001, 0x107b)].value
                df.loc[k, 'slice orientation'] = ds[(0x2001, 0x100b)].value
        
                df.loc[k, 'TM STUDY_TIME'] = ds[(0x0008, 0x0030)].value
                df.loc[k, 'TM SERIES_TIME'] = ds[(0x0008, 0x0031)].value
                df.loc[k, 'TM ACQUISITION_TIME '] = ds[(0x0008, 0x0032)].value
                df.loc[k, 'SH ACCESSION_NUMBER '] = ds[(0x0008, 0x0050)].value
        
                df.loc[k, 'acq duration'] = ds[(0x0018, 0x9073)].value
                df.loc[k, 'LO PROTOCOL_NAME'] = ds[(0x0018, 0x1030)].value
                df.loc[k, 'US ACQUISITION_MATRIX'] = str(ds[(0x0018, 0x1310)].value)
        
                df.loc[k, 'IS SERIES_NUMBER'] = ds[(0x0020, 0x0011)].value
                df.loc[k, 'IS INSTANCE_NUMBER'] = ds[(0x0020, 0x0013)].value
        
                df.loc[k, 'US ROWS '] = ds[(0x0028, 0x0010)].value
                df.loc[k, 'US COLUMNS'] = ds[(0x0028, 0x0011)].value
                df.loc[k, 'DS PIXEL_SPACING'] = str(ds[(0x0028, 0x0030)].value)
        
                df.loc[k, 'DS WINDOW_CENTER'] = ds[(0x0028, 0x1050)].value
                df.loc[k, 'DS WINDOW_WIDTH'] = ds[(0x0028, 0x1051)].value
                df.loc[k, 'DS RESCALE_INTERCEPT'] = ds[(0x0028, 0x1052)].value
                df.loc[k, 'DS RESCALE_SLOPE '] = ds[(0x0028, 0x1053)].value
        
                df.loc[k, 'CS PIIM_MR_SERIES_DEVELOPMENT_MODE'] = ds[(0x2005, 0x1013)].value            
            except:    
                print('Warning: some tag from Dicom could not be read!')
                
            list_data.append(fileName)
            k = k+1
    
    if not list_data:
        print('Error: Folder contains no scans!')
        return
        # sys.exit()
    
    num_pat =  len(df["Access no"].unique())
    
    print('There were found ' + str(len(list_data)) + ' scans containing ' + str(num_pat) + ' patients')
    
    k = 0
    for dirnName in list_data:
        print('Proccessing ' + str(k+1) + '. scan from ' + str(len(list_data)) + ' ...')
    
    
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dirnName+'\\')
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            dirnName, series_IDs[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        series_reader.LoadPrivateTagsOn()
        sitk_image = series_reader.Execute()
        sizeImg = sitk_image.GetSize()
        Spacing = sitk_image.GetSpacing()
        sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
        Data = sitk.GetArrayFromImage(sitk_image)
        
        # Data = Data[:,1:-1,1:-1]
    
        crop_prctg = 0.50
        img_size = [Data.shape[0]]
        img2, indCut = crop_image(np.sum(Data,axis=0), tol=0)
        b = list(np.shape(img2))
        # b = list(np.shape(Data[0, :, :]))
        img_size.extend(b)
    
        Grad = np.zeros((img_size[0], img_size[1]-2, img_size[2]-2))
        Grad2 = np.zeros((int(img_size[0]), int(
            img_size[1]*crop_prctg), int(img_size[2]*crop_prctg)))
        Grad3 = np.zeros((int(img_size[0]), int(
            img_size[1]*crop_prctg), int(img_size[2]*crop_prctg)))
    
        Spekt1 = np.zeros((img_size[0], img_size[1], img_size[2]))
        Spekt2 = np.zeros((int(img_size[0]), int(
            img_size[1]*crop_prctg), int(img_size[2]*crop_prctg)))
        Spekt1_masked = np.zeros((img_size[0], img_size[1], img_size[2]))
        Spekt2_masked = np.zeros((int(img_size[0]), int(
            img_size[1]*crop_prctg), int(img_size[2]*crop_prctg)))
    
        # threshold_G = 500
        # threshold_L = 80
        
        for id_slice in range(0, Data.shape[0]):
            img = Data[id_slice, :, :]
    
            # img = crop_image(img, tol=0)
            img = img[indCut]
    
            grad_mask1 = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])/Spacing[0]/2
            grad_mask2 = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])/Spacing[0]/2
            grad_mask3 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])/Spacing[0]/2
            grad_mask4 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])/Spacing[0]/2
    
            grad_1 = convolve2d(img, grad_mask1, mode='valid')
            grad_2 = convolve2d(img, grad_mask2, mode='valid')
            grad_3 = convolve2d(img, grad_mask3, mode='valid')
            grad_4 = convolve2d(img, grad_mask4, mode='valid')
    
            grad = np.sqrt(grad_1**2 + grad_2**2 + grad_3**2 + grad_4**2)
    
            grad2 = crop_center(
                grad, (int(img_size[1]*crop_prctg), int(img_size[2]*crop_prctg)))
    
            threshold_G = np.max(grad2)*0.1
            
            grad3 = np.zeros([grad2.shape[0], grad2.shape[1]])
            grad3[grad2 > threshold_G] = 1
    
            Grad[id_slice, :, :] = grad
            Grad2[id_slice, :, :] = grad2
            Grad3[id_slice, :, :] = grad3
    
            img = Data[id_slice, :, :]
            # img = crop_image(img, tol=0)
            img = img[indCut]
    
            radius = [20, 80]
            mask = gen_interCircle(img.shape, np.round(radius).astype(int))
    
            window1d_1 = np.abs(np.hamming(img.shape[0]))
            window1d_2 = np.abs(np.hamming(img.shape[1]))
            window2d = np.sqrt(np.outer(window1d_1, window1d_2))
    
            spekt_1 = (abs(fftshift(fft2(img*window2d, img.shape)))
                       )**2 / np.size(img)
            Spekt1_masked[id_slice, :, :] = spekt_1*mask
            Spekt1[id_slice, :, :] = spekt_1
    
            img = Data[id_slice, :, :]
            # img = crop_image(img, tol=0)
            img = img[indCut]
            img = crop_center(
                img, (int(img_size[1]*crop_prctg), int(img_size[2]*crop_prctg)))
    
            mask = gen_interCircle(img.shape, np.round(radius).astype(int))
    
            window1d_1 = np.abs(np.hamming(img.shape[0]))
            window1d_2 = np.abs(np.hamming(img.shape[1]))
            window2d = np.sqrt(np.outer(window1d_1, window1d_2))
    
            spekt_2 = (abs(fftshift(fft2(img*window2d, img.shape)))
                       )**2 / np.size(img)
    
            Spekt2_masked[id_slice, :, :] = spekt_2*mask
            Spekt2[id_slice, :, :] = spekt_2
            
            # plt.figure()
            # plt.imshow(grad3)
            # plt.show()
    
        df.loc[k, 'Sharpness value #1'] = np.mean(Grad)
        df.loc[k, 'Sharpness value #2'] = np.mean(Grad2)
        df.loc[k, 'Sharpness value #3'] = np.mean(Grad2[Grad3 > 0])
        df.loc[k, 'Sharpness value #4'] = np.mean(
            Spekt1_masked[Spekt1_masked > 0]) / np.mean(Spekt1) * 100
        df.loc[k, 'Sharpness value #5'] = np.mean(
            Spekt2_masked[Spekt2_masked > 0]) / np.mean(Spekt2) * 100
    
        Grad4 = Grad2.copy()
        Grad4[Grad3 == 0] = np.nan
    
        Spekt1_masked[Spekt1_masked == 0] = np.nan
        Spekt2_masked[Spekt2_masked == 0] = np.nan
    
        df.loc[k, 'Sh #1 slices'] = str(np.mean(np.mean(Grad, axis=1), axis=1)).replace(
            '\n', '').replace('  ', ' ')[1:-2]
        df.loc[k, 'Sh #2 slices'] = str(np.mean(np.mean(Grad2, axis=1), axis=1)).replace(
            '\n', '').replace('  ', ' ')[1:-2]
        df.loc[k, 'Sh #3 slices'] = str(np.nanmean(np.nanmean(
            Grad4, axis=1), axis=1)).replace('\n', '').replace('  ', ' ')[1:-2]
        df.loc[k, 'Sh #4 slices'] = str(np.nanmean(np.nanmean(
            Spekt1_masked, axis=1), axis=1)).replace('\n', '').replace('  ', ' ')[1:-2]
        df.loc[k, 'Sh #5 slices'] = str(np.nanmean(np.nanmean(
            Spekt2_masked, axis=1), axis=1)).replace('\n', '').replace('  ', ' ')[1:-2]
        
        k = k+1
    
    print('Saving excel document with results ...')
    
    path_save = path_save.replace('.xlsx','')
    df.to_excel(path_save + os.sep + name_file + '.xlsx', index=False)
    
    print('Saved ...')
    print('Program finished ...')
    
    
if __name__ == "__main__":
        
    path_data = 'D:\\Projekty\\Prostate_MRI\\WIP_DecRec_Quality\\Data\\dirVFN'
    # path_data = r'C:\Data\MRI_Glioms\MRI_Brain\test'
    path_save = 'D:\\Projekty\\Prostate_MRI\\WIP_DecRec_Quality\\results'
    name_file = 'results'
    
    QuantSharp(path_data, path_save, name_file='Results')
