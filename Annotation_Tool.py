
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib
import os
import skimage 
from skimage import color
from skimage.color import rgb2grey


# In[2]:


class paint():
    def __init__(self,img_name,r=[],cor_r=[],drawing = False,mode=True,auto_imsave_rec=[]):
        self.img_name = img_name
        self.img = cv2.imread(self.img_name,1)
        self.r=r
        self.drawing = drawing
        self.mode=mode
        self.auto_imsave_rec = auto_imsave_rec
        self.cor_r = cor_r
        
    def load_img(self):
        img = cv2.imread(self.img_name,1)
        return img
    
    def select_patch(self,event,x,y,flags,param):
        global qx, qy
        if event == cv2.EVENT_LBUTTONDBLCLK:
            qx,qy = x,y
            self.r.append([x,y])
            pt1 = (int(x - 128),int(y - 128))
            pt2 = ((pt1[0] + 256),(pt1[1]+256))
            cv2.rectangle(img_,(pt1[0]-10,pt1[1]-10),(pt2[0]+10,pt2[1]+10),(255,0,0),10,1)
            patch = img_[pt1[1]:pt2[1],pt1[0]:pt2[0],:]
            cv2.namedWindow('patch image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('patch image', 512,512)
            cv2.imshow('patch image',patch)
            
    def save_patch(self):
        #save selected patch
        final_x,final_y = self.r[-1]
        pt1 = (int(final_x - 128),int(final_y - 128))
        pt2 = ((pt1[0] + 256),(pt1[1]+256))
        final_patch = img_[pt1[1]:pt2[1],pt1[0]:pt2[0],:]
        name_string = 'selected_patch_for_%s_at_%d_%d.png' % (self.img_name,final_x,final_y)
        cv2.imwrite(name_string,final_patch)
        self.final_patch = final_patch
        print ('select pacth at location (%d,%d)'%(final_x,final_y))
        
    def cell_brush(self,event,x,y,flags,param):
        #https://github.com/ashwin-pajankar/Python-OpenCV3/blob/master/01%23%20Basics/prog13.py
        global ix, iy
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            (ix, iy) = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                if self.mode == True:
                    cv2.rectangle(pimg, (ix, iy), (x, y), [0, 0, 0], -1)
                else:
                    cv2.circle(pimg, (x, y), 1, [0, 0, 0], -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == True:
                cv2.rectangle(pimg, (ix, iy), (x, y), [0, 0, 0], -1)
            else:
                cv2.circle(pimg, (x, y), 1, [0, 0, 0], -1)
                
    def paint_cell(self):
        global pimg,mode
        final_x,final_y = self.r[-1]
        pt1 = (int(final_x - 128),int(final_y - 128))
        pt2 = ((pt1[0] + 256),(pt1[1]+256))
        pimg =  img_[pt1[1]:pt2[1],pt1[0]:pt2[0],:]
        cv2.namedWindow('paint cell', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('paint cell', 512,512)
        cv2.setMouseCallback('paint cell',self.cell_brush)
        while(1):
            cv2.imshow('paint cell',pimg)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('m') or k == ord('M'):
                self.mode = not self.mode
            elif k == 27:
                break
        cv2.destroyAllWindows()
        self.anotated_cell = pimg
        
    def save_paint_cell(self):
        final_x,final_y = self.r[-1]
        empty_canvas = np.empty((276,552,3))
        empty_canvas[:,:] = [130,130,130]
        grey_img = cv2.cvtColor(self.anotated_cell, cv2.COLOR_BGR2GRAY)
        g = (grey_img!=0).astype(np.int)
        g = g*255
        cv2.imwrite('anotated_patch_for_%s.png'%self.img_name,g)
        rgb_g = skimage.color.grey2rgb(g)
        bgr_g = rgb_g[...,::-1]
        empty_canvas[10:266,10:266,:] = self.final_patch
        empty_canvas[10:266,286:542,:] = bgr_g
        empty_canvas = np.array(empty_canvas,dtype = np.uint8)#as.astype(np.int)
        s = 'compare_for_%s_at_%d_%d.png'% (self.img_name,final_x,final_y)
        cv2.imwrite(s,empty_canvas)
        cv2.namedWindow('The original patch (left) and the anotated patch (right)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('The original patch (left) and the anotated patch (right)', 552,276)
        cv2.imshow('The original patch (left) and the anotated patch (right)',empty_canvas)
        cv2.waitKey(0)
        cv2.destroyWindow('The original patch (left) and the anotated patch (right)')
        self.g = np.array(g,dtype=np.uint8)
        
    def draw_contours_center(self):
        final_x,final_y = self.r[-1]
        white_padded_gp = np.pad(self.g, ((5,5),(5,5)), 'constant', constant_values=((255,255),(255,255)))
        _,contours,_ = cv2.findContours(white_padded_gp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        rgb_gp = skimage.color.grey2rgb(white_padded_gp)
        p = rgb_gp[...,::-1]
        p = np.array(p,dtype=np.uint8)
        for i in range(1,len(contours)):
            maxi = np.amax(contours[i].reshape(-1,2),axis=0)
            mini = np.amin(contours[i].reshape(-1,2),axis=0)
            x,y = np.round((maxi + mini)/2).astype(np.int)#,dtype=np.int)
            cv2.circle(p, (x, y), 2, [0, 0, 255], -1)
            cv2.drawContours(p, contours, i, (0,0,255), 1)       
        cv2.namedWindow('display contours and centers of nucleus', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('display contours and centers of nucleus', 512,512)
        cv2.imshow('display contours and centers of nucleus',p[5:256,5:256,:])
        cv2.waitKey(0)
        cv2.destroyWindow('display contours and centers of nucleus')
        s = 'contours_for_%s_at_%d_%d.png'% (self.img_name,final_x,final_y)
        cv2.imwrite(s,p[5:256,5:256,:])
    
    def patch_show(self):
        global img_
        img_ = self.load_img()
        cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
        img_shape_x,img_shape_y,_ = img_.shape
        resize_window_shape = (int(img_shape_x/10),int(img_shape_y/10))
        cv2.resizeWindow('original image', resize_window_shape[1],resize_window_shape[0])
        cv2.setMouseCallback('original image',self.select_patch)
        while(1):
            cv2.imshow('original image',img_)
            k = cv2.waitKey(200) & 0xFF
            if k == 13:
                break
            elif k == ord('p') or k == ord('P'):
                print (qy,qx)
        cv2.destroyAllWindows()


# In[ ]:


p = paint('He2_1_2_3.tiff')
p.patch_show()
p.save_patch()
p.paint_cell()
p.save_paint_cell()
p.draw_contours_center()


# In[ ]:


p = paint('He2_1_3_5.tiff')
p.patch_show()
p.save_patch()
p.paint_cell()
p.save_paint_cell()
p.draw_contours_center()


# In[ ]:


p = paint('He4_1_3_1.tiff')
p.patch_show()
p.save_patch()
p.paint_cell()
p.save_paint_cell()
p.draw_contours_center()


# In[ ]:


p = paint('He4_1_3_4.tiff')
p.patch_show()
p.save_patch()
p.paint_cell()
p.save_paint_cell()
p.draw_contours_center()


# In[ ]:


p = paint('He5_1_1_3.tiff')
p.patch_show()
p.save_patch()
p.paint_cell()
p.save_paint_cell()
p.draw_contours_center()
#select pacth at location (5910,2790)


# In[ ]:


p = paint('He5_1_3_4.tiff')
p.patch_show()
p.save_patch()
p.paint_cell()
p.save_paint_cell()
p.draw_contours_center()
#select pacth at location (3020,2252)


# In[ ]:


p = paint('He6_1_3_3.tiff')
p.patch_show()
p.save_patch()
p.paint_cell()
p.save_paint_cell()
p.draw_contours_center()
#select pacth at location (3830,4240)

