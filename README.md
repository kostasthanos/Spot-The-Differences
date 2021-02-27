# Spot-The-Differences-OpenCV
Spot the differences between two images using Python and OpenCV
<p align="center">
<img width="400" height="300" src="Images/spot_logo.png ">
</p>

## Main Idea
This program can spot-find the differences between two images.  
The user loads to the program 2 images that are mostly the same but also have some small differences. By running the program user gets both images side by side highlighting the differences.
 
## First Method
### Using *cv2.absdiff*
We are going to use images *city1.jpg* and *city2.jpg* for a better understanding.
First we are loading the two images
```python
img1 = cv2.imread('path_to_image_1')
img2 = cv2.imread('path_to_image_2')
```
Then convert both images to grayscale format
```python
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
```
Now it's time to find the absolute difference between the two images (arrays)
```python
diff = cv2.absdiff(gray1, gray2)
cv2.imshow("diff(img1, img2)", diff)
```
Apply threshold. Apply both THRESH_BINARY and THRESH_OTSU
```python
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Threshold", thresh)
```
We are going to use 2 iterations of dilation in order to increase the white region in the image.
```python
kernel = np.ones((5,5), np.uint8) 
dilate = cv2.dilate(thresh, kernel, iterations=2) 
cv2.imshow("Dilate", dilate)
```
Finally we are calculating the contours and draw rectangle in both images which are corresponding to the differences between the 2 images.
```python
contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
```

## Second Method
### Using *compare_ssim*
We are following the steps of the First Method we some small changes. Instead of **cv2.absdiff**, now we are computing the full structural similarity (*similar*) between the two gray images. Also we must convert *diff* array in range [0, 255]
```python
(simalr, diff) = compare_ssim(gray1, gray2, full=True)
diff = (diff*255).astype("uint8")
```
Now we should delte the dilation part from first method and continue with the contours calculation as before.


## Author
* **Konstantinos Thanos**
