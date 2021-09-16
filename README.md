# jax-perspective-transform
Image perspective transform hastily ported from Kornia (Pytorch) to JAX/Numpy. Could be buggy. Most of the docstrings and comments in the code should be mostly ignored because they're based on the old torch code. The relevant function is `warp_perspective`. [It's docstring](https://github.com/josephrocca/jax-perspective-transform/blob/main/module.py#L8) should be correct regarding the available options/parameters.

## Example
```python
# Example based on: https://kornia-tutorials.readthedocs.io/en/latest/warp_perspective.html
import PIL.Image as Image

# the source points are the region to crop corners ([x0,y0], [x1,y1], etc. clockwise starting from top left)
points_src = np.array([
    [125., 150.], [562., 40.], [562., 282.], [54., 328.], # corners of bruce lee poster
])

# the destination points are the image vertexes ([x0,y0], [x1,y1], etc. clockwise starting from top left)
dst_h, dst_w = 64, 128
points_dst = np.array([
    [0., 0.], [dst_w - 1., 0.], [dst_w - 1., dst_h - 1.], [0., dst_h - 1.],
])

img = np.array(Image.open('bruce.png').convert("RGB"))  
img = img.transpose(2, 0, 1) # CxHxW / np.uint8
print(img.shape)

# compute perspective transform
M = get_perspective_transform(points_src, points_dst)

# warp the original image by the found transform
img_warped = warp_perspective(img.astype('float32'), M, dsize=(dst_h, dst_w))
print(img_warped.shape)

# convert back to HxWxC
img = img.transpose(1, 2, 0)
img_warped = img_warped.transpose(1, 2, 0).astype('uint8')

Image.fromarray(onp.array(img)).show()
Image.fromarray(onp.array(img_warped)).show()
```
![image](https://user-images.githubusercontent.com/1167575/133641001-3ed600ef-cc41-4762-b3f3-80c4feedd51c.png)


## Random Transform Example
```python
import time
import PIL.Image as Image
import numpy as onp

def r(a=0, b=1): # generate a random float between `a` and `b`
    key = jax.random.PRNGKey(int(time.time()*1000)) # just for demo!
    return a + jax.random.uniform(key) * (b - a)

img = np.array(Image.open('happy_dog.png').convert("RGB"))  
img = img.transpose(2, 0, 1) # CxHxW / np.uint8
print(img.shape)

# the source points are the region to crop corners ([x0,y0], [x1,y1], etc. clockwise starting from top left)
in_w, in_h = (img.shape[2], img.shape[1])
points_src = np.array([
    [0, 0], [in_w, 0], [in_w, in_h], [0, in_h],
])

# the destination points are the image vertexes ([x0,y0], [x1,y1], etc. clockwise starting from top left)
dst_w, dst_h = (img.shape[2], img.shape[1])
scale = 0.1 # <-- move corners by 10% of image width/height
points_dst = np.array([
    [0+r(-scale*dst_w, scale*dst_w), 0+r(-scale*dst_h, scale*dst_h)],
    [dst_w+r(-scale*dst_w, scale*dst_w), 0+r(-scale*dst_h, scale*dst_h)],
    [dst_w+r(-scale*dst_w, scale*dst_w), dst_h+r(-scale*dst_h, scale*dst_h)],
    [0+r(-scale*dst_w, scale*dst_w), dst_h+r(-scale*dst_h, scale*dst_h)],
])

# compute perspective transform
M = get_perspective_transform(points_src, points_dst)

# warp the original image by the found transform
img_warped = warp_perspective(img.astype('float32'), M, dsize=(dst_h, dst_w))
print(img_warped.shape)

# convert back to HxWxC
img = img.transpose(1, 2, 0)
img_warped = img_warped.transpose(1, 2, 0).astype('uint8')

Image.fromarray(onp.array(img)).show()
Image.fromarray(onp.array(img_warped)).show()
```
![image](https://user-images.githubusercontent.com/1167575/133641744-b43ccd45-db18-4e50-87f8-beda88898d25.png)

* Related: https://github.com/deepmind/dm_pix https://github.com/4rtemi5/imax
* Licenced under Apache 2.0 to match Kornia: https://github.com/kornia/kornia/blob/master/LICENSE
* Happy dog image by Noémi Macavei-Katócz: https://unsplash.com/photos/c7bUIRBqapA and the Bruce Lee poster image is from here: https://kornia-tutorials.readthedocs.io/en/latest/warp_perspective.html
