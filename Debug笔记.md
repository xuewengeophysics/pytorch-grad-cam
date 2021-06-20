# Debug笔记



## EfficientNet

### numpy.AxisError: axis 2 is out of bounds for array of dimension 2

```python
Traceback (most recent call last):
  File "cam_EfficientNet_2021-06-20.py", line 107, in <module>
    grayscale_cam = cam(input_tensor=input_tensor,
  File "D:\AI\xwStudy\xwGithub\visual_recognition\pytorch-grad-cam\pytorch_grad_cam\base_cam.py", line 128, in __call__
    return self.forward(input_tensor,
  File "D:\AI\xwStudy\xwGithub\visual_recognition\pytorch-grad-cam\pytorch_grad_cam\base_cam.py", line 76, in forward
    cam = self.get_cam_image(input_tensor, target_category,
  File "D:\AI\xwStudy\xwGithub\visual_recognition\pytorch-grad-cam\pytorch_grad_cam\base_cam.py", line 46, in get_cam_image
    weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
  File "D:\AI\xwStudy\xwGithub\visual_recognition\pytorch-grad-cam\pytorch_grad_cam\grad_cam.py", line 16, in get_cam_weights
    return np.mean(grads, axis=(2, 3))
  File "<__array_function__ internals>", line 5, in mean
  File "D:\Anaconda3\envs\yolov5py38\lib\site-packages\numpy\core\fromnumeric.py", line 3372, in mean
    return _methods._mean(a, axis=axis, dtype=dtype,
  File "D:\Anaconda3\envs\yolov5py38\lib\site-packages\numpy\core\_methods.py", line 147, in _mean
    rcount = _count_reduce_items(arr, axis)
  File "D:\Anaconda3\envs\yolov5py38\lib\site-packages\numpy\core\_methods.py", line 66, in _count_reduce_items
    items *= arr.shape[mu.normalize_axis_index(ax, arr.ndim)]
numpy.AxisError: axis 2 is out of bounds for array of dimension 2
```

+ [Applying GradCAM on EfficientNet networks](https://github.com/jacobgil/pytorch-grad-cam/issues/95)

```python
target_layer = model._conv_head
```

