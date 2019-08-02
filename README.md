
### Questions

### Objectives
YWBAT
- Download multiple pictures from google search images using terminal
    - ```pbpaste | awk '{print substr($0, 1, length($0)-1) ">>" NR ".jpg"}' | bash```
- Implement Mongodb training and testing generators

### High level overview
- Filters: A Matrix that gets convolved over images (3x3, 5x5, etc)
     - Purpose: To find patterns
 - CNNs: Use gradient descent to find filters
     - Usually the first filter (layer) of a CNN finds EDGES
     - 2nd (filter) usually starts finding shapes within EDGES
 - Max Pooling - Is taking the max value of a filter on a layer and turning those pixels into that value. 
 - Padding - Creates equal opportunity for each pixel

Directory structure from my home
```
|-batproject
    |-train
        |-batman
        |-bat
    |-test
        |-batman
        |-bat
    |-validation
        |-batman
        |-bat
```

### Outline


```python
import uuid
import requests
import glob

import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras import layers
from keras import Sequential
from keras import optimizers
from keras import models

import matplotlib.pyplot as plt
```


```python
# Use glob to get your files and do a count
# this is just for a sanity check
batman_train_images = glob.glob("/Users/rafael/batproject/train/batman/*.jpg")
batman_all_images = glob.glob("/Users/rafael/batproject/*/batman/*.jpg")
bat_train_images = glob.glob("/Users/rafael/batproject/train/*/*.jpg")
len(batman_train_images), len(batman_all_images), len(bat_train_images)
```




    (478, 576, 760)




```python
train_gen = ImageDataGenerator(rescale=1./225)
test_gen = ImageDataGenerator(rescale=1./225)
validation_gen = ImageDataGenerator(rescale=1./225)
```


```python
vars(train_gen)
```




    {'featurewise_center': False,
     'samplewise_center': False,
     'featurewise_std_normalization': False,
     'samplewise_std_normalization': False,
     'zca_whitening': False,
     'zca_epsilon': 1e-06,
     'rotation_range': 0,
     'width_shift_range': 0.0,
     'height_shift_range': 0.0,
     'shear_range': 0.0,
     'zoom_range': [1.0, 1.0],
     'channel_shift_range': 0.0,
     'fill_mode': 'nearest',
     'cval': 0.0,
     'horizontal_flip': False,
     'vertical_flip': False,
     'rescale': 0.0044444444444444444,
     'preprocessing_function': None,
     'dtype': 'float32',
     'interpolation_order': 1,
     'data_format': 'channels_last',
     'channel_axis': 3,
     'row_axis': 1,
     'col_axis': 2,
     '_validation_split': 0.0,
     'mean': None,
     'std': None,
     'principal_components': None,
     'brightness_range': None}




```python
train_gen.flow_from_directory(directory='/Users/rafael/batproject/train/', 
                              batch_size=32,
                              class_mode="categorical",
                              shuffle=True,
                              target_size=(150, 150),
                              seed=42)

validation_gen.flow_from_directory(directory='/Users/rafael/batproject/validation/', 
                                   batch_size=10,
                                   class_mode="categorical",
                                   shuffle=True,
                                   target_size=(150, 150),
                                   seed=42)

test_gen.flow_from_directory(directory='/Users/rafael/batproject/test/', 
                              batch_size=1,
                              class_mode="categorical",
                              shuffle=True,
                              target_size=(150, 150),
                              seed=42)



# This is an iterator object that will only pass in valid image files
def my_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass

```

    Found 760 images belonging to 2 classes.
    Found 32 images belonging to 2 classes.
    Found 164 images belonging to 2 classes.



```python
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```


```python
history = model.fit_generator(my_gen(train_gen),
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=my_gen(validation_gen), 
                              validation_steps=5,
                              verbose=1)
```

    Epoch 1/30



    ------------------------------------------------

    KeyboardInterruptTraceback (most recent call last)

    <ipython-input-42-c791fb156d13> in <module>
          4                               validation_data=my_gen(validation_gen),
          5                               validation_steps=5,
    ----> 6                              verbose=1)
    

    /anaconda3/lib/python3.7/site-packages/keras/legacy/interfaces.py in wrapper(*args, **kwargs)
         89                 warnings.warn('Update your `' + object_name + '` call to the ' +
         90                               'Keras 2 API: ' + signature, stacklevel=2)
    ---> 91             return func(*args, **kwargs)
         92         wrapper._original_function = func
         93         return wrapper


    /anaconda3/lib/python3.7/site-packages/keras/engine/training.py in fit_generator(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
       1416             use_multiprocessing=use_multiprocessing,
       1417             shuffle=shuffle,
    -> 1418             initial_epoch=initial_epoch)
       1419 
       1420     @interfaces.legacy_generator_methods_support


    /anaconda3/lib/python3.7/site-packages/keras/engine/training_generator.py in fit_generator(model, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
        179             batch_index = 0
        180             while steps_done < steps_per_epoch:
    --> 181                 generator_output = next(output_generator)
        182 
        183                 if not hasattr(generator_output, '__len__'):


    /anaconda3/lib/python3.7/site-packages/keras/utils/data_utils.py in get(self)
        683         try:
        684             while self.is_running():
    --> 685                 inputs = self.queue.get(block=True).get()
        686                 self.queue.task_done()
        687                 if inputs is not None:


    /anaconda3/lib/python3.7/multiprocessing/pool.py in get(self, timeout)
        649 
        650     def get(self, timeout=None):
    --> 651         self.wait(timeout)
        652         if not self.ready():
        653             raise TimeoutError


    /anaconda3/lib/python3.7/multiprocessing/pool.py in wait(self, timeout)
        646 
        647     def wait(self, timeout=None):
    --> 648         self._event.wait(timeout)
        649 
        650     def get(self, timeout=None):


    /anaconda3/lib/python3.7/threading.py in wait(self, timeout)
        550             signaled = self._flag
        551             if not signaled:
    --> 552                 signaled = self._cond.wait(timeout)
        553             return signaled
        554 


    /anaconda3/lib/python3.7/threading.py in wait(self, timeout)
        294         try:    # restore state no matter what (e.g., KeyboardInterrupt)
        295             if timeout is None:
    --> 296                 waiter.acquire()
        297                 gotit = True
        298             else:


    KeyboardInterrupt: 



```python

```


```python

```

# debugging


```python
type(train_gen.flow_from_directory("/Users/rafael/batproject/train/"))
```

    Found 760 images belonging to 2 classes.





    keras_preprocessing.image.directory_iterator.DirectoryIterator




```python
next(my_gen(train_gen.flow_from_directory("/Users/rafael/batproject/train/")))
```

    Found 760 images belonging to 2 classes.


    Exception ignored in: <generator object my_gen at 0xb31bf4a98>
    RuntimeError: generator ignored GeneratorExit





    (array([[[[1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              ...,
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ]],
     
             [[1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              ...,
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ]],
     
             [[1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              ...,
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ]],
     
             ...,
     
             [[1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              ...,
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ]],
     
             [[1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              ...,
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ]],
     
             [[1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              ...,
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ],
              [1.1333333 , 1.1333333 , 1.1333333 ]]],
     
     
            [[[0.18222223, 0.36888888, 0.        ],
              [0.2       , 0.4       , 0.        ],
              [0.23111111, 0.44      , 0.01333333],
              ...,
              [0.05777778, 0.11111111, 0.05777778],
              [0.08      , 0.12888889, 0.05777778],
              [0.08      , 0.12888889, 0.05777778]],
     
             [[0.14666668, 0.33333334, 0.        ],
              [0.16      , 0.35555556, 0.        ],
              [0.19111112, 0.4       , 0.        ],
              ...,
              [0.05777778, 0.11111111, 0.05777778],
              [0.07555556, 0.12444445, 0.05333333],
              [0.07555556, 0.12444445, 0.05333333]],
     
             [[0.14666668, 0.33333334, 0.        ],
              [0.16      , 0.35555556, 0.        ],
              [0.19111112, 0.4       , 0.        ],
              ...,
              [0.05777778, 0.11111111, 0.05777778],
              [0.07555556, 0.12444445, 0.05333333],
              [0.07555556, 0.12444445, 0.05333333]],
     
             ...,
     
             [[0.20444445, 0.48444447, 0.00888889],
              [0.19555557, 0.47555557, 0.00888889],
              [0.18222223, 0.45333335, 0.        ],
              ...,
              [0.05777778, 0.07111111, 0.04      ],
              [0.07111111, 0.04888889, 0.03555556],
              [0.07111111, 0.04888889, 0.03555556]],
     
             [[0.20444445, 0.48444447, 0.00888889],
              [0.19555557, 0.47555557, 0.00888889],
              [0.18222223, 0.45333335, 0.        ],
              ...,
              [0.05777778, 0.07111111, 0.04      ],
              [0.07111111, 0.04888889, 0.03555556],
              [0.07111111, 0.04888889, 0.03555556]],
     
             [[0.18222223, 0.46222222, 0.        ],
              [0.17333333, 0.45333335, 0.        ],
              [0.16      , 0.42666668, 0.        ],
              ...,
              [0.05333333, 0.08      , 0.03555556],
              [0.07555556, 0.05333333, 0.04      ],
              [0.07555556, 0.05333333, 0.04      ]]],
     
     
            [[[0.51555556, 0.72      , 1.0977778 ],
              [0.52444446, 0.73777777, 1.1155555 ],
              [0.54222226, 0.75555557, 1.1333333 ],
              ...,
              [0.71555555, 0.93333334, 1.0755556 ],
              [0.7644445 , 0.96444446, 1.0666667 ],
              [0.7644445 , 0.96444446, 1.0666667 ]],
     
             [[0.50222224, 0.71555555, 1.08      ],
              [0.51111114, 0.72444445, 1.0977778 ],
              [0.52444446, 0.73777777, 1.1155555 ],
              ...,
              [0.72      , 0.9377778 , 1.0711111 ],
              [0.7733334 , 0.97333336, 1.0666667 ],
              [0.7733334 , 0.97333336, 1.0666667 ]],
     
             [[0.50222224, 0.71555555, 1.08      ],
              [0.51111114, 0.72444445, 1.0977778 ],
              [0.52444446, 0.73777777, 1.1155555 ],
              ...,
              [0.72      , 0.9377778 , 1.0711111 ],
              [0.7733334 , 0.97333336, 1.0666667 ],
              [0.7733334 , 0.97333336, 1.0666667 ]],
     
             ...,
     
             [[0.32      , 0.29777777, 0.16444445],
              [0.32      , 0.29777777, 0.1688889 ],
              [0.32      , 0.29777777, 0.1688889 ],
              ...,
              [0.28      , 0.35555556, 0.28444445],
              [0.28444445, 0.34222224, 0.22666667],
              [0.28444445, 0.34222224, 0.22666667]],
     
             [[0.32      , 0.29777777, 0.16444445],
              [0.32      , 0.29777777, 0.1688889 ],
              [0.32      , 0.29777777, 0.1688889 ],
              ...,
              [0.28      , 0.35555556, 0.28444445],
              [0.28444445, 0.34222224, 0.22666667],
              [0.28444445, 0.34222224, 0.22666667]],
     
             [[0.32444444, 0.30222222, 0.1688889 ],
              [0.32444444, 0.30222222, 0.17333333],
              [0.32444444, 0.30222222, 0.17333333],
              ...,
              [0.28444445, 0.36      , 0.2888889 ],
              [0.28444445, 0.3377778 , 0.23111111],
              [0.28444445, 0.3377778 , 0.23111111]]],
     
     
            ...,
     
     
            [[[0.2488889 , 0.15111111, 0.04444445],
              [0.27111113, 0.17333333, 0.06666667],
              [0.28      , 0.18222223, 0.07555556],
              ...,
              [0.36      , 0.25777778, 0.12      ],
              [0.36      , 0.25777778, 0.12      ],
              [0.35555556, 0.25333333, 0.11555555]],
     
             [[0.2488889 , 0.15111111, 0.04444445],
              [0.27111113, 0.17333333, 0.06666667],
              [0.28      , 0.18222223, 0.07555556],
              ...,
              [0.36      , 0.25777778, 0.12      ],
              [0.36      , 0.25777778, 0.12      ],
              [0.35555556, 0.25333333, 0.11555555]],
     
             [[0.26222223, 0.1688889 , 0.04888889],
              [0.28444445, 0.19111112, 0.07111111],
              [0.2888889 , 0.19555557, 0.07555556],
              ...,
              [0.34666666, 0.24444444, 0.10666667],
              [0.34666666, 0.24444444, 0.10666667],
              [0.34666666, 0.24444444, 0.10666667]],
     
             ...,
     
             [[0.3511111 , 0.25777778, 0.17333333],
              [0.31111112, 0.22666667, 0.15111111],
              [0.2488889 , 0.17777778, 0.11111111],
              ...,
              [0.5688889 , 0.37333333, 0.25333333],
              [0.5688889 , 0.37333333, 0.25333333],
              [0.5733333 , 0.36888888, 0.25333333]],
     
             [[0.36444446, 0.25777778, 0.15111111],
              [0.34222224, 0.24444444, 0.15111111],
              [0.29333335, 0.21777779, 0.14666668],
              ...,
              [0.5733333 , 0.37777779, 0.25777778],
              [0.5733333 , 0.37777779, 0.25777778],
              [0.5777778 , 0.37333333, 0.25777778]],
     
             [[0.36444446, 0.25777778, 0.15111111],
              [0.34222224, 0.24444444, 0.15111111],
              [0.29333335, 0.21777779, 0.14666668],
              ...,
              [0.5733333 , 0.37777779, 0.25777778],
              [0.5733333 , 0.37777779, 0.25777778],
              [0.5777778 , 0.37333333, 0.25777778]]],
     
     
            [[[1.1244445 , 1.0177778 , 0.43111113],
              [1.1244445 , 1.0177778 , 0.43111113],
              [1.1244445 , 1.0177778 , 0.43111113],
              ...,
              [1.0844445 , 1.0355556 , 0.75111115],
              [1.0844445 , 1.0355556 , 0.75111115],
              [1.0844445 , 1.0355556 , 0.75111115]],
     
             [[1.1155555 , 1.008889  , 0.42222223],
              [1.1155555 , 1.008889  , 0.42222223],
              [1.1155555 , 1.008889  , 0.42222223],
              ...,
              [1.0844445 , 1.0444444 , 0.75555557],
              [1.0844445 , 1.0444444 , 0.75555557],
              [1.0844445 , 1.0444444 , 0.75555557]],
     
             [[1.1022222 , 0.9955556 , 0.4088889 ],
              [1.1066667 , 1.        , 0.41333336],
              [1.1066667 , 1.        , 0.41333336],
              ...,
              [1.0844445 , 1.0444444 , 0.75555557],
              [1.0844445 , 1.0444444 , 0.75555557],
              [1.0844445 , 1.0444444 , 0.75555557]],
     
             ...,
     
             [[1.0355556 , 0.8       , 0.29333335],
              [1.0133333 , 0.7822223 , 0.3288889 ],
              [1.0133333 , 0.7822223 , 0.3288889 ],
              ...,
              [1.0400001 , 0.8977778 , 0.5288889 ],
              [1.0400001 , 0.8977778 , 0.5288889 ],
              [1.0400001 , 0.8977778 , 0.5288889 ]],
     
             [[1.0044445 , 0.7688889 , 0.26222223],
              [0.99111116, 0.76      , 0.30666667],
              [0.99111116, 0.76      , 0.30666667],
              ...,
              [1.0266666 , 0.8844445 , 0.51555556],
              [1.0266666 , 0.8844445 , 0.51555556],
              [1.0266666 , 0.8844445 , 0.51555556]],
     
             [[1.        , 0.7644445 , 0.25777778],
              [0.96888894, 0.73777777, 0.28444445],
              [0.96888894, 0.73777777, 0.28444445],
              ...,
              [1.0133333 , 0.8666667 , 0.50666666],
              [1.0133333 , 0.8666667 , 0.50666666],
              [1.0133333 , 0.8666667 , 0.50666666]]],
     
     
            [[[0.04444445, 0.07555556, 0.        ],
              [0.        , 0.01777778, 0.        ],
              [0.02222222, 0.03111111, 0.        ],
              ...,
              [0.03555556, 0.04888889, 0.        ],
              [0.        , 0.02666667, 0.        ],
              [0.03111111, 0.06666667, 0.        ]],
     
             [[0.03555556, 0.05333333, 0.        ],
              [0.02222222, 0.02666667, 0.        ],
              [0.00444444, 0.        , 0.        ],
              ...,
              [0.00444444, 0.00444444, 0.        ],
              [0.00444444, 0.01333333, 0.        ],
              [0.00444444, 0.03111111, 0.        ]],
     
             [[0.        , 0.00888889, 0.        ],
              [0.00888889, 0.00444444, 0.        ],
              [0.00444444, 0.        , 0.04      ],
              ...,
              [0.00444444, 0.        , 0.02222222],
              [0.00888889, 0.00888889, 0.        ],
              [0.02666667, 0.04      , 0.        ]],
     
             ...,
     
             [[0.        , 0.01333333, 0.        ],
              [0.00444444, 0.00444444, 0.        ],
              [0.00444444, 0.        , 0.02222222],
              ...,
              [0.00444444, 0.        , 0.02222222],
              [0.01333333, 0.01333333, 0.        ],
              [0.02666667, 0.04      , 0.        ]],
     
             [[0.00444444, 0.03111111, 0.        ],
              [0.02222222, 0.03111111, 0.        ],
              [0.01777778, 0.01777778, 0.        ],
              ...,
              [0.00444444, 0.00444444, 0.        ],
              [0.00444444, 0.01333333, 0.        ],
              [0.00444444, 0.03111111, 0.        ]],
     
             [[0.03555556, 0.07111111, 0.        ],
              [0.        , 0.02666667, 0.        ],
              [0.01333333, 0.02666667, 0.        ],
              ...,
              [0.05777778, 0.07111111, 0.        ],
              [0.        , 0.02666667, 0.        ],
              [0.02666667, 0.06222222, 0.        ]]]], dtype=float32),
     array([[0., 1.],
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
            [1., 0.],
            [1., 0.],
            [1., 0.],
            [1., 0.],
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
            [0., 1.]], dtype=float32))



### Assessment
- `yield` - it functions like a return but stores last step
- filters - how they work 
- workflow of creating a CNN in Keras


### Still Need
- to develop an intuition behind constructing Deep Learning Models
- 


```python

```
