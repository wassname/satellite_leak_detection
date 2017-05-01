Files:
- script_metadata.json - information on scraping script
- data.h5 contains X and y
    - X: tiff files for each band loaded into an array of shape (Leak, Bands, width, length)
    - y: True for before the leak, False for after
- data_metadata: array of metadata for each leak in X. Each contain info on leak, image, and image search

Loading:
```py
# load
metadatas = json.load(open('data_metadata.json'))
with h5py.File('data.h5','r') as h5f:
    X2 = h5f['X'][:]
    y2 = h5f['y'][:]
y
```
