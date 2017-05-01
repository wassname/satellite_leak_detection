"""
Filters for input data
"""
import numpy as np
import arrow
import logging
logger = logging.getLogger('leak_helpers.modelling.filters')


def normalise_bands(X):
    """Each band becomes -1 to 1 and 0 centered"""
    for i in range(X.shape[1]):

        x = X[:, i, :, :]
        if x.std() > 0:
            X[:, i, :, :] = (x - x.mean()) / (x.max() - x.min())
    return X


def is_not_cloudy(md, max_cover=0.3):
    if 'CLOUD_COVER' in md['image']['properties']:
        return md['image']['properties']['CLOUD_COVER'] / 100.0 < max_cover
    elif 'CLOUDY_PIXEL_PERCENTAGE' in md['image']['properties']:
        return md['image']['properties']['CLOUDY_PIXEL_PERCENTAGE'] / 100.0 < max_cover
    else:
        return True


def is_not_center_cloudy(X):
    """
    Check if the center pixel have the cloud mask on them.

    It checks the last band which is cloud in sentinel-2 and cloud/snow/etc in landsat-8.
    """
    # TODO for landsat-8 I need to look at std
    cut = int(np.ceil(X.shape[-2] / 2.0 - 1))  # if it's 25 wide cut at 12, if it's 24, at 11
    return X[:, -1, cut:-cut, cut:-cut].reshape((X.shape[0], -1)).any(-1) == False


def is_image_within(md, seconds=60 * 60 * 24):
    """Check image is within X time."""
    t_image = arrow.get(md['image']['properties']['system:time_end'] / 1000)
    t_leak = arrow.get(md['leak']['features'][0]['properties']['REPO_Date'])
    seconds_before_leak = (t_leak - t_image).total_seconds()
    return seconds_before_leak < seconds


def hash_rows(X_train):
    return [hash(X_train[i].tobytes()) for i in range(len(X_train))]


def is_not_dup(X1):
    """False for the duplicates (True for the first instance)"""
    n = hash_rows(X1)
    return np.array([(n[i] not in n[:i]) for i in range(len(n))])


def is_leak(md):
    """Check is says LEAK in one of the text fields"""
    props = md['leak']['features'][0]['properties']
    if 'WA' not in props['leak_id']:
        return True
    else:
        return np.any(["LEAK" in str(v) for v in md['leak']['features'][0]['properties'].values()])


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle


def filter_data(X_raw, y_raw, metadatas, max_cloud_cover=1, timespan_before=np.inf, random_seed=0, normalized=True, balanced_classes=True, filter_center_cloudy=False):
    # filter based on cloud and timespan
    filtr = np.array([
        [
            is_not_cloudy(metadata, max_cover=max_cloud_cover),
            is_leak(metadata),
            is_image_within(metadata, timespan_before),
        ] for metadata in metadatas
    ])
    if filter_center_cloudy:
        incc = is_not_center_cloudy(X_raw)
    else:
        incc = np.ones((len(X_raw)))
    filtr = np.hstack([
        filtr,
        incc.reshape((-1, 1)),
        is_not_dup(X_raw).reshape((-1, 1))
    ])
    filt = filtr.all(-1)

    # # QC filter
    # df_filt = pd.DataFrame(filtr, columns=['is_not_cloudy','is_image_within','is_not_center_cloudy'])
    # print('filter', df_filt.sum(0), len(df_filt))
    # print('"before" images passing time filter', df_filt.is_image_within.sum()-len(df_filt)/2)

    metadata_filtered = [metadatas[i] for i in range(len(filt)) if filt[i]]
    X = X_raw[filt]
    y = y_raw[filt]

    # now because I want balanced data I have to filter them by id now
    if balanced_classes:

        allowed_ids = set([metadata_filtered[i]['leak']['features'][0]['properties']['leak_id'] for i in range(len(y)) if y[i]])
        logger.debug('allowed_ids', len(allowed_ids))

        # there are move of class 0, so we mask them until we reach balance
        mask = np.ones((len(y)), dtype=np.bool)
        for i in range(len(mask)):
            if y[mask].mean() >= 0.5:
                # stop because it's balanced
                break
            if not y[i]:
                leak_id = metadata_filtered[i]['leak']['features'][0]['properties']['leak_id']
                # we target the onces that pair with the removed class 1's
                if leak_id not in allowed_ids:
                    mask[i] = False

        metadata_filtered = [metadata_filtered[i] for i in range(len(mask)) if mask[i]]
        X = X[mask]
        y = y[mask]
        X.shape, y.shape, len(metadata_filtered)

        if np.abs(0.5 - y.mean()) > 0.1:
            logger.error('balanced classes should have an even number of each')

    if normalized:
        X = normalise_bands(X)

    # shuffle
    return X, y, metadata_filtered


def filter_split_data(X_raw, y_raw, metadatas, max_cloud_cover=1, timespan_before=np.inf, test_fraction=0.3, val_fraction=0.3, random_seed=0, normalized=True, balanced_classes=True, filter_center_cloudy=False):
    X, y, metadata_filtered = filter_data(X_raw, y_raw, metadatas, max_cloud_cover=max_cloud_cover, timespan_before=timespan_before, random_seed=random_seed, normalized=normalized, balanced_classes=balanced_classes, filter_center_cloudy=filter_center_cloudy)

    X, y, metadata_filtered=shuffle(X, y, metadata_filtered, random_state=random_seed)

    X_train, X_test, y_train, y_test, metadata_train, metadata_test=train_test_split(
        X, y, metadata_filtered, test_size=test_fraction, random_state=random_seed)

    X_train, X_val, y_train, y_val, metadata_train, metadata_val=train_test_split(
        X_train, y_train, metadata_train, test_size=val_fraction, random_state=random_seed)
#     print(X_train.shape,y_train.shape, len(metadata_train))
#     print(X_test.shape,y_test.shape, len(metadata_test))
#     print(X_val.shape,y_val.shape, len(metadata_val))

    return X_train, y_train, metadata_train, X_val, y_val, metadata_val, X_test, y_test, metadata_test

# X_raw = np.random.random((100,10,10,10))
# y_raw = np.random.random(100)>0.5
# metadata = [dict(leak_id=i,REPO_Date=arrow.get().timestamp) for i in range(100)]
# X_train, y_train, metadata_train, X_val, y_val, metadata_val, X_test, y_test, metadata_test = filter_split_data(
#     X_raw,
#     y_raw,
#     metadatas,
#     max_cloud_cover=0.3,
#     timespan_before=60*60*24*3,
#     test_fraction=0.3,
#     random_seed=0,
#     balanced_classes=True,
#     normalized=False,
#     filter_center_cloudy=True,
# )
# print(X_train.shape,y_train.shape, len(metadata_train))
# print(X_test.shape,y_test.shape, len(metadata_test))
# print(X_val.shape,y_val.shape, len(metadata_val))
