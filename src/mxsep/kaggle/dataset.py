from mxsep.kaggle import KaggleStore


def upload_dataset(path):
    store = KaggleStore.load_store()
    store.dataset_paths[path] = path
    store.save_store()
 