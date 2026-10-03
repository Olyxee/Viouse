"""Download the OpenCV Zoo person mask model, checking its published LFS digest."""
import hashlib
import urllib.request
from backend.captions import MODEL_PATH

DIGEST='552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24'
BASE='https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/human_segmentation_pphumanseg/'
URL='https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx'

def setup():
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    if MODEL_PATH.exists() and hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()==DIGEST:
        print('Person mask model ready');return
    data=urllib.request.urlopen(URL,timeout=90).read()
    if hashlib.sha256(data).hexdigest()!=DIGEST: raise RuntimeError('Person mask model checksum mismatch')
    license_text=urllib.request.urlopen(BASE+'LICENSE',timeout=30).read()
    MODEL_PATH.with_suffix('.LICENSE').write_bytes(license_text)
    MODEL_PATH.write_bytes(data)
    print('Verified local person mask model installed (6.2 MB)')

if __name__=='__main__': setup()
