import numpy as np
import PIL.Image as Image
from eval import load_checkpoint
from nn import forward

def otsu(arr):
    hist,_=np.histogram(arr.flatten(),bins=256,range=(0,256))
    hist=hist.astype(float)
    total=hist.sum()
    sum_total=np.dot(np.arange(256),hist)
    best_thresh,best_val=0,0
    sum_bg,weight_bg=0.0,0.0

    for t in range(256):
        weight_bg+=hist[t]
        if weight_bg==0 or weight_bg==total:
            continue
        weight_fg=total-weight_bg
        sum_bg+=t*hist[t]
        mean_bg=sum_bg/weight_bg
        mean_fg=(sum_total-sum_bg)/weight_fg
        var=weight_fg*weight_bg*(mean_bg-mean_fg)**2
        if var>best_val:
            best_thresh=t
            best_val=var

    return best_thresh


def preprocess_image(image_path):
    img=Image.open(image_path).convert('L')
    arr=np.array(img,dtype=np.float32)

    if arr.mean()>127:
        arr=255-arr

    thresh=otsu(arr)
    arr=np.where(arr>thresh,arr,0.0)

    rows=np.any(arr>0,axis=1)
    cols=np.any(arr>0,axis=0)

    if not rows.any():
        return Image.fromarray(np.zeros((28,28),dtype=np.uint8))

    r_min,r_max=np.where(rows)[0][[0,-1]]
    c_min,c_max=np.where(cols)[0][[0,-1]]

    h=r_max-r_min+1
    w=c_max-c_min+1
    pad=int(max(h,w)*0.2)
    r_min=max(0,r_min-pad)
    c_min=max(0,c_min-pad)
    r_max=min(arr.shape[0]-1,r_max+pad)
    c_max=min(arr.shape[1]-1,c_max+pad)
    cropped=arr[r_min:r_max+1,c_min:c_max+1]

    h,w=cropped.shape
    if h>w:
        right=(h-w)//2
        left=(h-w)-right
        cropped=np.pad(cropped,((0,0),(left,right)))
    else:
        top=(w-h)//2
        bottom=(w-h)-top
        cropped=np.pad(cropped,((top,bottom),(0,0)))

    pil_20=Image.fromarray(cropped.astype(np.uint8)).resize((20,20),Image.Resampling.LANCZOS)

    canvas=np.zeros((28,28),dtype=np.uint8)
    canvas[4:24,4:24]=np.array(pil_20)

    return Image.fromarray(canvas)



def predict_image(image_path):
    params=load_checkpoint('checkpoints/final_best.npz')
    pil=preprocess_image(image_path)
    x = np.array(pil, dtype=np.float32).reshape(1, 784) / 255.0
    A2,_=forward(x,params)
    ans=np.argmax(A2)
    return ans

if __name__ == '__main__':
    path1=r'9879a277b88e10ea34ed54ad7438eae3.jpg'
    path2=r'8ccc9f4433a2ab69dd0a6e0b1197f196.jpg'
    path3=r'1c71587ec9701370c84a6af3c6fbcb7f.jpg'
    print(predict_image(path1),"\n")
    print(predict_image(path2),"\n")
    print(predict_image(path3), "\n")







